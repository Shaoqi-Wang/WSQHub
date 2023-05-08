import torch
import torch.nn as nn
from opt_einsum import contract, contract_expression
from einops import repeat, reduce, rearrange
import numpy as np
import math
import torch.nn.functional as F
from functools import partial
from typing import Sequence, Mapping, Optional, Callable


def cauchy_naive(v, z, w, conj=True):
    """
    v: (..., N)
    z: (..., L)
    w: (..., N)
    returns: (..., L) \sum v/(z-w)
    """
    if conj:
        v = _conj(v)
        w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)


_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


@torch.no_grad()
def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A)  # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        if v is None:
            powers = [powers[-1] @ powers[-1]]
        else:
            powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """
    Stores a representation of and computes the SSKernel function K_L(dt, A, B, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)
    """

    @torch.no_grad()
    def _setup_C(self, L):
        """ Construct C~ from C

        Two modes are supported: go directly to length L if self.L is 1, or length is doubled
        """

        if self.L.item() == 0:
            double_length = False
        elif L > self.L.item():  # 2*int(self.L) == L:
            double_length = True
            L = self.L.item()  # Convenience for the math below
        else:
            return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: prod = -prod  # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N]  # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        self.L = 2 * self.L if double_length else self.L + L  # Preserve type/device

    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform
        This should be called everytime the internal length self.L changes """

        # Use cached if available
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L // 2 + 1:
            return self.omega, self.z

        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        # Cache if necessary
        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def __init__(
            self,
            w, P, B, C, log_dt,
            L=None,  # starting/maximum length of kernel
            lr=None,
            verbose=False,
            keops=False,
            real_type='exp',  # ['none' | 'exp' | 'relu' | sigmoid']
            real_tolerance=1e-3,
            bandlimit=None,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        A is represented by diag(w) - PP^*
        w: (S, N) diagonal part
        P: (R, S, N) low-rank part

        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature
        lr: [dict | float | None] hook to set lr of special parameters (A, B, dt)

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        R (or rank): rank of low-rank part
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.verbose = verbose
        self.keops = keops
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.real_tolerance = real_tolerance

        # Rank of low-rank correction
        self.rank = P.shape[-3]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Check different SSM inits
        assert w.size(-2) == P.size(-2) == B.size(-2)  # n_ssm
        assert self.H % w.size(0) == 0
        self.n_ssm = w.size(0)
        self.repeat = self.H // w.size(0)  # Each trainable SSM needs to be duplicated this many times

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (C, H, N)
        B = B.unsqueeze(0)  # (1, 1, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register("log_dt", log_dt, lr_dict.get('dt', lr))
        self.register("B", _c2r(B), lr_dict.get('B', lr))
        self.register("P", _c2r(P), lr_dict.get('A', lr))
        self.register("inv_w_real", self._w_init(w.real), lr_dict.get('A', lr))
        self.register("w_imag", w.imag, lr_dict.get('A', lr))

        self.l_max = L
        self.register_buffer('L', torch.tensor(0))  # Internal length

    def _w_init(self, w_real):
        w_real = torch.clamp(w_real, max=-self.real_tolerance)
        if self.real_type == 'none':
            return -w_real
        elif self.real_type == 'exp':
            return torch.log(-w_real)  # Some of the HiPPO methods have real part 0
        elif self.real_type == 'relu':
            return -w_real
        elif self.real_type == 'sigmoid':
            return torch.logit(-w_real)
        elif self.real_type == 'softplus':
            return torch.log(torch.exp(-w_real) - 1)
        else:
            raise NotImplementedError

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.real_type == 'none':
            w_real = -self.inv_w_real
        elif self.real_type == 'exp':
            w_real = -torch.exp(self.inv_w_real)
        elif self.real_type == 'relu':
            w_real = -F.relu(self.inv_w_real)
        elif self.real_type == 'sigmoid':
            w_real = -F.sigmoid(self.inv_w_real)
        elif self.real_type == 'softplus':
            w_real = -F.softplus(self.inv_w_real)
        else:
            raise NotImplementedError
        w = w_real + 1j * self.w_imag
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """

        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.L.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) frequency rate
        if L is None:
            L = round(self.L.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate * L)
        while continuous_L > self.L.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.L.item() / rate)

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj()
        w = self._w()  # (S, N) where S=n_ssm

        # Address bandlimiting
        if self.bandlimit is not None:
            freqs = w.imag.abs() / (2 * math.pi)  # (H, N)
            freqs = dt[:, None] / rate * freqs  # (H, N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Get FFT nodes of right length
        omega, z = self._omega(discrete_L, dtype=w.dtype, device=w.device, cache=(rate == 1.0))

        # Broadcast parameters to same hidden features H
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.repeat)
        P = repeat(P, 'r t n -> r (v t) n', v=self.repeat)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.repeat)
        w = repeat(w, 't n -> (v t) n', v=self.repeat)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state  # (B H N)
            sA = (
                    s * _conj(w)  # (B H N)
                    - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]

            B = torch.cat([s, B], dim=-3)  # (B+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3)  # (B+1+R, H, N)
        C = torch.cat([C, Q], dim=-3)  # (C+R, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)

        # Calculate resolvent at omega
        r = cauchy_naive(v, z, w)
        r = r * dt[None, None, :, None]  # (B+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]  # BRC
            r01 = r[: -self.rank, -self.rank:, :, :]  # BRQ
            r10 = r[-self.rank:, : -self.rank, :, :]  # PRC
            r11 = r[-self.rank:, -self.rank:, :, :]  # PRQ
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :,
                                                                                          :]  # det(1+QRP)
            s = (
                    r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                    + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                    - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                    - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (B+1, C, H, L)

        # # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (B, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :]  # (C H L) TODO why choose the last one

        return k_B, k_state

    @torch.no_grad()
    def double_length(self):
        self._setup_C(2 * self.L)

    @torch.no_grad()
    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B)  # (H N)
        P = _r2c(self.P)
        Q = P.conj()

        # Repeat w shape properly
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.repeat)
        P = repeat(P, 'r t n -> r (v t) n', v=self.repeat)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.repeat)
        w = repeat(w, 't n -> (v t) n', v=self.repeat)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (torch.eye(self.rank, dtype=w.dtype, device=w.device) + 2 * contract('r h n, h n, s h n -> h r s', Q, D,
                                                                                 P).real)  # (H R R)
        Q_D = rearrange(Q * D, 'r h n -> h r n')
        try:
            R = torch.linalg.solve(R, Q_D)  # (H R N)
        except:
            R = torch.tensor(
                np.linalg.solve(R.to(Q_D).contiguous().detach().cpu(), Q_D.contiguous().detach().cpu())).to(Q_D)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D,  # (H N)
            "R": R,  # (R H N)
            "P": P,  # (R H N)
            "Q": Q,  # (R H N)
            "B": B,  # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w,  # (H N)
        }

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C)  # Just returns a view that we use for finding dtype/device

        state = torch.eye(2 * self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)  # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, '1 h n -> h n')  # (H N)
        return dA, dB

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C)  # View used for dtype/device

        if u is None:  # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:  # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N:  # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[
                                          ..., :self.N]  # inner outer product
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y)  # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (R H N)
        P = step_params["P"]  # (R H N)
        Q = step_params["Q"]  # (R H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state)  # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state

    def _setup_step(self, mode='linear'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self.dA, self.dB = self._setup_state()

        # Calculate original C
        C = _conj(_r2c(self.C))  # (H C N)
        if self.L.item() == 0:
            dC = C
        else:
            # self.C represents C_tilde
            dA_L = power(self.L.item(), self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)

            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2),
                C.unsqueeze(-1),
            ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2 * self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else:
            raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def step(self, u, state):
        """ Must have called self._setup_step() and created state with self.default_state() before calling this """
        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y.real, new_state

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        step_mode = getattr(self, "_step_mode",
                            "dense")  # Used in default_state, which is called without _setup_step() in forward_state()
        if step_mode != 'linear':
            N *= 2

            if step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N),  # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N),  # self.dC.shape
            batch_shape + (H, N),
        )

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        self._setup_step(step_mode)
        return state


def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1, 0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')


# TODO take in 'torch' option to return torch instead of numpy, and converts the shape of B from (N, 1) to (N)
def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
        B = (-1.) ** Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'legsd':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += .5 * B * B[None, :, 0]
        B = B / 2.0
    elif measure in ['fourier_diag', 'foud']:
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        A = A - .5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2 ** .5
        B[0] = 1
        B = B[:, None]
    elif measure in ['fourier', 'fout']:
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2 ** .5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    elif measure == 'fourier_decay':
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2 ** .5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - .5 * B[:, None] * B[None, :]
        B = .5 * B[:, None]
    elif measure == 'fourier2':  # Double everything: orthonormal on [0, 1]
        freqs = 2 * np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2 ** .5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :] * 2
        B = B[:, None] * 2
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))  # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0)  # (2 N)
        P *= 2 ** (-0.5)  # Halve the rank correct just like the original matrix was halved
    elif measure == 'lagt':
        assert rank >= 1
        P = .5 ** .5 * torch.ones(1, N, dtype=dtype)
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2 ** .5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure == 'fourier_decay':
        P = torch.zeros(N)
        P[0::2] = 2 ** .5
        P[0] = 1
        P = P.unsqueeze(0)
        P = P / 2 ** .5
    elif measure == 'fourier2':
        P = torch.zeros(N)
        P[0::2] = 2 ** .5
        P[0] = 1
        P = 2 ** .5 * P.unsqueeze(0)
    elif measure in ['fourier_diag', 'foud', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    else:
        raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)  # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)  # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)  # (r N)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N) > 1e-5:  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)

    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    w_im, V = torch.linalg.eigh(AP * -1j)  # (..., N) (..., N, N) eigh function automatically set diagonal value to zero if it is not real
    if diagonalize_precision: w_im, V = w_im.to(cdtype), V.to(cdtype)
    w = w_re + 1j * w_im
    # Check: V w V^{-1} = A
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    # Only keep half of each conjugate pair
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N // 2]
    w = w_sorted[:N // 2]
    assert w[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if w[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2 ** -0.5
        V[1, -1] = 2 ** -0.5 * 1j

    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    if ((err := torch.sum((2 * _AP.real - AP) ** 2) / N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    # C = initial_C(measure, N, dtype=dtype)
    B = contract('ij, j -> i', V_inv, B.to(V))  # V^* B
    # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = contract('ij, ...j -> ...i', V_inv, P.to(V))  # V^* P

    # return w, P, B, C, V
    return w, P, B, V


def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    w, P, B, V = nplr(measure, N, R, **ssm_args)
    w = repeat(w, 'n -> s n', s=H)
    P = repeat(P, 'r n -> r s n', s=H)
    B = repeat(B, 'n -> s n', s=H)
    V = repeat(V, 'n m -> s n m', s=H)
    return w, P, B, V


combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}


def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]

    assert S % len(measures) == 0, f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    w, P, B, V = zip(
        *[ssm(measure, N, R, S // len(measures), **ssm_args) for measure in measures]
    )
    w = torch.cat(w, dim=0)  # (S N)
    P = torch.cat(P, dim=1)  # (R S N)
    B = torch.cat(B, dim=0)  # (S N)
    V = torch.cat(V, dim=0)  # (S N N)
    return w, P, B, V


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    _setup_step()
    step()
    """

    def __init__(
            self,
            H,
            N=64,
            L=None,
            measure="legs",
            rank=1,
            channels=1,
            dt_min=0.001,
            dt_max=0.1,
            deterministic=False,
            lr=None,
            mode="nplr",
            n_ssm=None,
            verbose=False,
            measure_args={},
            **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$

        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config.
        N: State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need to be adjusted and doens't affect speed much.
        L: Maximum length of convolution kernel, if known. Should work in the majority of cases even if not known.
        measure: Options for initialization of (A, B). For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        rank: Rank of low-rank correction for NPLR mode. Needs to be increased for measure "legt"
        channels: C channels turns the SSM from a 1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters instead
        dt_min, dt_max: min and max values for the step size dt (\Delta)
        mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing
        n_ssm: Number of independent trainable (A, B) SSMs, e.g. n_ssm=1 means all A/B parameters are tied across the H different instantiations of C. n_ssm=None means all H SSMs are completely independent. Generally, changing this option can save parameters but doesn't affect performance or speed much. This parameter must divide H
        lr: Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (
                    math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

        # Compute the preprocessed representation
        w, P, B, V = combination(measure, self.N, rank, self.n_ssm, **measure_args)

        # Broadcast C to have H channels
        if deterministic:
            C = torch.zeros(channels, self.n_ssm, self.N, dtype=cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C)  # V^* C
            C = repeat(C, 'c t n -> c (v t) n', v=self.n_ssm // C.size(-2)).clone().contiguous()
        else:
            C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
               and self.n_ssm % P.size(-2) == 0 \
               and self.n_ssm % w.size(-2) == 0
        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()

        if mode == "nplr":
            self.kernel = SSKernelNPLR(
                w, P, B, C,
                log_dt, L=L,
                lr=lr,
                verbose=verbose,
                **kernel_args,
            )
        else:
            raise NotImplementedError(f"{mode=} is not valid")

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)


class TransposedLN(nn.Module):
    """ LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis

    This is slow and a dedicated CUDA/Triton implementation should provide substantial end-to-end speedup
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s / s) * (x - m + self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def get_initializer(name, activation=None):
    if activation in [None, 'id', 'identity', 'linear', 'modrelu']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu'  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")


def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False,  # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract('b u ..., v u -> b v ...', x, self.weight) + self.bias.view(-1, *[1] * num_axis)
        return y


class S4(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            channels=1,
            bidirectional=False,
            # Arguments for position-wise feedforward components
            activation='gelu',
            postact='glu',
            initializer=None,
            weight_norm=False,
            hyper_act=None,
            dropout=0.0, tie_dropout=False,
            bottleneck=None,
            gate=None,
            transposed=True,
            verbose=False,
            shift=False,
            linear=False,
            # SSM Kernel arguments
            **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF
        initializer: initializer on FF
        weight_norm: weight normalization on FF
        hyper_act: use a "hypernetwork" multiplication (experimental)
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        gate: add gated activation (GSS)
        bottleneck: reduce SSM dimension (GSS)
        shift: experimental option, shouldn't affect results
        linear: Remove pointwise components so that the entire module is a linear SSM

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.H,
                transposed=self.transposed,
                initializer=initializer,
                activation=activation,
                activate=True,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=self.transposed,
                initializer=initializer,
                activation=activation,
                activate=True,
                weight_norm=weight_norm,
            )
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=self.transposed,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        if not self.linear:
            self.activation = Activation(activation)
            # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
            dropout_fn = nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform to mix features
        if not self.linear:
            self.output_linear = LinearActivation(
                self.H * self.channels,
                self.d_model * (1 if self.gate is None else self.gate),
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state)  # (C H L) (B C H L)

        # Convolution

        k_f = torch.fft.rfft(k, n=L_kernel + L)  # (C H L)
        u_f = torch.fft.rfft(u, n=L_kernel + L)  # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel + L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state  #
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        if not self.linear:
            y = self.output_linear(y)

        if self.gate is not None:
            y = self.output_gate(y * v)

        return y, next_state


class SequenceModule(nn.Module):
    """Abstract sequence model class. All models must adhere to this interface

    A SequenceModule is generally a model that transforms an input of shape
    (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output)

    REQUIRED methods and attributes
    forward, d_model, d_output: controls standard forward pass, a sequence-to-sequence transformation
    __init__ should also satisfy the following interface; see SequenceIdentity for an example
        def __init__(self, d_model, transposed=False, **kwargs)

    OPTIONAL methods
    default_state, step: allows stepping the model recurrently with a hidden state
    state_to_tensor, d_state: allows decoding from hidden state
    """

    @property
    def d_model(self):
        """Model dimension (generally same as input dimension).

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline (e.g. model backbone, encoder) to track the internal shapes of the full model.
        """
        if getattr(self, "_d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_model")
        return self._d_model

    @d_model.setter
    def d_model(self, d):
        self._d_model = d

    @property
    def d_output(self):
        """Output dimension of model.

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline (e.g. model backbone, decoder) to track the internal shapes of the full model.
        """
        if getattr(self, "_d_output", None) is None:
            raise NotImplementedError("SequenceModule instantiation must specify d_output for decoder")
        return self._d_output

    @d_output.setter
    def d_output(self, d):
        self._d_output = d

    def forward(self, x, state=None, **kwargs):
        """Forward pass of sequence model, a sequence-to-sequence transformation with an optional state.

        Generally, this should map a tensor of shape (batch, length, self.d_model) to (batch, length, self.d_output)

        Additionally, it returns a "state" which can be any additional information
        For example, RNN and SSM layers may return their hidden state,
        while some types of transformer layers (e.g. Transformer-XL) may want to pass a state as well
        """
        return x, None

    @property
    def state_to_tensor(self):
        """Returns a function mapping a state to a single tensor.

        This method should be implemented if one wants to use the hidden state instead of the output sequence for final prediction.
        Currently only used with the StateDecoder.
        """
        return lambda _: None

    @property
    def d_state(self):
        """ Returns dimension of output of self.state_to_tensor """
        return None

    def default_state(self, *batch_shape, device=None):
        """Create initial state for a batch of inputs."""

        return None

    def step(self, x, state=None, **kwargs):
        """Step the model recurrently for one step of the input sequence.

        For example, this should correspond to unrolling an RNN for one step.
        If the forward pass has signature (B, L, H1) -> (B, L, H2),
        this method should generally have signature (B, H1) -> (B, H2) with an optional recurrent state.
        """
        raise NotImplementedError


class Normalization(nn.Module):
    def __init__(
            self,
            d,
            transposed=False,  # Length dimension is -1 or -2
            _name_='layer',
            **kwargs
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == 'layer':
            self.channel = True  # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == 'instance':
            self.channel = False
            norm_args = {'affine': False, 'track_running_stats': False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(d, **norm_args)  # (True, True) performs very poorly
        elif _name_ == 'batch':
            self.channel = False
            norm_args = {'affine': True, 'track_running_stats': True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == 'group':
            self.channel = False
            self.norm = nn.GroupNorm(1, d, *kwargs)
        elif _name_ == 'none':
            self.channel = True
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        # Handle higher dimension logic
        shape = x.shape
        if self.transposed:
            x = rearrange(x, 'b d ... -> b d (...)')
        else:
            x = rearrange(x, 'b ... d -> b (...)d ')

        # The cases of LayerNorm / no normalization are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x


class Residual(nn.Module):
    """ Residual connection with constant affine weights. Can simulate standard residual, no residual, and "constant gates". """

    def __init__(self, i_layer, d_input, d_model, alpha=1.0, beta=1.0):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__()
        assert (d_input == d_model) or alpha == 0.0
        self.i_layer = i_layer
        self.d_input = d_input
        self.d_model = d_model
        self.alpha = alpha
        self.beta = beta

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, y, transposed):  # TODO documentation of transposed
        y = self.beta * y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y


class DownAvgPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=None, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        if self.expand is not None:
            self.linear = LinearActivation(
                d_input,
                d_input * expand,
                transposed=transposed,
            )

    def forward(self, x):
        if not self.transposed:
            x = rearrange(x, 'b ... d -> b d ...')

        if self.stride > 1:
            # einops appears slower than F
            if x.ndim == 3:
                x = F.avg_pool1d(x, self.stride, self.stride)
            elif x.ndim == 4:
                x = F.avg_pool2d(x, self.stride, self.stride)
            else:
                # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
                reduce_str = "b d " + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim - 2)]) \
                             + " -> b d " + " ".join([f"l{i}" for i in range(x.ndim - 2)])
                x = reduce(x, reduce_str, 'mean')

        # if self.expand > 1:
        #     x = repeat(x, 'b d ... -> b (d e) ...', e=self.expand)

        if not self.transposed:
            x = rearrange(x, 'b d ... -> b ... d')
        if self.expand is not None:
            x = self.linear(x)
        return x, None


class SequenceResidualBlock(SequenceModule):
    def __init__(
            self,
            d_input,
            i_layer=None,  # Only needs to be passed into certain residuals like Decay
            prenorm=True,
            dropout=0.0,
            tie_dropout=False,
            transposed=False,
            layer=None,  # Config for black box module
            residual=None,  # Config for residual function
            norm=None,  # Config for normalization layer
            pool=None,
            drop_path=0.,
    ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        self.layer = instantiate(registry_layer, layer, d_input)  # todo 手动实现
        self.prenorm = prenorm
        self.transposed = transposed

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = instantiate(registry_resudial, residual, i_layer, d_input, self.layer.d_output)  # todo 手动实现
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = instantiate(registry_pool, pool, self.d_residual, transposed=self.transposed)  # todo 手动实现

        # Dropout
        dropout_cls = nn.Dropout
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = nn.Identity()

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm: y = self.norm(y)

        # Black box layer
        y, state = self.layer(y, state=state, **kwargs)

        # Residual
        if self.residual is not None: y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm: y = self.norm(y)

        # Pool
        if self.pool is not None: y, _ = self.pool(y)

        return y, state


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """Convert Sequence or Mapping object to dict

    lists get converted to {0: x[0], 1: x[1], ...}
    """
    if is_list(x):
        x = {i: v for i, v in enumerate(x)}
    if is_dict(x):
        if recursive:
            return {k: to_dict(v, recursive=recursive) for k, v in x.items()}
        else:
            return dict(x)
    else:
        return x


def to_list(x, recursive=False):
    """Convert an object to list.

    If Sequence (e.g. list, tuple, Listconfig): just return it

    Special case: If non-recursive and not a list, wrap in list
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


class SequenceModel(SequenceModule):
    def __init__(
            self,
            d_model,  # Resize input (useful for deep models with residuals)
            n_layers=1,  # Number of layers
            transposed=False,  # Transpose inputs so each layer receives (batch, dim, length)
            dropout=0.0,  # Dropout parameter applied on every residual and every layer
            tie_dropout=False,  # Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
            prenorm=True,  # Pre-norm vs. post-norm
            n_repeat=1,  # Each layer is repeated n times per stage before applying pooling
            layer=None,  # Layer config, must be specified
            residual=None,  # Residual config
            norm=None,  # Normalization config (e.g. layer vs batch)
            pool=None,  # Config for pooling layer per stage
            track_norms=True,  # Log norms of each layer output
            dropinp=0.0,  # Input dropout
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.track_norms = track_norms

        # Input dropout (not really used)
        dropout_fn = nn.Dropout
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()

        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get('dropout', None) is None:
                _layer['dropout'] = dropout
            # Ensure all layers are shaped the same way
            _layer['transposed'] = transposed

        # Duplicate layers
        layers = layer * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool if (l + 1) % n_repeat == 0 else None
            block = SequenceResidualBlock(d, l + 1, prenorm=prenorm, dropout=dropout, tie_dropout=tie_dropout, transposed=transposed, layer=layer, residual=residual, norm=norm,
                                          pool=pool_cfg)
            _layers.append(block)
            d = block.d_output

        self._d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(self.d_output, transposed=self.transposed, _name_=norm)
            else:
                self.norm = Normalization(self.d_output, transposed=self.transposed, **norm)
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """ Inputs assumed to be (batch, sequence, dim) """
        if self.transposed: inputs = rearrange(inputs, 'b ... d -> b d ...')
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms: output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
            next_states.append(state)
            if self.track_norms: output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None: outputs = self.norm(outputs)

        if self.transposed: outputs = rearrange(outputs, 'b d ... -> b ... d')

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f'norm/{i}': v for i, v in metrics.items()}

        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def d_output(self):
        return self._d_output

    @d_output.setter
    def d_output(self, d):
        self._d_output = d





registry_model = {
    # Backbones from this repo
    "model": "src.models.sequence.SequenceModel",
    "unet": "src.models.sequence.SequenceUNet",
    "sashimi": "src.models.sequence.sashimi.Sashimi",
    "sashimi_standalone": "sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm": "src.models.baselines.lstm.TorchLSTM",
    "gru": "src.models.baselines.gru.TorchGRU",
    "unicornn": "src.models.baselines.unicornn.UnICORNN",
    "odelstm": "src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn": "src.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn": "src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline": "src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn": "src.models.baselines.samplernn.SampleRNN",
    # Baseline CNNs
    "ckconv": "src.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan": "src.models.baselines.wavegan.WaveGANDiscriminator",  # DEPRECATED
    "wavenet": "src.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d": "src.models.baselines.resnet.TorchVisionResnet",
    # Nonaka 1D CNN baselines
    "nonaka/resnet18": "src.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception": "src.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50": "src.models.baselines.nonaka.xresnet.xresnet1d50",
}

registry_layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "sru": "src.models.sequence.rnns.sru.SRURNN",
    "lssl": "src.models.sequence.ss.lssl.LSSL",
    "s4": "src.models.sequence.ss.s4.S4",
    "standalone": "src.models.s4.s4.S4",
    "s4d": "src.models.s4.s4d.S4D",
    "ff": "src.models.sequence.ff.FF",
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    "conv1d": "src.models.sequence.convs.conv1d.Conv1d",
    "conv2d": "src.models.sequence.convs.conv2d.Conv2d",
    "performer": "src.models.sequence.attention.linear.Performer",
    "mega": "src.models.sequence.mega.MegaBlock",
}
registry_resudial = {
    'R': Residual
}
registry_pool = {
    'pool': DownAvgPool,
}


def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
    """
    registry: Dictionary mapping names to functions or target paths (e.g. {'model': 'models.SequenceModel'})
    config: Dictionary with a '_name_' key indicating which element of the registry to grab, and kwargs to be passed into the target constructor
    wrap: wrap the target class (e.g. ema optimizer or tasks.wrap)
    *args, **kwargs: additional arguments to override the config to pass into the target constructor
    """

    # Case 1: no config
    if config is None:
        return None
    # Case 2a: string means _name_ was overloaded
    if isinstance(config, str):
        _name_ = None
        _target_ = registry[config]
        config = {}
    # Case 2b: grab the desired callable from name
    else:
        _name_ = config.pop("_name_")
        _target_ = registry[_name_]

    # Retrieve the right constructor automatically based on type
    if isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    # Instantiate object
    if wrap is not None:
        fn = wrap(fn)
    import functools
    obj = functools.partial(fn, *args, **config, **kwargs)

    # Restore _name_
    if _name_ is not None:
        config["_name_"] = _name_

    if partial:
        return obj
    else:
        return obj()
