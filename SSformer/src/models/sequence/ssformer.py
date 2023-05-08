"""state space attention implementation.

ss-attention is the kernel of ssformer, which specify to be used in the recurrent situation,
ss-attention uses state space mechanism initialized as s4 kernel, with exception output matrix C initialized as legendre basis at -1 and T initialized at same way too.
ss-attention uses no convolution shortcut but faithfully do recurrent job, therefore, it can't process forecasting too long.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from opt_einsum import contract, contract_expression
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
import src.models.hippo.hippo as hippo
import src.models.sequence.ss.dplr as dplr
from src.models.functional.krylov import krylov, power
import src.utils.train
from torch.nn import init


def kaiming_init(weight, bias=None):
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(bias, -bound, bound)


log = src.utils.train.get_logger(__name__)

try:  # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult

    has_cauchy_extension = True
    log.info("CUDA extension for Cauchy multiplication found.")
except:
    log.warning(
        "CUDA extension for Cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try:
    import pykeops
    from src.models.functional.cauchy import cauchy_conj
    from src.models.functional.vandermonde import log_vandermonde, log_vandermonde_transpose

    has_pykeops = True
    log.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    from src.models.functional.cauchy import cauchy_naive
    from src.models.functional.vandermonde import log_vandermonde_naive as log_vandermonde
    from src.models.functional.vandermonde import log_vandermonde_transpose_naive as log_vandermonde_transpose

    if not has_cauchy_extension:
        log.warning(
            "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for memory efficiency."
        )
    log.warning(
        "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    )

_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0, "lr": lr}
            setattr(getattr(self, name), "_optim", optim)


class SSAttentionKernel(OptimModule):
    """
    Stores a representation of and computes the SSKernel function K_L(dt, A, B, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)
    """

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
            w, P, B, C, V, log_dt,
            L=None,  # starting/maximum length of kernel
            lr=None,
            verbose=False,
            keops=False,
            real_type='exp',  # ['none' | 'exp' | 'relu' | sigmoid']
            real_tolerance=1e-3,
            bandlimit=None,
            ssfeedback=False,
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
        self.V = V

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
        self.C = _c2r(_resolve_conj(C))

        self.register("log_dt", log_dt, 0.0003)
        self.register("B", _c2r(B), 0.0003)
        self.register("P", _c2r(P), 0.0003)
        self.register("inv_w_real", self._w_init(w.real), 0.0003)
        self.register("w_imag", w.imag, 0.0003)

        self.l_max = L
        self.L = L
        # self.register_buffer('L', torch.tensor(0))  # Internal length

        # self.discretize_init()
        eval_matrix = ss.eval_legendre(np.arange(self.N)[:, None], 1 - 2 * 1).T  # (N,)
        eval_matrix *= (np.arange(self.N) + 1) ** .5 * (-1) ** np.arange(self.N)

        # T = repeat(eval_matrix, '1 n -> h n', h=self.H)
        dC = repeat(eval_matrix, '1 n -> l h n', h=self.H, l=self.L)

        dC = (torch.tensor(dC).to(self.C)) @ self.V.squeeze(0)
        # dT = (torch.tensor(T).to(dA)) @ self.V.squeeze(0)

        self.register('dC', dC, 0.0003)
        # self.register('dT', _c2r(dT), 0.0003)

        self.register("transition", torch.empty(96, 384, ), 0.0003)
        kaiming_init(self.transition)

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

    def kernel(self, state=None, rate=1.0, L=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """

        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) frequency rate
        if L is None:
            L = round(self.L / rate)

        discrete_L = round(self.L.item() / rate)

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj()
        w = self._w()  # (S, N) where S=n_ssm

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
        # 关键一步，保留状态空间信息，C是一个eye矩阵
        C = torch.repeat_interleave(C.unsqueeze(1), repeats=self.H, dim=1)
        C = torch.cat([C, Q], dim=-3)  # (C+R, H, N)
        # C = Q  # (R, H, N) or C=0

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)

        # Calculate resolvent at omega
        if has_cauchy_extension and z.dtype == torch.cfloat and not self.keops:
            r = cauchy_mult(v, z, w, symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
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

    def forward(self, origin_state, L, u=None, previous_decode=None):
        use_poswer_naive = False
        dC = _r2c(self.dC)
        if previous_decode is not None:
            # do something
            k, k_state = self.kernel(L=L)
            k_f = torch.fft.rfft(k, n=L + L)  # (N H L)
            pd_f = torch.fft.rfft(previous_decode, n=L + L)  # (B H L)
            x_f = contract('bhl,chl->bhcl', pd_f, k_f)
            x_pred = torch.fft.irfft(x_f, n=L + L)[..., :L]  # (B H N L)
            output = contract('b h n l, l h n->b h l', x_pred, dC)
            return output, 0

        if not self.training:
            if use_poswer_naive:
                pass
            else:
                # origin_state [B H N L]
                x_pred = origin_state @ self.transition.t()
            output = contract('b h n l, l h n->b h l', x_pred, dC)
            return output, 0
        else:
            assert u is not None
            assert len(u) == L

            k, k_state = self.kernel(L=L)
            k_f = torch.fft.rfft(k, n=L + L)  # (N H L)
            u_f = torch.fft.rfft(u, n=L + L)  # (B H L)
            x_f = contract('b h l,c h l->b h c l', u_f, k_f)
            # x为应该的坐标值，x_pred为模型预测的坐标值
            x = torch.fft.irfft(x_f, n=L + L)[..., :L]  # (B H N L)
            if use_poswer_naive:
                pass
            else:
                # origin_state [B H N L]
                x_pred = origin_state @ self.transition.t()
            reconstruct_loss = self.reconstruct_loss(x_pred, x)

            output = contract('b h n l, l h n->b h l', x_pred, dC)
            return output, reconstruct_loss

    def reconstruct_loss(self, outputs, label):
        return torch.sum((outputs.flatten() - label.flatten()) ** 2)

    @torch.no_grad()
    def discretize_init(self):
        """ set up discretized C and T for step function

        """

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

        E = 2.0 / dt.unsqueeze(-1) + w

        self.step_params = {
            "D": D,  # (H N)
            "R": R,  # (R H N)
            "P": P,  # (R H N)
            "Q": Q,  # (R H N)
            "B": B,  # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w,  # (H N)
        }

        C = _r2c(self.C)  # Just returns a view that we use for finding dtype/device

        state = torch.eye(self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)  # (N 1 N)
        dA = self.compute_dAB(C, D, E, P, Q, R, B, state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self.compute_dAB(C, D, E, P, Q, R, B, u=u)
        # dB = _conj(dB)  # TODO 这种做法真的可行吗？简单的删除_conj()
        dB = rearrange(dB, '1 h n -> h n')  # (H N)

        self.register('dA', _c2r(dA), 0.0003)
        self.register('dB', _c2r(dB), 0.0003)

        #  初始化了一个N，对C和T都适用才对
        eval_matrix = ss.eval_legendre(np.arange(self.N)[:, None], 1 - 2 * 1).T  # (N,)
        eval_matrix *= (np.arange(self.N) + 1) ** .5 * (-1) ** np.arange(self.N)

        T = repeat(eval_matrix, '1 n -> h n', h=self.H)
        C = repeat(eval_matrix, '1 n -> l h n', h=self.H, l=self.L)

        dC = (torch.tensor(C).to(dA)) @ self.V.squeeze(0)
        dT = (torch.tensor(T).to(dA)) @ self.V.squeeze(0)

        self.register('dC', _c2r(dC), 0.0003)
        self.register('dT', _c2r(dT), 0.0003)

    def compute_dAB(self, C, D, E, P, Q, R, B, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """

        if u is None:  # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:  # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N:  # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N]  # inner outer product
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y)  # inner outer product

        d_A_B = E * state - contract_fn(P, Q, state)  # (B H N)
        d_A_B = d_A_B + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        d_A_B = D * (d_A_B - contract_fn(P, R, d_A_B))

        return d_A_B


class SSAttention(nn.Module):
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

        w, P, B, V = dplr.combination(measure, self.N, rank, self.n_ssm, **measure_args)

        # Broadcast C to have H channels
        C = torch.eye(self.N, self.N, dtype=cdtype)
        C = repeat(C, 'c n -> c h n', h=self.H)
        C = contract('hnm, chn -> chm', V, C)  # V^* C
        # C  = torch.randn(channels, self.H, self.N // 1, dtype=cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
               and self.n_ssm % P.size(-2) == 0 \
               and self.n_ssm % w.size(-2) == 0
        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()
        self.kernel = SSAttentionKernel(
            w, P, B, C, V,
            log_dt, L=L,
            lr=lr,
            verbose=verbose,
            **kernel_args,
        )

    def forward(self, origin_state, L, u=None, previous_decode=None):
        return self.kernel(origin_state=origin_state, L=L, u=u, previous_decode=previous_decode)


def tttest():
    a = nn.BatchNorm1d(5)
    b = torch.randn(32, 5)
    print(a(b).shape)
    print(a(None))


if __name__ == '__main__':
    tttest()
