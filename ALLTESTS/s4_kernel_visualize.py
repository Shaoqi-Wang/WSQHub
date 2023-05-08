"""

standalone file to visualize S4 kernel

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from torch import exp
from opt_einsum import contract, contract_expression
from einops import repeat, rearrange, reduce

dtype = torch.float
cdtype = torch.cfloat


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


def transition(measure, N):
    assert measure in ['legs', 'legsd', 'fourier', 'fout', 'fouwang']
    if measure == 'legs':
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
        A = np.pi * (np.diag(d, 1) - np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2 ** .5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    elif measure in ['fouwang']:
        N = 256
        freqs = np.arange(1, N, 2)
        d = np.stack([np.zeros_like(freqs), freqs], axis=-1).reshape(-1)[:-1]
        A = 1 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[1::2] = 2 ** .5
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

    else:
        raise RuntimeError('measure should be in ["legs","legsd"]')

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1 N)
    elif measure in ['fourier_diag', 'foud', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2 ** .5
        P[0] = 1
        P = P.unsqueeze(0)
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
    if (err := torch.sum((_A - _A[0, 0] * torch.eye(
            N)) ** 2) / N) > 1e-5:  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)

    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    w_im, V = torch.linalg.eigh(
        AP * -1j)  # (..., N) (..., N, N) eigh function automatically set diagonal value to zero if it is not real
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
    P_ori = P
    P = contract('ij, ...j -> ...i', V_inv, P.to(V))  # V^* P

    # return w, P, B, C, V
    return w, P, B, V, P_ori


_resolve_conj = lambda x: x.conj().resolve_conj()
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def legs(N):
    H = 5

    A, B = transition('fout', N)
    A, B = 2 * torch.from_numpy(A).float(), 2 * torch.from_numpy(B).t().float()  # nn 1n
    B = rearrange(B, '1 n->n')
    w = None

    # w, P, B, V, P_ori = nplr('fout', N)
    # A = None
    # V_inv = V.conj().t()
    # #
    C = torch.zeros(H, N).to(w)
    C[torch.arange(H), 0 + torch.arange(H)] = 1.
    #
    # A = -0.00 * P_ori.t() @ P_ori + V @ torch.diag_embed(w) @ V_inv + V.conj() @ torch.diag_embed(w.conj()) @ V.t()
    # B = (V @ B).real.to(w)

    # A = - P_ori.t() @ P_ori + V @ torch.diag_embed(w) @ V_inv + V.conj() @ torch.diag_embed(w.conj()) @ V.t()
    # B = 2 * (V @ B).real.to(w)

    # A = torch.diag_embed(w) - P.t() @ P
    # C = C @ V

    L = torch.arange(1000).to(A)  # L
    xx = torch.linspace(0, 3, 1000)

    def zoh(A, B, L, delta=0.003):
        A = delta * A
        dtA = A[None, :, :] * L[:, None, None]
        # K = C @ expm(dtA) @ B
        K = C @ expm(dtA) @ (torch.linalg.inv(A) @ (torch.from_numpy(expm(A)) - torch.eye(len(A))) @ (delta * B))
        return K.squeeze(-1)

    def bilinear(A, B, L, delta=0.003):
        eye = torch.eye(len(A))
        dA = torch.linalg.inv(eye - 0.5 * delta * A) @ (eye + 0.5 * delta * A)
        dB = torch.linalg.inv(eye - 0.5 * delta * A) @ (delta * B)

        A_L = [power(i, dA) for i in range(len(L))]
        A_L = torch.stack(A_L)

        K = C @ A_L @ dB
        return K.squeeze(-1)

    def continuous(A, B, L):
        T = torch.linspace(0, 3, len(L))
        K = C @ expm(A[None, :, :] * T[:, None, None]) @ B
        return K.squeeze(-1)

    k = continuous(A, B, L)

    plt.plot(xx.numpy(), 1 * k.real[:, :], label=['1', '2', '3', '4', '5'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    legs(256)
