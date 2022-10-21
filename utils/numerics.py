import numpy as np
from scipy.fftpack import fft, ifft
from scipy.sparse.linalg import spsolve
from scipy import sparse


# periodic boundary condition without end point
# # extrapolate the gradient at the boundary
def gradient_first(omega, dx, bc="one-sided"):
    nx = len(omega)
    d_omega = np.copy(omega)
    d_omega[1:nx - 1] = (omega[2:nx] - omega[0:nx - 2]) / (2 * dx)

    if bc == "periodic":
        d_omega[0] = (omega[1] - omega[nx - 1]) / (2 * dx)
        d_omega[nx - 1] = (omega[0] - omega[nx - 2]) / (2 * dx)
    else:  # one-sided gradient
        d_omega[0] = (omega[1] - omega[0]) / (dx)
        d_omega[nx - 1] = (omega[nx - 1] - omega[nx - 2]) / (dx)
    return d_omega


# periodic boundary condition without end point
# # extrapolate the gradient at the boundary
def gradient_second(omega, dx, bc="one-sided"):
    nx = len(omega)
    dd_omega = np.copy(omega)
    dd_omega[1:nx - 1] = (omega[2:nx] + omega[0:nx - 2] - 2 * omega[1:nx - 1]) / (dx ** 2)

    if bc == "periodic":
        dd_omega[0] = (omega[1] + omega[nx - 1] - 2 * omega[0]) / (dx ** 2)
        dd_omega[nx - 1] = (omega[0] + omega[nx - 2] - 2 * omega[nx - 1]) / (dx ** 2)
    else:  # one-sided gradient
        dd_omega[0] = dd_omega[1]
        dd_omega[nx - 1] = dd_omega[nx - 2]
    return dd_omega


# compute gradient from face states to cell gradients, no boundary
def gradient_first_f2c(omega, dx, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        d_omega = np.copy(omega)
        d_omega[0:nx - 1] = (omega[1:nx] - omega[0:nx - 1]) / (dx)
        d_omega[nx - 1] = (omega[0] - omega[nx - 1]) / (dx)
    else:
        d_omega = (omega[1:nx] - omega[0:nx - 1]) / (dx)
    return d_omega


# compute gradient from cell states to face gradients, no boundary
def gradient_first_c2f(omega, dx, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        d_omega = np.zeros(nx)
        d_omega[1:nx] = (omega[1:nx] - omega[0:nx - 1]) / (dx)
        d_omega[0] = (omega[0] - omega[nx - 1]) / (dx)
    else:
        d_omega = (omega[1:nx] - omega[0:nx - 1]) / (dx)
    return d_omega


# compute gradient from cell states to face gradients, no boundary
def interpolate_f2c(omega, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        c_omega = np.zeros(nx)
        c_omega[0:nx - 1] = (omega[0:nx - 1] + omega[1:nx]) / 2.0
        c_omega[nx - 1] = (omega[0] + omega[nx - 1]) / 2.0
    else:
        c_omega = (omega[0:nx - 1] + omega[1:nx]) / 2.0
    return c_omega


# compute gradient from cell states to face gradients
# extrapolate the gradient at the boundary
def interpolate_c2f(omega, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        f_omega = np.zeros(nx)
        f_omega[1:nx] = (omega[1:nx] + omega[0:nx - 1]) / 2.0
        f_omega[0] = (omega[0] + omega[nx - 1]) / 2.0
    else:
        f_omega[1:nx] = (omega[0:nx - 1] + omega[1:nx]) / 2.0
    return f_omega



def precomp_fft(N):
    k2 = np.zeros(N, dtype=np.int64)
    k2[0:N // 2] = np.arange(0, N // 2)
    k2[N - 1:N - N // 2:-1] = -np.arange(1, N // 2)

    dealiasing_filter = np.ones_like(k2)
    dealiasing_filter[np.abs(k2) >= N / 3] = 0.0

    return k2, dealiasing_filter


def gradient_fft(omega, dx, order, k2=None, dealiasing_filter=None):
    N = len(omega)
    L = dx * N

    if k2 is None or dealiasing_filter is None:
        k2, dealiasing_filter = precomp_fft(N)
    #     k2= np.zeros(N, dtype=np.int64)
    #     k2[0:N//2] = np.arange(0,N//2)
    #     k2[N-1:N-N//2:-1] = -np.arange(1,N//2)
    #     dealiasing_filter = np.ones_like(k2)
    #     dealiasing_filter[np.abs(k2) >=  N/3] = 0.0

    domega = (2 * np.pi / L) ** order * ifft((1j * k2) ** order * fft(omega) * dealiasing_filter).real

    return domega


# solve psi from q by FFT
# q_1 = dd psi_1 + F1(psi_2 - psi_1)
# q_2 = dd psi_2 + F2(psi_1 - psi_2)
def psi_fft_sol(q, F1, F2, dy, k2=None, dealiasing_filter=None):
    _, N = q.shape
    L = dy * N

    if k2 is None or dealiasing_filter is None:
        k2, dealiasing_filter = precomp_fft(N)

    #     k2= np.zeros(N)
    #     k2[0:N//2] = np.arange(0,N//2)
    #     k2[N-1:N-N//2:-1] = -np.arange(1,N//2)
    #     dealiasing_filter = np.ones_like(k2)
    #     dealiasing_filter[np.abs(k2) >=  N/3] = 0.0

    ddx = - (2 * np.pi / L) ** 2 * k2 ** 2

    q_h = np.zeros((2, N), dtype=np.complex64)
    psi_h = np.zeros((2, N), dtype=np.complex64)
    psi = np.copy(q)

    q_h[0, :], q_h[1, :] = fft(q[0, 0:N]), fft(q[1, 0:N])

    det = ddx ** 2 - (F2 + F1) * ddx

    non_zero_ind = np.logical_and((np.fabs(det) > 1e-10), dealiasing_filter)

    psi_h[0, non_zero_ind] = ((ddx[non_zero_ind] - F2) * q_h[0, non_zero_ind] - F1 * q_h[1, non_zero_ind]) / det[
        non_zero_ind]
    psi_h[1, non_zero_ind] = (-F2 * q_h[0, non_zero_ind] + (ddx[non_zero_ind] - F1) * q_h[1, non_zero_ind]) / det[
        non_zero_ind]
    #     psi_h[:,zero_ind] = 0.0

    psi[0, 0:N] = ifft(psi_h[0, :]).real
    psi[1, 0:N] = ifft(psi_h[1, :]).real

    return psi
