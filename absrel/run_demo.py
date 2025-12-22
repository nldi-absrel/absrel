import sys
import os
import numpy as np

from field16 import Field16

print("=== run_demo.py: top-level print ===")
print("Python executable:", sys.executable)
print("This file (__file__):", __file__)
print("Current working directory:", os.getcwd())
print("====================================\n")


# ---------- Discrete spatial operators (1D) ----------

def derivative_x(arr: np.ndarray, dx: float) -> np.ndarray:
    """
    Simple 1D central difference derivative along x.
    Works for arr shape (..., nx).
    """
    d = np.zeros_like(arr)
    d[..., 1:-1] = (arr[..., 2:] - arr[..., :-2]) / (2.0 * dx)

    # crude, but stable-ish boundary handling: copy neighbor
    d[..., 0] = d[..., 1]
    d[..., -1] = d[..., -2]
    return d


def grad_x_scalar(s: np.ndarray, dx: float) -> np.ndarray:
    """
    Gradient of a scalar field in 1D, returned as a 3-vector field.

    Only the x-component is non-zero:
        (∂_x s, 0, 0)
    """
    nx = s.shape[0]
    g = np.zeros((3, nx), dtype=s.dtype)
    g[0] = derivative_x(s, dx)
    return g


def div_x_vec(v: np.ndarray, dx: float) -> np.ndarray:
    """
    Divergence of a 3-vector field in 1D.

    For fields depending only on x, only the x-component contributes:
        ∇·v = ∂_x v_x
    """
    return derivative_x(v[0], dx)


def curl_1d(v: np.ndarray, dx: float) -> np.ndarray:
    """
    1D reduction of the 3D curl for fields that depend only on x.

    Full 3D curl is:
        curl(v)_x = ∂_y v_z - ∂_z v_y  -> 0 here
        curl(v)_y = ∂_z v_x - ∂_x v_z  -> -∂_x v_z
        curl(v)_z = ∂_x v_y - ∂_y v_x  ->  ∂_x v_y

    So in 1D (v = v(x)):
        curl(v) = (0, -∂_x v_z, ∂_x v_y)
    """
    curl = np.zeros_like(v)
    curl[1] = -derivative_x(v[2], dx)
    curl[2] = derivative_x(v[1], dx)
    return curl


# ---------- Williamson-like RHS in 1D ----------

def williamson_rhs_1d(state: Field16, dx: float, c: float = 1.0) -> Field16:
    """
    Discrete 1+1D version of the extended Maxwell / Williamson equation dΨ = 0.

    Mapping of components (following Williamson / Martin-John notation):
        S      ↔ state.pivot       (root-mass-like scalar)
        Q      ↔ state.quedgehog   (dual root-mass-like scalar)
        A0     ↔ state.time        (scalar potential)
        A      ↔ state.space       (3-vector potential)
        E      ↔ state.efield      (3-vector electric field)
        B      ↔ state.bfield      (3-vector magnetic field)
        T0     ↔ state.hedgehog    (temporal trivector / spin scalar)
        T      ↔ state.spin        (3-vector spin trivector)

    The 8 real equations from dΨ = 0 are (schematically, in natural units c=1):

        ∂_t S   = -∇·E
        ∂_t Q   = -∇·B

        ∂_t E   = -∇S   + ∇×B
        ∂_t B   = -∇Q   - ∇×E

        ∂_t A0  = -∇·A
        ∂_t A   = -∇A0  - ∇×T

        ∂_t T0  = -∇·T
        ∂_t T   = -∇T0  + ∇×A

    Here we implement these in 1D with standard vector-calculus reductions.
    For now we treat c as a simple scaling (c=1 reproduces the natural-unit form).
    """

    rhs = Field16.zeros(state.nx)

    # Rename for readability
    S   = state.pivot       # scalar
    Q   = state.quedgehog   # scalar
    A0  = state.time        # scalar
    A   = state.space       # (3, nx)
    E   = state.efield      # (3, nx)
    B   = state.bfield      # (3, nx)
    T0  = state.hedgehog    # scalar
    T   = state.spin        # (3, nx)

    # Scalars S, Q
    rhs.pivot      = -c**2 * div_x_vec(E, dx)   # ∂_t S  = -∇·E
    rhs.quedgehog  = -c**2 * div_x_vec(B, dx)   # ∂_t Q  = -∇·B

    # E and B fields
    grad_S = grad_x_scalar(S, dx)
    grad_Q = grad_x_scalar(Q, dx)
    curl_B = curl_1d(B, dx)
    curl_E = curl_1d(E, dx)

    rhs.efield = -c**2 * grad_S + c**2 * curl_B    # ∂_t E = -∇S + ∇×B
    rhs.bfield = -c**2 * grad_Q - c**2 * curl_E    # ∂_t B = -∇Q - ∇×E

    # Vector potential A0, A
    rhs.time  = -c**2 * div_x_vec(A, dx)          # ∂_t A0 = -∇·A
    rhs.space = -c**2 * grad_x_scalar(A0, dx) - c**2 * curl_1d(T, dx)

    # Spin T0, T
    rhs.hedgehog = -c**2 * div_x_vec(T, dx)       # ∂_t T0 = -∇·T
    rhs.spin     = -c**2 * grad_x_scalar(T0, dx) + c**2 * curl_1d(A, dx)

    return rhs


def step_euler(state: Field16, dt: float, dx: float, nsteps: int, c: float = 1.0) -> Field16:
    """
    Explicit Euler time-stepping using the Williamson-like RHS.

    This is numerically crude but fine for a toy model. Once this is
    stable you can upgrade to leapfrog / RK4.
    """
    for _ in range(nsteps):
        rhs = williamson_rhs_1d(state, dx, c)
        # Field16 implements scalar multiplication and +=
        state += dt * rhs
    return state


# ---------- Main demo ----------

def main():
    print("Running abs-rel 1D Williamson demo...")

    # 1D grid
    nx = 400
    L  = 40.0
    x  = np.linspace(-L / 2, L / 2, nx)
    dx = x[1] - x[0]

    # Create empty 16-component multivector field
    state = Field16.zeros(nx)

    # Initial EM condition: Gaussian pulse in E_y with matching B_z
    width = 2.0
    Ey0 = np.exp(-x**2 / (2 * width**2))

    # In our indexing convention:
    #   efield[0] = E_x
    #   efield[1] = E_y
    #   efield[2] = E_z
    #   bfield[0] = B_x
    #   bfield[1] = B_y
    #   bfield[2] = B_z
    state.efield[1] = Ey0
    state.bfield[2] = Ey0.copy()

    # All pivot, spin, potentials etc start at zero – this keeps us
    # in the "pure photon" subspace at t=0.

    # Time-stepping parameters
    c = 1.0
    dt = 0.3 * dx / c   # conservative-ish CFL
    nsteps = 600

    # Evolve
    state = step_euler(state, dt, dx, nsteps, c)

    # Diagnostics focusing on the EM subset for now
    Ey = state.efield[1]
    Bz = state.bfield[2]

    energy_density = 0.5 * (Ey**2 + Bz**2)  # ignore other components for now
    total_energy = np.trapz(energy_density, x)

    print(f"Grid points: {nx}")
    print(f"Time step:   {dt:.4g}, steps: {nsteps}, final time: {nsteps * dt:.4g}")
    print(f"Approx total EM field energy (1D) ~ {total_energy:.6g}")
    print(f"Max |E_y| = {np.max(np.abs(Ey)):.4g}, Max |B_z| = {np.max(np.abs(Bz)):.4g}")

    # Optional: quick check on how much we've excited the extra sectors
    S_norm   = np.linalg.norm(state.pivot)
    Q_norm   = np.linalg.norm(state.quedgehog)
    A0_norm  = np.linalg.norm(state.time)
    T0_norm  = np.linalg.norm(state.hedgehog)
    A_norm   = np.linalg.norm(state.space)
    T_norm   = np.linalg.norm(state.spin)

    print("\nNon-EM sector norms (should be ~0 for pure photon initial data):")
    print(f"  ||S (pivot)||      = {S_norm:.3e}")
    print(f"  ||Q (quedgehog)||  = {Q_norm:.3e}")
    print(f"  ||A0 (time)||      = {A0_norm:.3e}")
    print(f"  ||T0 (hedgehog)||  = {T0_norm:.3e}")
    print(f"  ||A (space)||      = {A_norm:.3e}")
    print(f"  ||T (spin)||       = {T_norm:.3e}")


if __name__ == "__main__":
    main()
