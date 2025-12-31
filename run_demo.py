from field16 import Field16
import sys, os
import numpy as np

print("=== run_demo.py: top-level print ===")
print("Python executable:", sys.executable)
print("This file (__file__):", __file__)
print("Current working directory:", os.getcwd())
print("====================================\n")


def derivative_x(arr: np.ndarray, dx: float) -> np.ndarray:
    """
    Simple 1D central difference derivative along x.
    Works for arr shape (..., nx).
    """
    d = np.zeros_like(arr)
    d[..., 1:-1] = (arr[..., 2:] - arr[..., :-2]) / (2.0 * dx)
    d[..., 0] = d[..., 1]       # simple copy at boundaries
    d[..., -1] = d[..., -2]
    return d


def williamson_like_rhs(state: Field16, dx: float, c: float = 1.0) -> Field16:
    rhs = Field16.zeros(state.nx)

    # --- Maxwell sector (Ey,Bz) ---
    Ey = state.E[1]
    Bz = state.B[2]

    rhs.E[1] = c**2 * derivative_x(Bz, dx)
    rhs.B[2] = derivative_x(Ey, dx)

    # --- Simple nonlinear couplings into S (pivot) and T (spin) ---

    # Poynting-like flow along x in 1D: S_x ∝ E_y * B_z
    poynting_x = Ey * Bz  # shape (nx,)

    # EM invariants (already used in diagnostics):
    # I1 = B^2 - E^2 (for the E_y, B_z components we’re using)
    I1 = Bz**2 - Ey**2
    # I2 = E ⋅ B is zero for pure Ey/Bz, so we skip it here

    # Coupling strengths (tune these SMALL to avoid instability)
    g_pivot = 1e-3   # how strongly I1 sources S
    g_spin  = 1e-3   # how strongly Poynting sources T_x

    # Source “pivot” scalar S where the field departs from null
    rhs.S += g_pivot * I1

    # Source spin trivector T_x where there is EM momentum flow
    # In your indexing: T[0] is the "0yz" component (spin around x)
    rhs.T[0] += g_spin * poynting_x

    return rhs



def step_rk4(state: Field16, dt: float, dx: float, nsteps: int, c: float = 1.0) -> Field16:
    """
    4th‑order Runge–Kutta time stepping for the first‑order system
        ∂_t state = RHS(state).

    This is stable for wave‑like systems where explicit Euler is not.
    """
    for _ in range(nsteps):
        k1 = williamson_like_rhs(state, dx, c)
        k2 = williamson_like_rhs(state + 0.5 * dt * k1, dx, c)
        k3 = williamson_like_rhs(state + 0.5 * dt * k2, dx, c)
        k4 = williamson_like_rhs(state + dt * k3, dx, c)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


# ------------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------------

def init_single_photon(state: Field16, x: np.ndarray, width: float = 2.0) -> None:
    """
    Simple Gaussian pulse in E_y and matching B_z.

    This is essentially the earlier single‑packet demo.
    """
    Ey0 = np.exp(-x**2 / (2.0 * width**2))
    state.E[1] = Ey0
    state.B[2] = Ey0.copy()


def init_dual_photon(state: Field16,
                     x: np.ndarray,
                     width: float = 1.5,
                     k0: float = 4.0,
                     sep: float = 6.0) -> None:
    """
    Dual‑photon setup: two counter‑propagating wave packets.

    Right‑moving packet (centered at -sep/2):
        E_y = f_R(x)
        B_z = +f_R(x)

    Left‑moving packet (centered at +sep/2):
        E_y = f_L(x)
        B_z = -f_L(x)

    At t = 0 we superpose them so that later they collide
    near x ~ 0.
    """
    x_left_center = -0.5 * sep
    x_right_center = +0.5 * sep

    # Local coordinates for each packet
    xl = x - x_left_center
    xr = x - x_right_center

    # Gaussian envelopes with a carrier cos(k0 x)
    f_left = np.exp(-xl**2 / (2.0 * width**2)) * np.cos(k0 * xl)
    f_right = np.exp(-xr**2 / (2.0 * width**2)) * np.cos(k0 * xr)

    Ey = f_left + f_right
    Bz = -f_left + f_right   # ensures opposite propagation directions

    state.E[1] = Ey
    state.B[2] = Bz


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def compute_em_energy(state: Field16, x: np.ndarray) -> float:
    """
    Compute 1D "total EM energy" by integrating u = (E^2 + B^2)/2.

    Uses all three components of E and B, not just Ey,Bz, so we
    automatically keep track if we later excite other EM directions.
    """
    # Sum over spatial components, shape (nx,)
    E2 = np.sum(state.E**2, axis=0)
    B2 = np.sum(state.B**2, axis=0)
    energy_density = 0.5 * (E2 + B2)

    return float(np.trapz(energy_density, x))


def compute_invariants(state: Field16) -> dict:
    """
    Compute the two standard Lorentz invariants of the EM field:

        I1 = B^2 - E^2   (∝ F_{μν} F^{μν} / 2)
        I2 = E·B         (∝ F_{μν} *F^{μν} / 4)

    Here we just look at their max and RMS magnitude over the grid.
    In the pure-photon case we expect I1 ≈ 0, I2 ≈ 0 (null field).
    """
    E2 = np.sum(state.E**2, axis=0)           # shape (nx,)
    B2 = np.sum(state.B**2, axis=0)
    I1 = B2 - E2

    # E·B at each grid point
    I2 = np.sum(state.E * state.B, axis=0)

    def stats(arr: np.ndarray):
        max_abs = float(np.max(np.abs(arr)))
        rms = float(np.sqrt(np.mean(arr**2)))
        return max_abs, rms

    I1_max, I1_rms = stats(I1)
    I2_max, I2_rms = stats(I2)

    return {
        "I1_max": I1_max,
        "I1_rms": I1_rms,
        "I2_max": I2_max,
        "I2_rms": I2_rms,
    }


def non_em_norms(state: Field16) -> dict:
    """
    Norms of the non‑EM sectors, so we can see when they start to get excited.

    Right now the RHS does not couple into these components, so they
    will stay numerically at ~0 up to roundoff; later we'll add couplings.
    """
    norms = {}

    def l2(arr: np.ndarray) -> float:
        return float(np.sqrt(np.sum(arr**2)))

    norms["S"] = l2(state.S)
    norms["Q"] = l2(state.Q)
    norms["A0"] = l2(state.A0)
    norms["T0"] = l2(state.T0)
    norms["A"] = l2(state.A)
    norms["T"] = l2(state.T)

    return norms

def compute_observables(state, x, c=1.0):
    """
    Compute simple 1D 'particle-like' observables from a Field16 snapshot.

    These are very provisional:
      - energy from E,B
      - effective mass via E = m c^2
      - a 1D 'charge-like' integral from dE/dx
      - a 'spin-like' integral from |T|
    """
    dx = x[1] - x[0]

    # EM sector
    Ey = state.E[1]      # E_y(x)
    Bz = state.B[2]      # B_z(x)

    # Energy density and total energy
    energy_density = 0.5 * (Ey**2 + Bz**2)
    total_energy = np.trapz(energy_density, x)

    # Effective mass from E = m c^2
    m_eff = total_energy / (c**2)

    # Very crude 1D 'charge density' proxy:
    # in real 3D Gauss: rho ~ div(E); here we just take d/dx of E_y as a toy.
    rho_like = np.gradient(Ey, dx)
    total_charge_like = np.trapz(rho_like, x)

    # 'Spin density' from the magnitude of the spin 3-vector T(x)
    # T has shape (3, nx)
    spin_density = np.linalg.norm(state.T, axis=0)
    total_spin_like = np.trapz(spin_density, x)

    # Optional: center of energy (useful later when we have localized lumps)
    if total_energy > 0:
        x_center = np.trapz(x * energy_density, x) / total_energy
    else:
        x_center = 0.0

    return {
        "total_energy": total_energy,
        "m_eff": m_eff,
        "total_charge_like": total_charge_like,
        "total_spin_like": total_spin_like,
        "x_center": x_center,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # Choose scenario from command line, e.g.
    #   python run_demo.py single
    #   python run_demo.py dual
    if len(sys.argv) > 1:
        scenario = sys.argv[1].lower()
    else:
        scenario = "dual"   # default

    print(f"Running abs-rel 1D Williamson demo (scenario: {scenario})...")

    # 1D grid
    nx = 400
    L = 40.0
    x = np.linspace(-L / 2.0, L / 2.0, nx)
    dx = x[1] - x[0]

    # Create empty multivector field
    state = Field16.zeros(nx)

    # Initial condition
    if scenario.startswith("single"):
        print("Initial condition: single Gaussian photon-like packet")
        init_single_photon(state, x)
    else:
        print("Initial condition: dual-photon collision (left+right packets)")
        init_dual_photon(state, x)

    # Time-stepping parameters
    c = 1.0
    # RK4 is much more stable than Euler; CFL-like safety factor
    dt = 0.25 * dx / c
    nsteps = 600

    # Evolve
    state = step_rk4(state, dt, dx, nsteps, c)

    # Basic EM diagnostics
    total_energy = compute_em_energy(state, x)
    Ey = state.E[1]
    Bz = state.B[2]

    print(f"Grid points: {nx}")
    print(f"Scenario:    {scenario}")
    print(f"Time step:   {dt:.4g}, steps: {nsteps}, final time: {nsteps * dt:.4g}")
    print(f"Approx total EM field energy (1D) ~ {total_energy:.6g}")
    print(f"Max |E_y| = {np.max(np.abs(Ey)):.4g}, Max |B_z| = {np.max(np.abs(Bz)):.4g}")

    # Non-EM sectors
    norms = non_em_norms(state)
    print("\nNon-EM sector norms:")
    print(f"  ||S (pivot)||      = {norms['S']:.3e}")
    print(f"  ||Q (quedgehog)||  = {norms['Q']:.3e}")
    print(f"  ||A0 (time)||      = {norms['A0']:.3e}")
    print(f"  ||T0 (hedgehog)||  = {norms['T0']:.3e}")
    print(f"  ||A (space)||      = {norms['A']:.3e}")
    print(f"  ||T (spin)||       = {norms['T']:.3e}")

    # EM invariants (null-field check)
    inv = compute_invariants(state)
    print("\nEM invariants (null-field diagnostics):")
    print(f"  I1 = B^2 - E^2:  max |I1| ~ {inv['I1_max']:.3e},  RMS(I1) ~ {inv['I1_rms']:.3e}")
    print(f"  I2 = E·B:        max |I2| ~ {inv['I2_max']:.3e},  RMS(I2) ~ {inv['I2_rms']:.3e}")

    # --- Existing diagnostics (energy, norms, invariants) stay as-is above ---
    # ...

    # New: compute simple 'particle-like' observables
    obs = compute_observables(state, x, c)

    print("\nField-derived observables (1D toy):")
    print(f"  Total EM energy        ~ {obs['total_energy']:.6g}")
    print(f"  Effective mass E/c^2   ~ {obs['m_eff']:.6g}")
    print(f"  'Charge-like' integral ~ {obs['total_charge_like']:.6g}")
    print(f"  'Spin-like' integral   ~ {obs['total_spin_like']:.6g}")
    print(f"  Energy centroid x_c    ~ {obs['x_center']:.6g}")


if __name__ == "__main__":
    main()
