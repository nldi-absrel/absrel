from field16 import Field16
import sys
import os
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
    d[..., 0] = d[..., 1]       # crude boundary: copy neighbour
    d[..., -1] = d[..., -2]
    return d


def williamson_like_rhs(state: Field16,
                        dx: float,
                        c: float = 1.0,
                        kappa_S: float = 0.1,
                        kappa_T: float = 0.1) -> Field16:
    """
    Prototype RHS:
      - Maxwell-like evolution for a 1D Ey/Bz pair.
      - Plus *passive* 'mass' (S) and 'spin' (T) channels driven
        by EM invariants:
          S_t ~ (E^2 - B^2)
          T_x_t ~ (E * B)

    We do *not* feed S/T back into E/B yet – they just integrate
    whatever the EM field does.
    """
    rhs = Field16.zeros(state.nx)

    # 1D "photon" sector: Ey ≡ E[1], Bz ≡ B[2]
    Ey = state.E[1]
    Bz = state.B[2]

    # Maxwell subset (c=1 units by default)
    rhs.E[1] = c**2 * derivative_x(Bz, dx)
    rhs.B[2] = derivative_x(Ey, dx)

    # EM invariants in 1D:
    #   F^2 ~ (E^2 - B^2)
    #   F⋆F ~ E·B   (here E and B only have Ey,Bz)
    F_sq = Ey**2 - Bz**2       # breaks null condition
    EB   = Ey * Bz             # pseudoscalar-like invariant

    # Toy "pivot / root-mass" channel:
    # S_t = κ_S * (E^2 - B^2)
    rhs.S = kappa_S * F_sq

    # Toy "spin" channel: spin density along x (T[0])
    # T_x_t = κ_T * E_y B_z
    rhs.T[0] = kappa_T * EB

    # Everything else remains zero (for now)
    return rhs


def step_euler(state: Field16,
               dt: float,
               dx: float,
               nsteps: int,
               c: float = 1.0,
               kappa_S: float = 0.1,
               kappa_T: float = 0.1) -> Field16:
    """
    Explicit Euler time-stepping.
    """
    for _ in range(nsteps):
        rhs = williamson_like_rhs(state, dx, c=c,
                                  kappa_S=kappa_S, kappa_T=kappa_T)
        state += dt * rhs
    return state


def init_single_photon(state: Field16, x: np.ndarray) -> None:
    """
    Old initial condition:
      - Single Gaussian pulse in Ey with matching Bz.
    """
    width = 2.0
    Ey0 = np.exp(-x**2 / (2 * width**2))

    state.E[1] = Ey0
    state.B[2] = Ey0.copy()


def init_dual_photon(state: Field16, x: np.ndarray) -> None:
    """
    Dual-photon initial condition:
      - Two counter-propagating "photon-like" packets constructed
        via the F± = E ± B decomposition.

    For the 1D Maxwell system:
      F+ = E + B  travels with v = -c
      F- = E - B  travels with v = +c

    So we choose F+ as a left-moving packet, F- as a right-moving
    packet. Then:
      E = (F+ + F-) / 2
      B = (F+ - F-) / 2

    This gives one packet moving left, one moving right.
    """
    # Separation and width of the packets
    x0 = 6.0
    width = 1.0
    k0 = 6.0  # carrier wavenumber for a bit of internal structure

    def gaussian_packet(x, x0, w, k):
        # Gaussian envelope with a short-wavelength carrier
        return np.exp(-(x - x0)**2 / (2 * w**2)) * np.cos(k * (x - x0))

    # Left-moving packet (centered at -x0) => F+
    F_plus = gaussian_packet(x, -x0, width, k0)

    # Right-moving packet (centered at +x0) => F-
    F_minus = gaussian_packet(x, +x0, width, k0)

    Ey = 0.5 * (F_plus + F_minus)
    Bz = 0.5 * (F_plus - F_minus)

    state.E[1] = Ey
    state.B[2] = Bz


def main():
    print("Running abs-rel 1D Williamson demo (step 4: dual photons)...")

    # 1D grid
    nx = 400
    L = 40.0
    x = np.linspace(-L / 2, L / 2, nx)
    dx = x[1] - x[0]

    # Simulation controls
    c = 1.0
    dt = 0.4 * dx / c
    nsteps = 600

    # Choose scenario from command line:
    #   python run_demo.py           -> "dual"
    #   python run_demo.py single    -> "single"
    #   python run_demo.py dual      -> "dual"
    scenario = "dual"
    if len(sys.argv) >= 2:
        if sys.argv[1].lower().startswith("single"):
            scenario = "single"
        elif sys.argv[1].lower().startswith("dual"):
            scenario = "dual"

    # Create empty multivector field
    state = Field16.zeros(nx)

    # Initialise according to scenario
    if scenario == "single":
        print("Initial condition: single-photon Gaussian pulse")
        init_single_photon(state, x)
    else:
        print("Initial condition: dual-photon collision (left+right packets)")
        init_dual_photon(state, x)

    # Evolve
    state = step_euler(state, dt, dx, nsteps, c=c,
                       kappa_S=0.1, kappa_T=0.1)

    # Diagnostics: EM energy
    Ey = state.E[1]
    Bz = state.B[2]
    energy_density = 0.5 * (Ey**2 + Bz**2)
    total_energy = np.trapz(energy_density, x)

    print(f"Grid points: {nx}")
    print(f"Scenario:    {scenario}")
    print(f"Time step:   {dt:.4g}, steps: {nsteps}, final time: {nsteps * dt:.4g}")
    print(f"Approx total EM field energy (1D) ~ {total_energy:.6g}")
    print(f"Max |E_y| = {np.max(np.abs(Ey)):.4g}, Max |B_z| = {np.max(np.abs(Bz)):.4g}")

    # Non-EM sector norms
    def l2_norm(arr):
        return np.sqrt(np.sum(arr**2) * dx)

    print("\nNon-EM sector norms:")
    print(f"  ||S (pivot)||      = {l2_norm(state.S):.3e}")
    print(f"  ||Q (quedgehog)||  = {l2_norm(state.Q):.3e}")
    print(f"  ||A0 (time)||      = {l2_norm(state.A0):.3e}")
    print(f"  ||T0 (hedgehog)||  = {l2_norm(state.T0):.3e}")
    print(f"  ||A (space)||      = {l2_norm(state.A):.3e}")
    print(f"  ||T (spin)||       = {l2_norm(state.T):.3e}")


if __name__ == "__main__":
    main()
