import sys, os

print("=== run_demo.py: top-level print ===")
print("Python executable:", sys.executable)
print("This file (__file__):", __file__)
print("Current working directory:", os.getcwd())
print("====================================\n")


import numpy as np

class MultiVectorField1D:
    """
    1D version of the 16-component multivector field.

    Components:
      - pivot        : Ξ_P        (scalar)
      - time         : Ξ_0        (scalar / frequency-space root-energy)
      - space        : Ξ_i        (3-vector)
      - efield       : Ξ_i0       (3-vector, electric field-like)
      - bfield       : Ξ_jk       (3-vector, magnetic field-like)
      - spin         : Ξ_0jk      (3-vector, spin / angular momentum space)
      - hedgehog     : Ξ_123      (pseudoscalar in space)
      - quedgehog    : Ξ_0123     (full pseudoscalar / dual-mass-like)
    """

    def __init__(self, nx: int):
        self.nx = nx

        # Scalars (shape: (nx,))
        self.pivot = np.zeros(nx)      # Ξ_P
        self.time = np.zeros(nx)       # Ξ_0
        self.hedgehog = np.zeros(nx)   # Ξ_123
        self.quedgehog = np.zeros(nx)  # Ξ_0123

        # 3-vectors (shape: (3, nx))
        self.space = np.zeros((3, nx))   # Ξ_i
        self.efield = np.zeros((3, nx))  # Ξ_i0
        self.bfield = np.zeros((3, nx))  # Ξ_jk
        self.spin = np.zeros((3, nx))    # Ξ_0jk

    @classmethod
    def zeros(cls, nx: int) -> "MultiVectorField1D":
        return cls(nx)

    def copy(self) -> "MultiVectorField1D":
        new = MultiVectorField1D(self.nx)
        new.pivot = self.pivot.copy()
        new.time = self.time.copy()
        new.hedgehog = self.hedgehog.copy()
        new.quedgehog = self.quedgehog.copy()

        new.space = self.space.copy()
        new.efield = self.efield.copy()
        new.bfield = self.bfield.copy()
        new.spin = self.spin.copy()
        return new

    # Simple arithmetic so we can do state + dt * rhs
    def __iadd__(self, other: "MultiVectorField1D"):
        self.pivot += other.pivot
        self.time += other.time
        self.hedgehog += other.hedgehog
        self.quedgehog += other.quedgehog

        self.space += other.space
        self.efield += other.efield
        self.bfield += other.bfield
        self.spin += other.spin
        return self

    def __add__(self, other: "MultiVectorField1D") -> "MultiVectorField1D":
        result = self.copy()
        result += other
        return result

    def __mul__(self, scalar: float) -> "MultiVectorField1D":
        result = self.copy()
        result.pivot *= scalar
        result.time *= scalar
        result.hedgehog *= scalar
        result.quedgehog *= scalar

        result.space *= scalar
        result.efield *= scalar
        result.bfield *= scalar
        result.spin *= scalar
        return result

    __rmul__ = __mul__


def derivative_x(arr: np.ndarray, dx: float) -> np.ndarray:
    """
    Simple 1D central difference derivative along x.
    Works for arr shape (..., nx).
    """
    d = np.zeros_like(arr)
    d[..., 1:-1] = (arr[..., 2:] - arr[..., :-2]) / (2.0 * dx)
    d[..., 0] = d[..., 1]       # copy near boundary
    d[..., -1] = d[..., -2]
    return d


def williamson_like_rhs(state: MultiVectorField1D, dx: float, c: float = 1.0) -> MultiVectorField1D:
    """
    Very simple prototype: evolve one E and one B component like 1D Maxwell.

    We treat:
      E_y ≡ efield[1]
      B_z ≡ bfield[2]

    ∂_t E_y =  c^2 ∂_x B_z
    ∂_t B_z =  ∂_x E_y

    Everything else is held fixed (for now).
    """
    rhs = MultiVectorField1D.zeros(state.nx)

    Ey = state.efield[1]
    Bz = state.bfield[2]

    rhs.efield[1] = c**2 * derivative_x(Bz, dx)
    rhs.bfield[2] = derivative_x(Ey, dx)

    # Later we’ll add couplings into spin, pivot, etc.
    return rhs


def step_euler(state: MultiVectorField1D, dt: float, dx: float, nsteps: int, c: float = 1.0) -> MultiVectorField1D:
    """
    Explicit Euler time-stepping. Good enough for a first toy model.
    """
    for _ in range(nsteps):
        rhs = williamson_like_rhs(state, dx, c)
        state += dt * rhs
    return state


def main():
    print("Running abs-rel 1D demo...")

    # 1D grid
    nx = 400
    L = 40.0
    x = np.linspace(-L / 2, L / 2, nx)
    dx = x[1] - x[0]

    # Create empty multivector field
    state = MultiVectorField1D.zeros(nx)

    # Initial condition: a Gaussian pulse in E_y with matching B_z
    width = 2.0
    Ey0 = np.exp(-x**2 / (2 * width**2))
    state.efield[1] = Ey0
    state.bfield[2] = Ey0.copy()

    # Time-stepping parameters (CFL-ish)
    c = 1.0
    dt = 0.4 * dx / c  # stable-ish
    nsteps = 600

    # Evolve
    state = step_euler(state, dt, dx, nsteps, c)

    # Simple diagnostics
    Ey = state.efield[1]
    Bz = state.bfield[2]
    energy_density = 0.5 * (Ey**2 + Bz**2)  # ignoring other components for now

    total_energy = np.trapz(energy_density, x)

    print(f"Grid points: {nx}")
    print(f"Time step:   {dt:.4g}, steps: {nsteps}, final time: {nsteps * dt:.4g}")
    print(f"Approx total field energy (1D) ~ {total_energy:.6g}")
    print(f"Max |E_y| = {np.max(np.abs(Ey)):.4g}, Max |B_z| = {np.max(np.abs(Bz)):.4g}")


if __name__ == "__main__":
    main()
