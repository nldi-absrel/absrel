import numpy as np

class Field16:
    """
    1D version of a 16-component Clifford multivector field on a grid of length nx.

    Components:
      - pivot      : Ξ_P        (scalar)
      - time       : Ξ_0        (scalar / root-energy-like)
      - space      : Ξ_i        (3-vector, spacelike)
      - efield     : Ξ_i0       (3-vector, electric-field-like)
      - bfield     : Ξ_jk       (3-vector, magnetic-field-like)
      - spin       : Ξ_0jk      (3-vector, spin / angular-momentum-like)
      - hedgehog   : Ξ_123      (pseudoscalar in space)
      - quedgehog  : Ξ_0123     (full pseudoscalar / dual-mass-like)
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

    # --- Constructors -------------------------------------------------------
    @classmethod
    def zeros(cls, nx: int) -> "Field16":
        """Create a Field16 with all components set to zero."""
        return cls(nx)

    @classmethod
    def zeros_like(cls, other: "Field16") -> "Field16":
        """Create a zero Field16 with the same grid size as another field."""
        return cls(other.nx)

    # --- Copy ---------------------------------------------------------------
    def copy(self) -> "Field16":
        new = Field16(self.nx)
        new.pivot = self.pivot.copy()
        new.time = self.time.copy()
        new.hedgehog = self.hedgehog.copy()
        new.quedgehog = self.quedgehog.copy()

        new.space = self.space.copy()
        new.efield = self.efield.copy()
        new.bfield = self.bfield.copy()
        new.spin = self.spin.copy()
        return new

    # --- Arithmetic so we can do state + dt * rhs --------------------------
    def __iadd__(self, other: "Field16"):
        self.pivot += other.pivot
        self.time += other.time
        self.hedgehog += other.hedgehog
        self.quedgehog += other.quedgehog

        self.space += other.space
        self.efield += other.efield
        self.bfield += other.bfield
        self.spin += other.spin
        return self

    def __add__(self, other: "Field16") -> "Field16":
        result = self.copy()
        result += other
        return result

    def __mul__(self, scalar: float) -> "Field16":
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
