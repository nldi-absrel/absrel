import numpy as np


class Field16:
    """
    1D 16‑component Clifford / Williamson field on a grid of length nx.

    Components (all real):
      - S   : scalar "pivot"          (Ξ_P)
      - Q   : pseudoscalar "quedgehog" (Ξ_0123)
      - A0  : time‑like vector        (Ξ_0)
      - T0  : spatial pseudoscalar "hedgehog" (Ξ_123)

      - A   : 3‑vector (space)        (Ξ_i)
      - E   : 3‑vector (electric‑like bivector, Ξ_i0)
      - B   : 3‑vector (magnetic‑like bivector, Ξ_jk)
      - T   : 3‑vector (spin trivector, Ξ_0jk)
    """

    def __init__(self, nx: int):
        self.nx = nx

        # Scalars (shape: (nx,))
        self.S = np.zeros(nx)   # pivot
        self.Q = np.zeros(nx)   # quedgehog
        self.A0 = np.zeros(nx)  # time-like component
        self.T0 = np.zeros(nx)  # hedgehog-like component

        # 3‑vectors / 3‑component objects (shape: (3, nx))
        self.A = np.zeros((3, nx))  # space
        self.E = np.zeros((3, nx))  # electric-like bivector
        self.B = np.zeros((3, nx))  # magnetic-like bivector
        self.T = np.zeros((3, nx))  # spin

    # --------- Constructors / basic utilities ---------

    @classmethod
    def zeros(cls, nx: int) -> "Field16":
        """Factory: a zero field of length nx."""
        return cls(nx)

    def copy(self) -> "Field16":
        """Deep copy of the field."""
        new = Field16(self.nx)

        new.S = self.S.copy()
        new.Q = self.Q.copy()
        new.A0 = self.A0.copy()
        new.T0 = self.T0.copy()

        new.A = self.A.copy()
        new.E = self.E.copy()
        new.B = self.B.copy()
        new.T = self.T.copy()
        return new

    # --------- Arithmetic: state + dt * rhs style ---------

    def __iadd__(self, other: "Field16"):
        self.S += other.S
        self.Q += other.Q
        self.A0 += other.A0
        self.T0 += other.T0

        self.A += other.A
        self.E += other.E
        self.B += other.B
        self.T += other.T
        return self

    def __add__(self, other: "Field16") -> "Field16":
        result = self.copy()
        result += other
        return result

    def __mul__(self, scalar: float) -> "Field16":
        result = self.copy()

        result.S *= scalar
        result.Q *= scalar
        result.A0 *= scalar
        result.T0 *= scalar

        result.A *= scalar
        result.E *= scalar
        result.B *= scalar
        result.T *= scalar
        return result

    __rmul__ = __mul__

    # --------- Backwards‑compatibility aliases ---------
    # These keep older code (pivot/time/space/efield/bfield/spin...) working.

    # Scalars
    @property
    def pivot(self):
        return self.S

    @pivot.setter
    def pivot(self, value):
        self.S[...] = value

    @property
    def quedgehog(self):
        return self.Q

    @quedgehog.setter
    def quedgehog(self, value):
        self.Q[...] = value

    @property
    def time(self):
        return self.A0

    @time.setter
    def time(self, value):
        self.A0[...] = value

    @property
    def hedgehog(self):
        return self.T0

    @hedgehog.setter
    def hedgehog(self, value):
        self.T0[...] = value

    # 3‑vectors
    @property
    def space(self):
        return self.A

    @space.setter
    def space(self, value):
        self.A[...] = value

    @property
    def efield(self):
        return self.E

    @efield.setter
    def efield(self, value):
        self.E[...] = value

    @property
    def bfield(self):
        return self.B

    @bfield.setter
    def bfield(self, value):
        self.B[...] = value

    @property
    def spin(self):
        return self.T

    @spin.setter
    def spin(self, value):
        self.T[...] = value
