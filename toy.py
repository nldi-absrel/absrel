import numpy as np

# Parameters
E0 = 1.0
lam = 1.0               # wavelength
k = 2.0 * np.pi / lam
omega = 2.0 * np.pi     # just choose units so that c = 1

# Grids
z = np.linspace(-2*lam, 2*lam, 400)
t = np.linspace(0, 1.0, 200)   # one period
Z, T = np.meshgrid(z, t)

# Fields for the two same-helicity CP beams, summed
Ex =  2 * E0 * np.cos(k*Z) * np.cos(omega*T)
Ey = -2 * E0 * np.cos(k*Z) * np.sin(omega*T)
Bx = -2 * E0 * np.sin(k*Z) * np.cos(omega*T)
By =  2 * E0 * np.sin(k*Z) * np.sin(omega*T)

# Invariants and energy density
E2 = Ex**2 + Ey**2
B2 = Bx**2 + By**2
I1 = E2 - B2          # E^2 - B^2
I2 = Ex*Bx + Ey*By    # E · B

Sz = Ex*By - Ey*Bx    # z-component of E×B; should be identically 0

print("Max |S_z|:", np.max(np.abs(Sz)))  # should be ~ 0 (up to rounding)

# Example: look at a time slice, say t index 50
idx_t = 50
import matplotlib.pyplot as plt

plt.figure()
plt.plot(z, E2[idx_t], label="E^2")
plt.plot(z, B2[idx_t], label="B^2", linestyle="--")
plt.plot(z, I1[idx_t], label="I1 = E^2 - B^2", linestyle=":")
plt.legend()
plt.xlabel("z")
plt.title("Field invariants in the standing CP collision")
plt.show()
