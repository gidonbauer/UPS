import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson


def indicator(x, lo, hi):
  return (lo <= x) * (x <= hi)


def shock_location(t):
    return np.sqrt(1.0 + t)


def u_analytical(x, t):
  return x / (1.0 + t) * indicator(x, 0.0, shock_location(t))


x  = np.load("output/x.npy")
u0 = np.load("output/u0.npy")
u  = np.load("output/u.npy")

L1_error = simpson(np.abs(u - u_analytical(x, 1.0)), x=x)

plt.figure()

xs = np.linspace(0, 2, 1000)
# plt.plot(x, u0, label=R"$u(x, 0)$")
plt.plot(xs, u_analytical(xs, 0.0), label=R"$u(x, 0)$")
plt.plot(x, u, label=R"$u(x, 1)$")
plt.plot(xs, u_analytical(xs, 1.0), linestyle="--", label=R"$u_a(x, 1)$")
plt.xlabel(R"$x$")
plt.ylabel(R"$u(x)$")
plt.legend()

plt.text(0.35, 0.9, f"L1 error = {L1_error:.8f}", ha='center', bbox={'boxstyle': 'square', 'fill': False})

plt.show()
