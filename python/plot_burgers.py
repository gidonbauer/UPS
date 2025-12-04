import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
import sys


def indicator(x, lo, hi):
  return (lo <= x) * (x <= hi)


def shock_location(t):
    return np.sqrt(1.0 + t)


def u_analytical(x, t):
  return x / (1.0 + t) * indicator(x, 0.0, shock_location(t))


def main():
  if len(sys.argv) < 3:
     print(f"Usage: {sys.argv[0]} <x input file> <u input file>", file=sys.stderr)
     sys.exit(1)

  x = np.load(sys.argv[1])
  u = np.load(sys.argv[2])
  L1_error = simpson(np.abs(u - u_analytical(x, 1.0)), x=x)

  plt.figure()

  xs = np.linspace(0, 2, 1000)
  plt.plot(xs, u_analytical(xs, 0.0), label=R"$u(x, 0)$")
  plt.plot(x, u, label=R"$u(x, 1)$")
  plt.plot(xs, u_analytical(xs, 1.0), linestyle="--", label=R"$u_a(x, 1)$")
  plt.xlabel(R"$x$")
  plt.ylabel(R"$u(x)$")
  plt.legend()

  plt.text(0.35, 0.9, f"L1 error = {L1_error:.8f}", ha='center', bbox={'boxstyle': 'square', 'fill': False})

  plt.show()


if __name__ == "__main__":
   main()