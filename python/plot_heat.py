import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
import sys

a = 2.0
x_min = 0.0
x_max = 10.0


def u_analytical(x, t):
    z = [1.0, 3.0]

    res = 0.0
    for zi in z:
        res += np.cos((zi * np.pi) / (x_max - x_min) * x) * \
            np.exp(-a * ((zi * np.pi) / (x_max - x_min))**2.0 * t)
    return res


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <x input file> <u input file>", file=sys.stderr)
        sys.exit(1)

    x = np.load(sys.argv[1])
    u = np.load(sys.argv[2])
    L1_error = simpson(np.abs(u - u_analytical(x, 1.0)), x=x)

    plt.figure()

    xs = np.linspace(x_min, x_max, 1000)
    plt.plot(xs, u_analytical(xs, 0.0), label=R"$u(x, 0)$")
    plt.plot(x, u, label=R"$u(x, 1)$")
    plt.plot(xs, u_analytical(xs, 1.0), linestyle="--", label=R"$u_a(x, 1)$")
    plt.xlabel(R"$x$")
    plt.ylabel(R"$u(x)$")
    plt.legend()

    plt.text(4, 1.8, f"L1 error = {L1_error:.8f}", ha='center', bbox={
             'boxstyle': 'square', 'fill': False})

    plt.show()


if __name__ == "__main__":
    main()
