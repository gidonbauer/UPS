import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import newton
import sys


x_min = 0.0
x_max = 2.0
t_end = 0.5
a     = 2.0
def u_analytical(x, t):
    def indicator(x, lo, hi): return (lo <= x) * (x <= hi)
    def u0(x):                return indicator(x, 0.25, 0.75)
    return u0(x - a * t)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <x input file> <u input file>", file=sys.stderr)
        sys.exit(1)

    x = np.load(sys.argv[1])
    u = np.load(sys.argv[2])
    L1_error = simpson(np.abs(u - u_analytical(x, t_end)), x=x)

    plt.figure()

    xs = np.linspace(x_min, x_max, 1000)
    plt.plot(xs, u_analytical(xs, 0.0), label=R"$u(x, 0)$")
    plt.plot(x, u, label=f"$u(x, {t_end})$")
    plt.plot(xs, u_analytical(xs, t_end), linestyle="--", label=f"$u_a(x, {t_end})$")
    plt.xlabel(R"$x$")
    plt.ylabel(R"$u(x)$")
    plt.title(f"L1 error = {L1_error:.8f}")
    plt.legend()

    # plt.text(0.5, 0.5, f"L1 error = {L1_error:.8f}", ha='center', bbox={'boxstyle': 'square', 'fill': False})

    plt.show()


if __name__ == "__main__":
    main()
