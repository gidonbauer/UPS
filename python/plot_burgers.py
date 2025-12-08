import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import newton
import sys


x_min_ramp = 0.0
x_max_ramp = 2.0
t_end_ramp = 1.0
def u_analytical_ramp(x, t):
    def indicator(x, lo, hi): return (lo <= x) * (x <= hi)
    def shock_location(t):    return np.sqrt(1.0 + t)
    return x / (1.0 + t) * indicator(x, 0.0, shock_location(t))


x_min_sin = 0.0
x_max_sin = 2.0 * np.pi
t_end_sin = 0.75
def u_analytical_sin_impl(x, t):
    if np.abs(t) < 1e-8:
        return np.sin(x)
    else:
        def f(xi):
            return np.sin(xi) * t + xi - x
        xi = newton(f, x)
        return np.sin(xi)
u_analytical_sin = np.vectorize(u_analytical_sin_impl, excluded=(1,))


x_min_rarefaction = 0.0
x_max_rarefaction = 2.0
t_end_rarefaction = 0.5
def u_analytical_rarefaction_impl(x, t):
    x0 = 1.0
    x1 = 1.0 * t + x0

    if x < x0:
        return 0.0
    elif x < x1:
        return (1.0 - 0.0) / (x1 - x0) * (x - x0) if np.abs(t) > 1e-8 else 0.0
    else:
        return 1.0
u_analytical_rarefaction = np.vectorize(u_analytical_rarefaction_impl, excluded=(1,))


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <test case (0, 1, or 2)> <x input file> <u input file>", file=sys.stderr)
        sys.exit(1)

    test_case = int(sys.argv[1])
    if test_case == 0:
        x_min = x_min_ramp
        x_max = x_max_ramp
        t_end = t_end_ramp
        u_analytical = u_analytical_ramp
    elif test_case == 1:
        x_min = x_min_sin
        x_max = x_max_sin
        t_end = t_end_sin
        u_analytical = u_analytical_sin
    elif test_case == 2:
        x_min = x_min_rarefaction
        x_max = x_max_rarefaction
        t_end = t_end_rarefaction
        u_analytical = u_analytical_rarefaction
    else:
        print("Invalid test case `{test_case}`: Choices are 0, 1, or 2.", file=sys.stderr)
        sys.exit(1)

    x = np.load(sys.argv[2])
    u = np.load(sys.argv[3])
    L1_error = simpson(np.abs(u - u_analytical(x, t_end)), x=x)

    plt.figure()

    xs = np.linspace(x_min, x_max, 1000)
    plt.plot(xs, u_analytical(xs, 0.0), label=R"$u(x, 0)$")
    plt.plot(x, u, label=f"$u(x, {t_end})$")
    plt.plot(xs, u_analytical(xs, t_end), linestyle="--", label=f"$u_a(x, {t_end})$")
    plt.xlabel(R"$x$")
    plt.ylabel(R"$u(x)$")
    plt.legend()

    plt.text(0.5, 0.5, f"L1 error = {L1_error:.8f}", ha='center', bbox={'boxstyle': 'square', 'fill': False})

    plt.show()


if __name__ == "__main__":
    main()
