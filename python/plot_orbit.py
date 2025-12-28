import numpy as np
from matplotlib import pyplot as plt
import sys


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    data = np.load(input_file)
    ts = data[:, 0]
    xs = data[:, 1]
    ys = data[:, 2]
    zs = data[:, 3]

    print(f"{data.shape = }")
    print(f"{xs.shape = }")
    print(f"{xs[:10]  = }")

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(0, 0, zs=0, linestyle='', marker='o', markersize=10, color="tab:blue", label="Central attractor")
    # ax.plot(xs[0], ys[0], zs=zs[0], linestyle='', marker='*', markersize=10, color="tab:orange", label="Start")
    ax.plot(xs, ys,zs=zs, color="tab:orange", label="Orbit")

    ax.set_xlabel(R"$x$")
    ax.set_ylabel(R"$y$")
    ax.set_zlabel(R"$z$")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
