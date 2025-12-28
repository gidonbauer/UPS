import numpy as np
from matplotlib import pyplot as plt
import sys


def plot_orbit(filename, name, ax):
    data = np.load(filename)
    xs = data[:, 1]
    ys = data[:, 2]
    zs = data[:, 3]
    ax.plot(xs, ys,zs=zs, label=name)



def main():
    # if len(sys.argv) < 2:
    #     print(f"Usage: {sys.argv[0]} <input file>", file=sys.stderr)
    #     sys.exit(1)
    # input_file = sys.argv[1]

    fig = plt.figure(figsize=(10, 5))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(0, 0, zs=0, linestyle='', marker='o', markersize=10, color="black", label="Central attractor")
    # plot_orbit(input_file, "Orbit", ax)
    # plot_orbit("./output/Orbit-AdamsBashforth.npy", "AdamsBashforth", ax)
    # plot_orbit("./output/Orbit-ExplicitEuler.npy", "ExplicitEuler", ax)
    # plot_orbit("./output/Orbit-RungeKutta2.npy", "RungeKutta2", ax)
    plot_orbit("./output/Orbit-RungeKutta4.npy", "RungeKutta4", ax)
    # plot_orbit("./output/Orbit-SemiImplicitCrankNicolson-5.npy", "SemiImplicitCrankNicolson", ax)
    plot_orbit("./output/Orbit-LeapFrog.npy", "LeapFrog", ax)
    plot_orbit("./output/Orbit-SymplecticEuler.npy", "SymplecticEuler", ax)


    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))
    ax.set_zlim((-15, 15))
    ax.set_xlabel(R"$x$")
    ax.set_ylabel(R"$y$")
    ax.set_zlabel(R"$z$")
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    plt.show()

if __name__ == "__main__":
    main()
