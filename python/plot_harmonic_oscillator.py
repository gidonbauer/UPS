import numpy as np
from matplotlib import pyplot as plt


def pos_analytical(t, k):
    return np.cos(np.sqrt(k) * t)


def vel_analytical(t, k):
    return -np.sqrt(k) * np.sin(np.sqrt(k) * t)


def main():
    data_ab  = np.load("./output/HarmonicOscillator-AdamsBashforth.npy")
    data_ee  = np.load("./output/HarmonicOscillator-ExplicitEuler.npy")
    data_lf  = np.load("./output/HarmonicOscillator-LeapFrog.npy")
    data_rk2 = np.load("./output/HarmonicOscillator-RungeKutta2.npy")
    data_rk4 = np.load("./output/HarmonicOscillator-RungeKutta4.npy")
    data_cn  = np.load("./output/HarmonicOscillator-SemiImplicitCrankNicolson-5.npy")
    data_se  = np.load("./output/HarmonicOscillator-SymplecticEuler.npy")

    # Plot cartesian coordinates
    fig, ax = plt.subplots(nrows=2, figsize=(14, 7), layout='tight')

    def plot(data, name):
        ax[0].plot(data[:, 0], data[:, 1], label=name)
        ax[1].plot(data[:, 0], data[:, 2], label=name)
    # plot(data_ee,  "ExplicitEuler")
    # plot(data_ab,  "AdamsBashforth")
    # plot(data_rk2, "RungeKutta2")
    plot(data_rk4, "RungeKutta4")
    plot(data_cn,  "SemiImplicitCrankNicolson-5")
    plot(data_se,  "SymplecticEuler")
    plot(data_lf,  "LeapFrog")

    tend = data_ee[-1, 0]
    ts = np.linspace(0, tend, 10 * data_ee.shape[0])
    ax[0].plot(ts, pos_analytical(ts, k=1), label="Analytical", linestyle="--", color="black")
    ax[1].plot(ts, vel_analytical(ts, k=1), label="Analytical", linestyle="--", color="black")

    t_plot_begin = np.max([0.0, tend-100])
    ax[0].set_xlim((t_plot_begin , tend))
    ax[0].set_ylim((-2.5, 2.5))
    ax[0].set_xlabel("Time", fontsize=12)
    ax[0].set_ylabel("Position", fontsize=12)
    ax[0].legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=12)

    ax[1].set_xlim((t_plot_begin , tend))
    ax[1].set_ylim((-2.5, 2.5))
    ax[1].set_xlabel("Time", fontsize=12)
    ax[1].set_ylabel("Velocity", fontsize=12)
    ax[1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=12)

    plt.show()

    # Plot phase space
    fig, ax = plt.subplots(figsize=(14, 7), layout='tight')

    def plot_ps(data, name):
        # ax.plot(data[:, 1], data[:, 2], label=name)
        filt = data[:, 0] >= t_plot_begin
        ax.plot(data[filt, 1], data[filt, 2], label=name)
    # plot_ps(data_ee,  "ExplicitEuler")
    # plot_ps(data_ab,  "AdamsBashforth")
    # plot_ps(data_rk2, "RungeKutta2")
    plot_ps(data_rk4, "RungeKutta4")
    plot_ps(data_cn,  "SemiImplicitCrankNicolson-5")
    plot_ps(data_se,  "SymplecticEuler")
    plot_ps(data_lf,  "LeapFrog")

    ts = np.linspace(t_plot_begin, tend, 10 * data_ee.shape[0])
    ax.plot(pos_analytical(ts, k=1), vel_analytical(ts, k=1), label="Analytical", linestyle="--", color="black")

    t_plot_begin = np.max([0.0, tend-100])
    # ax.set_xlim((-2.5, 2.5))
    # ax.set_ylim((-2.5, 2.5))
    ax.set_aspect('equal')
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Velocity", fontsize=12)
    ax.set_title("Phase space", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=12)

    plt.show()

if __name__ == "__main__":
    main()
