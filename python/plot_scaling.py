import pandas as pd
from matplotlib import pyplot as plt
import sys

ti2name = {
    "ExplicitEuler": "EE",
    "SemiImplicitCrankNicolson-5": "CN(5)",
    "RungeKutta2": "RK2",
    "RungeKutta4": "RK4",
}

rhs2name = {
    "FV_Godunov": "FV1",
    "FD_Upwind": "FD",
    "LaxWendroff": "LW",
    "FV_HighResolution-Minmod": "FV2(Minmod)",
    "FV_HighResolution-Superbee": "FV2(Superbee)",
    "FV_HighResolution-Koren": "FV2(Koren)",
}


def plot_all(df):
    plt.figure()

    for ((ti, rhs), sub_df) in df.groupby(["TimeIntegrator", "RHS"]):
        name = f"{ti2name[ti]}-{rhs2name[rhs]}"
        plt.loglog(sub_df['N'], sub_df['L1_error'], label=name, marker='o')

    ylim = plt.ylim()
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0])/sub_df['N'], linestyle='--', color='black')
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0]**2)/sub_df['N']**2, linestyle='-.', color='black')
    plt.ylim(ylim)

    plt.xlabel("Grid size")
    plt.ylabel("L1-error")
    plt.legend(ncols=2)

    plt.show()


def plot_rhs(df, rhs):
    plt.figure()

    for ((ti, ), sub_df) in df.loc[df["RHS"] == rhs].groupby(["TimeIntegrator"]):
        name = ti
        plt.loglog(sub_df['N'], sub_df['L1_error'], label=name, marker='o')

    ylim = plt.ylim()
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0])/sub_df['N'], linestyle='--', color='black')
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0]**2)/sub_df['N']**2, linestyle='-.', color='black')
    plt.ylim(ylim)

    plt.xlabel("Grid size")
    plt.ylabel("L1-error")
    plt.title(f"Right-hand side: {rhs}")
    plt.legend()

    plt.show()


def plot_time_integrator(df, ti):
    plt.figure()

    for ((rhs, ), sub_df) in df.loc[df["TimeIntegrator"] == ti].groupby(["RHS"]):
        name = rhs
        plt.loglog(sub_df['N'], sub_df['L1_error'], label=name, marker='o')

    ylim = plt.ylim()
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0])/sub_df['N'], linestyle='--', color='black')
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0]**2)/sub_df['N']**2, linestyle='-.', color='black')
    plt.ylim(ylim)

    plt.xlabel("Grid size")
    plt.ylabel("L1-error")
    plt.title(f"Time-integrator: {ti}")
    plt.legend()

    plt.show()


def plot_both_fixed(df, ti, rhs):
    plt.figure()

    sub_df = df.loc[(df["TimeIntegrator"] == ti) & (df["RHS"] == rhs)]
    plt.loglog(sub_df['N'], sub_df['L1_error'], marker='o')

    ylim = plt.ylim()
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0])/sub_df['N'], linestyle='--', color='black')
    plt.loglog(sub_df['N'], (sub_df['L1_error'].iloc[0] * sub_df['N'].iloc[0]**2)/sub_df['N']**2, linestyle='-.', color='black')
    plt.ylim(ylim)

    plt.xlabel("Grid size")
    plt.ylabel("L1-error")
    plt.title(f"Time-integrator: {ti}, RHS: {rhs}")

    plt.show()


def main():
    input_file = ""
    time_integrator = ""
    rhs = ""

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-ti":
            i = i+1
            assert i < len(sys.argv), "Expected argument to flag `-ti`"
            time_integrator = sys.argv[i]
        elif sys.argv[i] == "-rhs":
            i = i+1
            assert i < len(sys.argv), "Expected argument to flag `-ti`"
            rhs = sys.argv[i]
        else:
            input_file = sys.argv[i]
        i = i+1

    if len(input_file) == 0:
        print(
            f"Usage: {sys.argv[0]} [-ti time integrator] [-rhs right-hand side] <input csv file>", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_file).sort_values("N")

    if len(time_integrator) == 0 and len(rhs) == 0:
        plot_all(df)
    elif len(time_integrator) == 0:
        plot_rhs(df, rhs)
    elif len(rhs) == 0:
        plot_time_integrator(df, time_integrator)
    else:
        plot_both_fixed(df, time_integrator, rhs)


if __name__ == "__main__":
    main()
