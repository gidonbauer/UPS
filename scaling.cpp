#include "BoundaryConditions.hpp"
#include "Burgers.hpp"
#include "TimeIntegrator.hpp"
using namespace UPS;
using namespace UPS::Burgers;

// = Analytical solution ===========================================================================
constexpr auto indicator(double x, double lo, double hi) noexcept -> double {
  return static_cast<double>(lo <= x && x <= hi);
}

constexpr auto shock_location(double t) noexcept -> double { return std::sqrt(1.0 + t); }

constexpr auto u_analytical(double x, double t) noexcept -> double {
  return x / (1.0 + t) * indicator(x, 0.0, shock_location(t));
}
// = Analytical solution ===========================================================================

// = Simpson's rule to integrate a function in 1D ==================================================
[[nodiscard]] constexpr auto
simpsons_rule_1d(const Vector<double>& f, double x_min, double x_max) noexcept -> double {
  const auto N = f.extent();
  IGOR_ASSERT(N > 0 && N % 2 == 1, "n must be an odd number larger than zero");

  double res = 0;
  for (Index i = 1; i <= (N - 1) / 2; ++i) {
    res += f[2 * i - 2] + 4 * f[2 * i - 1] + f[2 * i];
  }
  const auto dx = (x_max - x_min) / static_cast<double>(N);
  return res * dx / 3;
}

// =================================================================================================
template <template <class RHS, class BCond, class AdjustTimestep> class TI,
          typename RHS,
          typename... RHSArgs>
void run_scaling_test(std::string_view ti_name,
                      std::string_view rhs_name,
                      Index N,
                      RHSArgs... rhs_args) {
  const Index NGhost = 2;
  const double x_min = 0.0;
  const double x_max = 2.0;
  const double t_end = 1.0;
  Grid grid(x_min, x_max, N, NGhost);

  RHS rhs(rhs_args...);
  // Burgers::FV_HighResolution rhs{Burgers::Limiter::SUPERBEE};
  // Burgers::FD_Upwind rhs{};

  DirichletZero bcond{};

  Burgers::AdjustTimestep adjust_timestep{0.5};

  Vector<double> u0(N, NGhost);
  for (Index i = -NGhost; i < N + NGhost; ++i) {
    u0[i] = u_analytical(grid.xm[i], 0.0);
  }
  bcond(u0);

  TI solver(grid, rhs, bcond, adjust_timestep, u0);
  // ExplicitEuler solver(grid, rhs, bcond, adjust_timestep, u0);
  // SemiImplicitCrankNicolson solver(grid, rhs, bcond, adjust_timestep, u0, 5);
  // RungeKutta2 solver(grid, rhs, bcond, adjust_timestep, u0);
  solver.solve(1.0);

  Vector<double> abs_diff(N, 0);
  for (Index i = 0; i < N; ++i) {
    abs_diff[i] = std::abs(solver.u[i] - u_analytical(grid.xm[i], t_end));
  }
  const auto L1_error = simpsons_rule_1d(abs_diff, x_min, x_max);
#pragma omp critical
  std::cout << ti_name << ',' << rhs_name << ',' << N << ',' << L1_error << '\n';
}

#define RUN_SCALING_TEST(TI, RHS, ...) run_scaling_test<TI, RHS>(#TI, #RHS, __VA_ARGS__)  // NOLINT

auto main() -> int {

  Index N = 1;
  std::cout << "TimeIntegrator" << ',' << "RHS" << ',' << 'N' << ',' << "L1_error" << '\n';
#pragma omp parallel
#pragma omp single
  for (Index i = 0; i < 4; ++i) {
    N *= 10;

#pragma omp task firstprivate(N)
    {
      RUN_SCALING_TEST(ExplicitEuler, FV_Godunov, N + 1);
      RUN_SCALING_TEST(SemiImplicitCrankNicolson, FV_Godunov, N + 1);
      RUN_SCALING_TEST(RungeKutta2, FV_Godunov, N + 1);
      RUN_SCALING_TEST(RungeKutta4, FV_Godunov, N + 1);
    }

#pragma omp task firstprivate(N)
    {
      RUN_SCALING_TEST(ExplicitEuler, FD_Upwind, N + 1);
      RUN_SCALING_TEST(SemiImplicitCrankNicolson, FD_Upwind, N + 1);
      RUN_SCALING_TEST(RungeKutta2, FD_Upwind, N + 1);
      RUN_SCALING_TEST(RungeKutta4, FD_Upwind, N + 1);
    }

#pragma omp task firstprivate(N)
    {
      RUN_SCALING_TEST(ExplicitEuler, LaxWendroff, N + 1);
      RUN_SCALING_TEST(SemiImplicitCrankNicolson, LaxWendroff, N + 1);
      RUN_SCALING_TEST(RungeKutta2, LaxWendroff, N + 1);
      RUN_SCALING_TEST(RungeKutta4, LaxWendroff, N + 1);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_HighResolution>(
          "ExplicitEuler", "FV_HighResolution(Minmod)", N + 1, Limiter::MINMOD);
      run_scaling_test<SemiImplicitCrankNicolson, FV_HighResolution>(
          "SemiImplicitCrankNicolson", "FV_HighResolution(Minmod)", N + 1, Limiter::MINMOD);
      run_scaling_test<RungeKutta2, FV_HighResolution>(
          "RungeKutta2", "FV_HighResolution(Minmod)", N + 1, Limiter::MINMOD);
      run_scaling_test<RungeKutta4, FV_HighResolution>(
          "RungeKutta4", "FV_HighResolution(Minmod)", N + 1, Limiter::MINMOD);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_HighResolution>(
          "ExplicitEuler", "FV_HighResolution(Superbee)", N + 1, Limiter::SUPERBEE);
      run_scaling_test<SemiImplicitCrankNicolson, FV_HighResolution>(
          "SemiImplicitCrankNicolson", "FV_HighResolution(Superbee)", N + 1, Limiter::SUPERBEE);
      run_scaling_test<RungeKutta2, FV_HighResolution>(
          "RungeKutta2", "FV_HighResolution(Superbee)", N + 1, Limiter::SUPERBEE);
      run_scaling_test<RungeKutta4, FV_HighResolution>(
          "RungeKutta4", "FV_HighResolution(Superbee)", N + 1, Limiter::SUPERBEE);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_HighResolution>(
          "ExplicitEuler", "FV_HighResolution(Koren)", N + 1, Limiter::KOREN);
      run_scaling_test<SemiImplicitCrankNicolson, FV_HighResolution>(
          "SemiImplicitCrankNicolson", "FV_HighResolution(Koren)", N + 1, Limiter::KOREN);
      run_scaling_test<RungeKutta2, FV_HighResolution>(
          "RungeKutta2", "FV_HighResolution(Koren)", N + 1, Limiter::KOREN);
      run_scaling_test<RungeKutta4, FV_HighResolution>(
          "RungeKutta4", "FV_HighResolution(Koren)", N + 1, Limiter::KOREN);
    }
  }
}
