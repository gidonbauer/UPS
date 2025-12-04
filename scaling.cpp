#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>

#include "BoundaryConditions.hpp"
#include "Burgers.hpp"
#include "Quadrature.hpp"
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
  IGOR_ASSERT(f.extent() > 0 && f.extent() % 2 == 1,
              "Number of points must be an odd number larger than zero");
  const auto N = f.extent() - 1;  // Number of intervals

  double res   = 0;
  for (Index i = 1; i <= N / 2; ++i) {
    res += f[2 * i - 2] + 4 * f[2 * i - 1] + f[2 * i];
  }
  const auto dx = (x_max - x_min) / static_cast<double>(N);
  return res * dx / 3;
}

// =================================================================================================
template <template <class RHS, class BCond, class AdjustTimestep> class TI,
          typename RHS,
          typename... RHSArgs>
void run_scaling_test(Index N, RHSArgs... rhs_args) {
  const Index NGhost = 2;
  const double x_min = 0.0;
  const double x_max = 2.0;
  const double t_end = 1.0;
  Grid grid(x_min, x_max, N, NGhost);

  RHS rhs(rhs_args...);
  DirichletZero bcond{};
  Burgers::AdjustTimestep adjust_timestep{0.5};

  Vector<double> u0(N, NGhost);
  for (Index i = -NGhost; i < N + NGhost; ++i) {
    u0[i] = quadrature([](double x) { return u_analytical(x, 0.0); }, grid.x[i], grid.x[i + 1]) /
            grid.dx;
  }
  bcond(u0);

  TI solver(grid, rhs, bcond, adjust_timestep, u0);
  solver.solve(1.0);

  Vector<double> abs_diff(N, 0);
  for (Index i = 0; i < N; ++i) {
    abs_diff[i] = std::abs(solver.u[i] - u_analytical(grid.xm[i], t_end));
  }
  const auto L1_error = simpsons_rule_1d(abs_diff, x_min + 0.5 * grid.dx, x_max - 0.5 * grid.dx);

#define SAVE_SOLUTION
#ifdef SAVE_SOLUTION
  const auto x_filename =
      Igor::detail::format("output/x-{}-{}-{}.npy", solver.name(), rhs.name(), N);
  if (!Igor::mdspan_to_npy(std::mdspan(grid.xm.data() + grid.xm.nghost(), grid.xm.extent()),
                           x_filename)) {
    Igor::Error("Could not save grid to `{}`", x_filename);
  }

  const auto u_filename =
      Igor::detail::format("output/u-{}-{}-{}.npy", solver.name(), rhs.name(), N);
  if (!Igor::mdspan_to_npy(std::mdspan(solver.u.data() + solver.u.nghost(), solver.u.extent()),
                           u_filename)) {
    Igor::Error("Could not save solution to `{}`", x_filename);
  }
#endif

#pragma omp critical
  std::cout << solver.name() << ',' << rhs.name() << ',' << N << ',' << L1_error << '\n';
}

auto main() -> int {

  Index N = 1;
  std::cout << "TimeIntegrator" << ',' << "RHS" << ',' << 'N' << ',' << "L1_error" << '\n';
#pragma omp parallel
#pragma omp single
  for (Index i = 0; i < 4; ++i) {
    N *= 10;

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_Godunov>(N + 1);
      run_scaling_test<SemiImplicitCrankNicolson, FV_Godunov>(N + 1);
      run_scaling_test<RungeKutta2, FV_Godunov>(N + 1);
      run_scaling_test<RungeKutta4, FV_Godunov>(N + 1);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FD_Upwind>(N + 1);
      run_scaling_test<SemiImplicitCrankNicolson, FD_Upwind>(N + 1);
      run_scaling_test<RungeKutta2, FD_Upwind>(N + 1);
      run_scaling_test<RungeKutta4, FD_Upwind>(N + 1);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, LaxWendroff>(N + 1);
      run_scaling_test<SemiImplicitCrankNicolson, LaxWendroff>(N + 1);
      run_scaling_test<RungeKutta2, LaxWendroff>(N + 1);
      run_scaling_test<RungeKutta4, LaxWendroff>(N + 1);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_HighResolution>(N + 1, Limiter::MINMOD);
      run_scaling_test<SemiImplicitCrankNicolson, FV_HighResolution>(N + 1, Limiter::MINMOD);
      run_scaling_test<RungeKutta2, FV_HighResolution>(N + 1, Limiter::MINMOD);
      run_scaling_test<RungeKutta4, FV_HighResolution>(N + 1, Limiter::MINMOD);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_HighResolution>(N + 1, Limiter::SUPERBEE);
      run_scaling_test<SemiImplicitCrankNicolson, FV_HighResolution>(N + 1, Limiter::SUPERBEE);
      run_scaling_test<RungeKutta2, FV_HighResolution>(N + 1, Limiter::SUPERBEE);
      run_scaling_test<RungeKutta4, FV_HighResolution>(N + 1, Limiter::SUPERBEE);
    }

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler, FV_HighResolution>(N + 1, Limiter::KOREN);
      run_scaling_test<SemiImplicitCrankNicolson, FV_HighResolution>(N + 1, Limiter::KOREN);
      run_scaling_test<RungeKutta2, FV_HighResolution>(N + 1, Limiter::KOREN);
      run_scaling_test<RungeKutta4, FV_HighResolution>(N + 1, Limiter::KOREN);
    }
  }
}
