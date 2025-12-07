#include <charconv>
#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>
#include <Igor/Timer.hpp>

#include "BoundaryConditions.hpp"
#include "Burgers.hpp"
#include "Grid.hpp"
#include "Quadrature.hpp"
#include "TimeIntegrator.hpp"
#include "Vector.hpp"
using namespace UPS;

// =================================================================================================
#if 0
constexpr double x_min = 0.0;
constexpr double x_max = 2.0;
constexpr double t_end = 1.0;

constexpr auto indicator(double x, double lo, double hi) noexcept -> double {
  return static_cast<double>(lo <= x && x <= hi);
}

constexpr auto shock_location(double t) noexcept -> double { return std::sqrt(1.0 + t); }

constexpr auto u_analytical(double x, double t) noexcept -> double {
  return x / (1.0 + t) * indicator(x, 0.0, shock_location(t));
}

#else

constexpr double x_min = 0.0;
constexpr double x_max = 2.0 * std::numbers::pi;
constexpr double t_end = 0.75;

constexpr auto u_analytical(double x, double t) noexcept -> double {
  if (std::abs(t) < 1e-8) {
    return std::sin(x);
  } else {
    auto f     = [=](double xi) { return std::sin(xi) * t + xi - x; };
    auto dfdxi = [=](double xi) { return std::cos(xi) * t + 1.0; };

    double xi  = x;  // (x_max + x_min) / 2.0;
    Index i;
    constexpr Index MAX_ITER = 100;
    constexpr double TOL     = 1e-8;
    constexpr double RELAX   = 1.0;
    for (i = 0; i < MAX_ITER && std::abs(f(xi)) > TOL; ++i) {
      xi -= RELAX * f(xi) / dfdxi(xi);
    }
    IGOR_ASSERT(i < MAX_ITER || std::abs(f(xi)) < TOL,
                "Newton method did not converge for x={:.6e}, t={:.6e}: |f({:.6e})| = {:.6e}.",
                x,
                t,
                xi,
                std::abs(f(xi)));
    return std::sin(xi);
  }
}

#endif

// =================================================================================================
template <template <typename RHS, typename BCond, typename AdjustTimestep> class TI,
          typename RHS,
          typename... RHSArgs>
auto run_solver(Index N, RHSArgs... rhs_args) -> bool {
  Igor::ScopeTimer timer("Solver");

  const Index NGhost = 2;
  Grid grid(x_min, x_max, N, NGhost);

  RHS rhs(rhs_args...);

  DirichletZero bcond{};
  // NeumannZero bcond{};

  Burgers::AdjustTimestep adjust_timestep{0.5};

  Vector<double> u0(N, NGhost);
  for (Index i = -NGhost; i < N + NGhost; ++i) {
    // u0[i] = u_analytical(grid.xm[i], 0.0);
    u0[i] = quadrature([](double x) { return u_analytical(x, 0.0); }, grid.x[i], grid.x[i + 1]) /
            grid.dx;
  }
  bcond(u0);

  TI solver(grid, rhs, bcond, adjust_timestep, u0);

  Igor::Info("Time integration side: {}", solver.name());
  Igor::Info("Right-hand side:       {}", rhs.name());
  if (!solver.solve(t_end)) {
    Igor::Error("Solver failed.");
    return false;
  }

  {
    constexpr auto x_filename = "output/x.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(grid.xm.data() + grid.xm.nghost(), grid.xm.extent()),
                             x_filename)) {
      return false;
    }
    Igor::Info("Saved grid to `{}`", x_filename);

    constexpr auto u_filename = "output/u.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(solver.u.data() + solver.u.nghost(), solver.u.extent()),
                             u_filename)) {
      return false;
    }
    Igor::Info("Saved solution to `{}`", u_filename);
  }
  return true;
}

// =================================================================================================
template <template <typename RHS, typename BCond, typename AdjustTimestep> class TI>
auto run_solver_wrapper(std::string rhs, Index N) -> bool {
  if (rhs == "FV_Godunov") {
    return run_solver<TI, Burgers::FV_Godunov>(N);
  } else if (rhs == "FV_HighResolution-Minmod") {
    return run_solver<TI, Burgers::FV_HighResolution>(N, Burgers::Limiter::MINMOD);
  } else if (rhs == "FV_HighResolution-Superbee") {
    return run_solver<TI, Burgers::FV_HighResolution>(N, Burgers::Limiter::SUPERBEE);
  } else if (rhs == "FV_HighResolution-Koren") {
    return run_solver<TI, Burgers::FV_HighResolution>(N, Burgers::Limiter::KOREN);
  } else if (rhs == "FD_Upwind") {
    return run_solver<TI, Burgers::FD_Upwind>(N);
  } else if (rhs == "FD_Upwind2") {
    return run_solver<TI, Burgers::FD_Upwind2>(N);
  } else if (rhs == "LaxWendroff") {
    return run_solver<TI, Burgers::LaxWendroff>(N);
  } else {
    Igor::Error("Invalid right-hand side `{}`, possible choices are FV_Godunov, "
                "FV_HighResolution-Minmod, FV_HighResolution-Superbee, FV_HighResolution-Koren, "
                "FD_Upwind, FD_Upwind2, and LaxWendroff.",
                rhs);
    return false;
  }
}

// =================================================================================================
auto main(int argc, char** argv) -> int {
  if (argc < 4) {
    Igor::Error("Usage: {} <Time integrator> <Right-hand side> <number grid points>", *argv);
    return 1;
  }

  const std::string ti  = argv[1];
  const std::string rhs = argv[2];
  const char* N_as_cstr = argv[3];

  Index N               = 0;
  const auto [ptr, ec]  = std::from_chars(N_as_cstr, N_as_cstr + std::strlen(N_as_cstr), N, 10);
  if (ec != std::errc()) {
    Igor::Error("Could not parse cstring `{}` as integer.", N_as_cstr);
    return 1;
  }
  if (ptr != N_as_cstr + std::strlen(N_as_cstr)) {
    Igor::Error("Invalid input for N: {}", N_as_cstr);
    return 1;
  }
  if (N <= 0) {
    Igor::Error("Number of grid points must be greater than 0 but is {}", N);
    return 1;
  }

  if (ti == "ExplicitEuler") {
    return run_solver_wrapper<ExplicitEuler>(rhs, N) ? 0 : 1;
  } else if (ti == "SemiImplicitCrankNicolson") {
    return run_solver_wrapper<SemiImplicitCrankNicolson>(rhs, N) ? 0 : 1;
  } else if (ti == "RungeKutta2") {
    return run_solver_wrapper<RungeKutta2>(rhs, N) ? 0 : 1;
  } else if (ti == "RungeKutta4") {
    return run_solver_wrapper<RungeKutta4>(rhs, N) ? 0 : 1;
  } else {
    Igor::Error("Invalid time integrator `{}`, possible choices are ExplicitEuler, "
                "SemiImplicitCrankNicolson, RungeKutta2, and RungeKutta4.",
                ti);
    return 1;
  }
}
