#include <charconv>

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
constexpr auto indicator(double x, double lo, double hi) noexcept -> double {
  return static_cast<double>(lo <= x && x <= hi);
}

constexpr auto shock_location(double t) noexcept -> double { return std::sqrt(1.0 + t); }

constexpr auto u_analytical(double x, double t) noexcept -> double {
  return x / (1.0 + t) * indicator(x, 0.0, shock_location(t));
}

// =================================================================================================
auto main(int argc, char** argv) -> int {
  Igor::ScopeTimer timer("Solver");

  if (argc < 2) {
    Igor::Error("Usage: {} <number grid points>", *argv);
    return 1;
  }

  const char* N_as_cstr = argv[1];
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
  const Index NGhost = 2;
  const double x_min = 0.0;
  const double x_max = 2.0;
  Grid grid(x_min, x_max, N, NGhost);

  // Burgers::FV_Godunov rhs{};
  // Burgers::FV_HighResolution rhs{Burgers::Limiter::SUPERBEE};
  // Burgers::FD_Upwind rhs{};
  Burgers::LaxWendroff rhs{};

  // DirichletZero bcond{};
  NeumannZero bcond{};

  Burgers::AdjustTimestep adjust_timestep{0.5};

  Vector<double> u0(N, NGhost);
  for (Index i = -NGhost; i < N + NGhost; ++i) {
    // u0[i] = u_analytical(grid.xm[i], 0.0);
    u0[i] = quadrature([](double x) { return u_analytical(x, 0.0); }, grid.x[i], grid.x[i + 1]) /
            grid.dx;
  }
  bcond(u0);

  // ExplicitEuler solver(grid, rhs, bcond, adjust_timestep, u0);
  // SemiImplicitCrankNicolson solver(grid, rhs, bcond, adjust_timestep, u0, 5);
  // RungeKutta2 solver(grid, rhs, bcond, adjust_timestep, u0);
  RungeKutta4 solver(grid, rhs, bcond, adjust_timestep, u0);
  solver.solve(1.0);

  {
    constexpr auto x_filename = "output/x.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(grid.xm.data() + grid.xm.nghost(), grid.xm.extent()),
                             x_filename)) {
      return 1;
    }
    Igor::Info("Saved grid to `{}`", x_filename);

    constexpr auto u_filename = "output/u.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(solver.u.data() + solver.u.nghost(), solver.u.extent()),
                             u_filename)) {
      return 1;
    }
    Igor::Info("Saved solution to `{}`", u_filename);
  }
}
