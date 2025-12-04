#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>
#include <Igor/Timer.hpp>

#include "BoundaryConditions.hpp"
#include "Grid.hpp"
#include "Heat.hpp"
#include "Quadrature.hpp"
#include "TimeIntegrator.hpp"
#include "Vector.hpp"
using namespace UPS;

// = One possible analytical solution for Neumann boundary conditions with derivative zero =========
[[nodiscard]] constexpr auto
u_analytical(double x, double t, double a, double x_min, double x_max) noexcept -> double {
  constexpr std::array z = {1.0, 3.0};

  double res             = 0.0;
  for (auto zi : z) {
    res += std::cos((zi * std::numbers::pi) / (x_max - x_min) * x) *
           std::exp(-a * std::pow((zi * std::numbers::pi) / (x_max - x_min), 2.0) * t);
  }
  return res;
}

// =================================================================================================
auto main() -> int {
  const double a     = 2.0;
  const Index N      = 50;
  const Index NGhost = 2;
  const double x_min = 0.0;
  const double x_max = 10.0;
  Grid grid(x_min, x_max, N, NGhost);

  Heat::FD_Central2 rhs(a);
  // Heat::FD_Central4 rhs(a);

  NeumannZero bcond{};

  Heat::AdjustTimestep adjust_timestep(a, 0.5);

  Vector<double> u0(N, NGhost);
  for (Index i = -NGhost; i < N + NGhost; ++i) {
    u0[i] = quadrature([=](double x) { return u_analytical(x, 0.0, a, x_min, x_max); },
                       grid.x[i],
                       grid.x[i + 1]) /
            grid.dx;
  }
  bcond(u0);

  // ExplicitEuler solver(grid, rhs, bcond, adjust_timestep, u0);
  // SemiImplicitCrankNicolson solver(grid, rhs, bcond, adjust_timestep, u0, 5);
  // RungeKutta2 solver(grid, rhs, bcond, adjust_timestep, u0);
  RungeKutta4 solver(grid, rhs, bcond, adjust_timestep, u0);
  solver.solve(1.0);

  Igor::Info("Right-hand side: {}", rhs.name());
  Igor::Info("Time integration side: {}", solver.name());

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
