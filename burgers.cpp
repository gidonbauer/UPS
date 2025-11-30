#include <numeric>
#include <vector>

#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>
#include <Igor/Timer.hpp>

#include "Quadrature.hpp"
#include "Vector.hpp"

constexpr auto indicator(double x, double lo, double hi) noexcept -> double {
  return static_cast<double>(lo <= x && x <= hi);
}

constexpr auto shock_location(double t) noexcept -> double { return std::sqrt(1.0 + t); }

constexpr auto u_analytical(double x, double t) noexcept -> double {
  return x / (1.0 + t) * indicator(x, 0.0, shock_location(t));
}

struct Grid {
  double dx;
  Vector<double> x;
  Vector<double> xm;

  constexpr Grid(double x_min, double x_max, Index N) noexcept
      : dx{(x_max - x_min) / static_cast<double>(N)},
        x(N + 1, 1),
        xm(N, 1) {
    for (Index i = -x.nghost(); i < x.extent() + x.nghost(); ++i) {
      x[i] = static_cast<double>(i) * dx + x_min;
    }
    for (Index i = -xm.nghost(); i < xm.extent() + xm.nghost(); ++i) {
      xm[i] = (x[i] + x[i + 1]) / 2.0;
    }
  }
};

constexpr auto abs_max(const Vector<double>& u) noexcept -> double {
  return std::transform_reduce(
      u.data(),
      u.data() + u.size(),
      0.0,
      [](double a, double b) { return std::max(a, b); },
      [](double ui) { return std::abs(ui); });
}

constexpr void zero_dirichlet(Vector<double>& u) noexcept {
  u[-1]                          = 0.0;
  u[u.extent() + u.nghost() - 1] = 0.0;
}

constexpr void zero_neumann(Vector<double>& u) noexcept {
  u[-1]                          = u[0];
  u[u.extent() + u.nghost() - 1] = u[u.extent() + u.nghost() - 2];
}

constexpr auto f(double u) noexcept -> double { return 0.5 * u * u; }

constexpr auto solve_burgers(const Grid& grid,
                             const Vector<double>& u0,
                             double tend,
                             double CFL_safety_factor = 0.5) noexcept -> Vector<double> {
  // From LeVeque: Numerical Methods for Conservation Laws 2nd edition (13.24)
  constexpr auto godunov_flux = []<typename T>(const T& u_left,
                                               const T& u_right) constexpr noexcept -> T {
    constexpr auto zero = static_cast<T>(0);
    if (u_left <= u_right) {
      // min u in [u_left, u_right] f(u) = 0.5 * u^2
      if (u_left <= zero && u_right >= zero) { return f(zero); }
      return f(std::min(std::abs(u_left), std::abs(u_right)));
    }
    // max u in [u_right, u_left] f(u) = 0.5 * u^2
    return f(std::max(std::abs(u_left), std::abs(u_right)));
  };

  Vector<double> u_curr(u0.extent(), u0.nghost());
  Vector<double> u_next(u0.extent(), u0.nghost());
  std::copy_n(u0.data(), u0.size(), u_curr.data());
  std::fill_n(u_next.data(), u_next.size(), 0.0);

  for (double t = 0.0; t < tend;) {
    const double CFL_factor = abs_max(u_curr);
    if (std::isnan(CFL_factor) || std::isinf(CFL_factor)) {
      Igor::Panic("CFL_factor is invalid at time t={}: CFL_factor = {}", t, CFL_factor);
    }
    const double dt = std::min(CFL_safety_factor * grid.dx / CFL_factor, tend - t);

    // Solve for interior points
    for (Index i = 0; i < u0.extent(); ++i) {
      const auto F_minus = godunov_flux(u_curr[i - 1], u_curr[i]);
      const auto F_plus  = godunov_flux(u_curr[i], u_curr[i + 1]);
      u_next[i]          = u_curr[i] - (dt / grid.dx) * (F_plus - F_minus);
    }

    // Solve for boundary cells
    zero_dirichlet(u_next);
    // zero_neumann(u_next);

    // Update time
    t += dt;

    // std::swap(u_curr, u_next);
    std::copy_n(u_next.data(), u_next.size(), u_curr.data());
  }

  return u_curr;
}

auto main(int argc, char** argv) -> int {
  Igor::ScopeTimer timer("Solver");

  if (argc < 2) {
    Igor::Error("Usage: {} <number grid points>", *argv);
    return 1;
  }

  const Index N = std::atoi(argv[1]);
  if (N <= 0) {
    Igor::Error("Number of grid points must be greater than 0 but is {}", N);
    return 1;
  }

  const double x_min = 0.0;
  const double x_max = 2.0;
  Grid grid(x_min, x_max, N);

  Vector<double> u0(N, 1);
  for (Index i = -u0.nghost(); i < u0.extent() + u0.nghost(); ++i) {
    u0[i] = quadrature([](double x) { return u_analytical(x, 0.0); }, grid.x[i], grid.x[i + 1]) /
            grid.dx;
  }
  zero_dirichlet(u0);
  // zero_neumann(u0);

  const Vector<double> u = solve_burgers(grid, u0, 1.0);

  {
    constexpr auto x_filename = "output/x.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(grid.xm.data() + u.nghost(), grid.xm.extent()),
                             x_filename)) {
      return 1;
    }
    Igor::Info("Saved grid to `{}`", x_filename);

    constexpr auto u0_filename = "output/u0.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(u0.data() + u.nghost(), u0.extent()), u0_filename)) {
      return 1;
    }
    Igor::Info("Saved initial condition to `{}`", u0_filename);

    constexpr auto u_filename = "output/u.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(u.data() + u.nghost(), u.extent()), u_filename)) {
      return 1;
    }
    Igor::Info("Saved solution to `{}`", u_filename);
  }
}
