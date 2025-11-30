#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>
#include <Igor/Timer.hpp>

#include "Vector.hpp"

// =================================================================================================
struct Grid {
  Index N;
  Index NGhost;
  double dx;
  Vector<double> x;
  Vector<double> xm;

  constexpr Grid(double x_min, double x_max, Index N, Index NGhost) noexcept
      : N(N),
        NGhost(NGhost),
        dx{(x_max - x_min) / static_cast<double>(N)},
        x(N + 1, NGhost),
        xm(N, NGhost) {
    for (Index i = -x.nghost(); i < x.extent() + x.nghost(); ++i) {
      x[i] = static_cast<double>(i) * dx + x_min;
    }
    for (Index i = -xm.nghost(); i < xm.extent() + xm.nghost(); ++i) {
      xm[i] = (x[i] + x[i + 1]) / 2.0;
    }
  }
};

// =================================================================================================
template <typename RHS, typename BCond>
class ExplicitEuler {
  RHS rhs;
  BCond bcond;
  Grid grid;

  Vector<double> dudt;

 public:
  Vector<double> u;

  constexpr ExplicitEuler(Grid grid, RHS rhs, BCond bcond, const Vector<double>& u0)
      : rhs(std::move(rhs)),
        bcond(std::move(bcond)),
        grid(std::move(grid)),
        dudt(grid.N, grid.NGhost),
        u(u0) {
    IGOR_ASSERT(grid.N == u0.extent(), "Incompatible size of grid and u0");
    IGOR_ASSERT(grid.NGhost == u0.nghost(), "Incompatible size of grid and u0");
  }

  constexpr auto do_step(double dt) noexcept -> double {
    rhs(grid, u, dudt);
    for (Index i = 0; i < u.extent(); ++i) {
      u[i] += dt * dudt[i];
    }
    bcond(u);

    return dt;
  }

  constexpr void solve(double tend, double dt) noexcept {
    double t = 0.0;
    while (t < tend) {
      dt  = std::min(dt, tend - t);
      dt  = do_step(dt);
      t  += dt;
    }
  }
};

// =================================================================================================
class BurgersRhs {
  static constexpr auto f(double u) noexcept -> double { return 0.5 * u * u; }

  // From LeVeque: Numerical Methods for Conservation Laws 2nd edition (13.24)
  static constexpr auto godunov_flux(double u_left, double u_right) noexcept -> double {
    if (u_left <= u_right) {
      // min u in [u_left, u_right] f(u) = 0.5 * u^2
      if (u_left <= 0.0 && u_right >= 0.0) { return f(0.0); }
      return f(std::min(std::abs(u_left), std::abs(u_right)));
    }
    // max u in [u_right, u_left] f(u) = 0.5 * u^2
    return f(std::max(std::abs(u_left), std::abs(u_right)));
  }

 public:
  static constexpr void
  operator()(const Grid& grid, const Vector<double>& u, Vector<double>& dudt) noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      const auto F_minus = godunov_flux(u[i - 1], u[i]);
      const auto F_plus  = godunov_flux(u[i], u[i + 1]);
      dudt[i]            = -(1.0 / grid.dx) * (F_plus - F_minus);
    }
  }
};

// =================================================================================================
class DirichletZero {
 public:
  static constexpr void operator()(Vector<double>& u) {
    for (Index i = -u.nghost(); i < 0; ++i) {
      u[i] = 0.0;
    }
    for (Index i = u.extent(); i < u.extent() + u.nghost(); ++i) {
      u[i] = 0.0;
    }
  }
};

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

  const Index N = std::atoi(argv[1]);
  if (N <= 0) {
    Igor::Error("Number of grid points must be greater than 0 but is {}", N);
    return 1;
  }
  const Index NGhost = 1;
  const double x_min = 0.0;
  const double x_max = 2.0;
  Grid grid(x_min, x_max, N, NGhost);

  BurgersRhs rhs{};
  DirichletZero bcond{};

  Vector<double> u0(N, NGhost);
  for (Index i = -NGhost; i < N + NGhost; ++i) {
    u0[i] = u_analytical(grid.xm[i], 0.0);
  }
  bcond(u0);

  ExplicitEuler solver(grid, rhs, bcond, u0);
  solver.solve(1.0, 0.5 * grid.dx);

  {
    constexpr auto x_filename = "output/x.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(grid.xm.data() + grid.xm.nghost(), grid.xm.extent()),
                             x_filename)) {
      return 1;
    }
    Igor::Info("Saved grid to `{}`", x_filename);

    constexpr auto u0_filename = "output/u0.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(u0.data() + u0.nghost(), u0.extent()), u0_filename)) {
      return 1;
    }
    Igor::Info("Saved initial condition to `{}`", u0_filename);

    constexpr auto u_filename = "output/u.npy";
    if (!Igor::mdspan_to_npy(std::mdspan(solver.u.data() + solver.u.nghost(), solver.u.extent()),
                             u_filename)) {
      return 1;
    }
    Igor::Info("Saved solution to `{}`", u_filename);
  }
}
