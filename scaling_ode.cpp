#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>

#include "TimeIntegrator.hpp"
using namespace UPS;

// = Analytical solution ===========================================================================
constexpr double T_END = 1.0;
constexpr double ALPHA = 1.0;

constexpr auto u_analytical(double t) noexcept -> double { return std::exp(ALPHA * t); }
// = Analytical solution ===========================================================================

// =================================================================================================
class RHS {
 public:
  static constexpr void operator()(const Grid& /*grid*/,
                                   const Vector<double>& u,
                                   double /*dt*/,
                                   Vector<double>& dudt) noexcept {
    IGOR_ASSERT(u.size() == 1, "Expected size one");
    dudt[0] = ALPHA * u[0];
  }

  static constexpr auto name() noexcept -> std::string { return "ODE"; }
};

class NoopBcond {
 public:
  static constexpr void operator()(Vector<double>& /*u*/) {}
};

class ConstantTimestep {
  double dt;

 public:
  constexpr ConstantTimestep(double dt)
      : dt(dt) {
    IGOR_ASSERT(dt > 0.0, "dt must be greater than 0 but is {:.6e}", dt);
  }

  constexpr auto operator()(const Grid& /*grid*/, const Vector<double>& /*u*/) const noexcept
      -> double {
    return dt;
  }
};

// =================================================================================================
template <template <class RHS, class BCond, class AdjustTimestep> class TI>
void run_scaling_test(Index N) {
  Grid grid(0.0, 0.0, 1, 0);

  RHS rhs{};
  NoopBcond bcond{};
  ConstantTimestep adjust_timestep(1.0 / static_cast<double>(N));

  Vector<double> u0(1, 0);
  u0[0] = u_analytical(0.0);

  TI solver(grid, rhs, bcond, adjust_timestep, u0);
  if (!solver.solve(T_END)) {
    Igor::Error("{}, {}, {} failed.", solver.name(), rhs.name(), N);
    return;
  }

  const auto L1_error = std::abs(solver.u[0] - u_analytical(T_END));

#pragma omp critical
  std::cout << solver.name() << ',' << rhs.name() << ',' << N << ',' << L1_error << '\n';
}

auto main() -> int {

  Index N = 1;
  std::cout << "TimeIntegrator" << ',' << "RHS" << ',' << 'N' << ',' << "L1_error" << '\n';
#pragma omp parallel
#pragma omp single
  for (Index i = 0; i < 12; ++i) {
    N *= 2;

#pragma omp task firstprivate(N)
    {
      run_scaling_test<ExplicitEuler>(N);
      run_scaling_test<SemiImplicitCrankNicolson>(N);
      run_scaling_test<RungeKutta2>(N);
      run_scaling_test<RungeKutta4>(N);
    }
  }
}
