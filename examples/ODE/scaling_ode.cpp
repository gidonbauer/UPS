#include <Igor/Logging.hpp>
#include <Igor/MdspanToNpy.hpp>

#include "UPS.hpp"
using namespace UPS;
using namespace UPS::ODE;

// = Analytical solution ===========================================================================
constexpr double T_END = 1.0;
constexpr double ALPHA = 5.0;

constexpr auto u_analytical(double t) noexcept -> double { return std::exp(ALPHA * t); }
// = Analytical solution ===========================================================================

// =================================================================================================
class RHS {
 public:
  static constexpr void operator()(double u, double /*dt*/, double& dudt) noexcept {
    dudt = ALPHA * u;
  }

  static constexpr auto name() noexcept -> std::string { return "ODE"; }
};

// =================================================================================================
template <template <class State, class RHS> class TI>
void run_scaling_test(Index N) {
  RHS rhs{};
  const double dt = 1.0 / static_cast<double>(N);
  double u0       = u_analytical(0.0);

  TI solver(rhs, u0);
  if (!solver.solve(dt, T_END)) {
    Igor::Error("{}, {}, {} failed.", solver.name(), rhs.name(), N);
    return;
  }

  const auto L1_error = std::abs(solver.u - u_analytical(T_END));

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
