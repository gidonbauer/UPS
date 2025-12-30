#include <Igor/MdspanToNpy.hpp>

#include "UPS.hpp"

// = Analytical solution ===========================================================================
constexpr auto analytical_solution(double t, double k) -> UPS::PhaseState1 {
  return {
      .pos = std::cos(std::sqrt(k) * t),
      .vel = -std::sqrt(k) * std::sin(std::sqrt(k) * t),
  };
}

// = Harmonic Oscillator ===========================================================================
class HarmonicOscillator {
  double m_k;

 public:
  constexpr HarmonicOscillator(double k) noexcept
      : m_k(k) {}

  constexpr void
  operator()(const UPS::PhaseState1& u, double /*dt*/, UPS::PhaseState1& dudt) const noexcept {
    dudt.pos = u.vel;
    dudt.vel = -m_k * u.pos;
  }

  [[nodiscard]] constexpr auto k() const -> double { return m_k; }

  static constexpr auto name() noexcept -> std::string { return "HarmonicOscillator"; }
};

// =================================================================================================
template <typename Solution>
auto save_solution(const std::string& filename, const std::vector<Solution>& sol) -> bool {
  constexpr size_t NUM_ELEM = 3;
  static_assert(std::is_same_v<decltype(sol[0].t), double>);
  static_assert(sizeof(Solution) == NUM_ELEM * sizeof(double));
  IGOR_ASSERT(static_cast<const void*>(sol.data()) == static_cast<const void*>(&sol[0].t),
              "Invalid layout of struct `Solution`.");
  return Igor::mdspan_to_npy(std::mdspan(&sol[0].t, sol.size(), NUM_ELEM), filename);
}

// =================================================================================================
template <template <class State, class RHS> class TI>
void run(HarmonicOscillator rhs, UPS::PhaseState1 u0, double tend, double dt) {
  TI solver(rhs, u0);
  const double dt_write = -1.0;  // std::max(1e-1, dt);
  std::vector<typename TI<UPS::PhaseState1, HarmonicOscillator>::Solution> sol{};
  sol.emplace_back(0.0, u0);
  if (!solver.solve(dt, tend, dt_write, sol)) {
    Igor::Error("{}-{} failed.", solver.name(), rhs.name());
    return;
  }
  const UPS::PhaseState1 u_analytical = analytical_solution(tend, rhs.k());
#pragma omp critical
  {
    Igor::Info("{}", solver.name());
    Igor::Info("  pos({:.1e}) = {:+.6e} (rel. error = {:.6e})",
               tend,
               solver.u.pos,
               std::abs((solver.u.pos - u_analytical.pos) / u_analytical.pos));
    Igor::Info("  vel({:.1e}) = {:+.6e} (rel. error = {:.6e})",
               tend,
               solver.u.vel,
               std::abs((solver.u.vel - u_analytical.vel) / u_analytical.vel));

    const std::string filename =
        Igor::detail::format("./output/HarmonicOscillator-{}.npy", solver.name());
    if (!save_solution(filename, sol)) {
      Igor::Error("Could not save solution to `{}`", filename);
    } else {
      Igor::Info("Saved solution to `{}`", filename);
    }
    std::cout << '\n';
  }
}

// =================================================================================================
auto main() -> int {
  constexpr double k = 1.0;
  HarmonicOscillator rhs(k);
  UPS::PhaseState1 u0{.pos = 1.0, .vel = 0.0};
  double tend             = 1e4;
  double dt               = 0.25;

  const auto u_analytical = analytical_solution(tend, k);
  Igor::Info("Analytical solution");
  Igor::Info("  pos({:.1e}) = {:+.6e}", tend, u_analytical.pos);
  Igor::Info("  vel({:.1e}) = {:+.6e}", tend, u_analytical.vel);
  std::cout << '\n';

#pragma omp parallel
#pragma omp single
  {
#pragma omp task
    run<UPS::ODE::ExplicitEuler>(rhs, u0, tend, dt);
#pragma omp task
    run<UPS::ODE::RungeKutta2>(rhs, u0, tend, dt);
#pragma omp task
    run<UPS::ODE::RungeKutta4>(rhs, u0, tend, dt);
#pragma omp task
    run<UPS::ODE::SemiImplicitCrankNicolson>(rhs, u0, tend, dt);
#pragma omp task
    run<UPS::ODE::AdamsBashforth>(rhs, u0, tend, dt);
#pragma omp task
    run<UPS::ODE::LeapFrog>(rhs, u0, tend, dt);
#pragma omp task
    run<UPS::ODE::SymplecticEuler>(rhs, u0, tend, dt);
  }
}
