#include <Igor/Math.hpp>
#include <Igor/MdspanToNpy.hpp>
#include <Igor/Timer.hpp>

#include "UPS.hpp"

// = Gravitiy ======================================================================================
class Gravity {
  static constexpr double G            = 1.0;  // 6.67430e-11;  // m^3 / (kg * s^2)
  static constexpr UPS::Vector3 center = {.x = 0.0, .y = 0.0, .z = 0.0};
  double m_center;

 public:
  constexpr Gravity(double m_center) noexcept
      : m_center(m_center) {}

  constexpr void
  operator()(const UPS::PhaseState3& u, double /*dt*/, UPS::PhaseState3& dudt) const noexcept {
    const double r_sqr = Igor::sqr(u.pos.x - center.x) + Igor::sqr(u.pos.y - center.y) +
                         Igor::sqr(u.pos.z - center.z);
    const double F = G * m_center / r_sqr;
    const auto dir = normalize(center - u.pos);

    dudt.pos       = u.vel;
    dudt.vel       = F * dir;
  }

  static constexpr auto name() noexcept -> std::string { return "Gravity"; }
};

// =================================================================================================
template <typename Solution>
auto save_solution(const std::string& filename, const std::vector<Solution>& sol) -> bool {
  constexpr size_t NUM_ELEM = 7;
  static_assert(std::is_same_v<decltype(sol[0].t), double>);
  static_assert(sizeof(Solution) == NUM_ELEM * sizeof(double));
#if 0
  std::vector<double> data(7 * sol.size());
  for (size_t i = 0; i < sol.size(); ++i) {
    data[7 * i + 0] = sol[i].t;
    data[7 * i + 1] = sol[i].u.x;
    data[7 * i + 2] = sol[i].u.y;
    data[7 * i + 3] = sol[i].u.z;
    data[7 * i + 4] = sol[i].u.u;
    data[7 * i + 5] = sol[i].u.v;
    data[7 * i + 6] = sol[i].u.w;
  }
  return Igor::mdspan_to_npy(std::mdspan(data.data(), sol.size(), NUM_ELEM), filename);
#else
  IGOR_ASSERT(static_cast<const void*>(sol.data()) == static_cast<const void*>(&sol[0].t),
              "Invalid layout of struct `Solution`.");
  return Igor::mdspan_to_npy(std::mdspan(&sol[0].t, sol.size(), NUM_ELEM), filename);
#endif
}

// =================================================================================================
template <template <class State, class RHS> class TI>
void run(Gravity rhs, UPS::PhaseState3 u0, double tend, double dt) {
  TI solver(rhs, u0);
  const double dt_write = -1.0;  // std::max(1e-1, dt);
  std::vector<typename TI<UPS::PhaseState3, Gravity>::Solution> sol{};
  sol.emplace_back(0.0, u0);
  // IGOR_TIME_SCOPE(solver.name()) {
  if (!solver.solve(dt, tend, dt_write, sol)) {
    Igor::Error("{}-{} failed.", solver.name(), rhs.name());
    return;
  }
  // }
#pragma omp critical
  {
    Igor::Info("{}", solver.name());
    Igor::Info("  pos({:.1e}) = ({:+.6e}, {:+.6e}, {:+.6e})",
               tend,
               solver.u.pos.x,
               solver.u.pos.y,
               solver.u.pos.z);
    Igor::Info("  vel({:.1e}) = ({:+.6e}, {:+.6e}, {:+.6e})",
               tend,
               solver.u.vel.x,
               solver.u.vel.y,
               solver.u.vel.z);

    const std::string filename = Igor::detail::format("./output/Orbit-{}.npy", solver.name());
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
  constexpr double m_sun = 1e3;  // 1.989e+30;  // kg
  Gravity rhs(m_sun);
  UPS::PhaseState3 u0{
      .pos = {.x = 10.0, .y = 10.0, .z = 10.0},
      .vel = {.x = 5.0, .y = 0.0, .z = 0.0},
  };
  double tend = 2e2;
  double dt   = 1e-1;

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
