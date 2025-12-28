#include <Igor/Math.hpp>
#include <Igor/MdspanToNpy.hpp>
#include <Igor/Timer.hpp>

#include "UPS.hpp"

// = Vector3 =======================================================================================
struct Vector3 {
  double x, y, z;
};

constexpr auto operator-(Vector3 lhs, const Vector3& rhs) noexcept -> Vector3 {
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  return lhs;
}

constexpr auto operator/(Vector3 lhs, double rhs) noexcept -> Vector3 {
  lhs.x /= rhs;
  lhs.y /= rhs;
  lhs.z /= rhs;
  return lhs;
}

constexpr auto norm(const Vector3& vec) noexcept -> double {
  return std::sqrt(Igor::sqr(vec.x) + Igor::sqr(vec.y) + Igor::sqr(vec.z));
}

constexpr auto normalize(Vector3 vec) noexcept -> Vector3 { return vec / norm(vec); }

// = State =========================================================================================
struct State {
  double x, y, z;
  double u, v, w;
};

[[nodiscard]] constexpr auto operator*(double lhs, State rhs) noexcept -> State {
  rhs.x *= lhs;
  rhs.y *= lhs;
  rhs.z *= lhs;
  rhs.u *= lhs;
  rhs.v *= lhs;
  rhs.w *= lhs;
  return rhs;
}

[[nodiscard]] constexpr auto operator/(State lhs, double rhs) noexcept -> State {
  lhs.x /= rhs;
  lhs.y /= rhs;
  lhs.z /= rhs;
  lhs.u /= rhs;
  lhs.v /= rhs;
  lhs.w /= rhs;
  return lhs;
}

constexpr auto operator-(State lhs, const State& rhs) noexcept -> State {
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  lhs.u -= rhs.u;
  lhs.v -= rhs.v;
  lhs.w -= rhs.w;
  return lhs;
}

constexpr auto operator+(State lhs, const State& rhs) noexcept -> State {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  lhs.u += rhs.u;
  lhs.v += rhs.v;
  lhs.w += rhs.w;
  return lhs;
}

constexpr auto operator+=(State& lhs, const State& rhs) noexcept -> State& {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  lhs.u += rhs.u;
  lhs.v += rhs.v;
  lhs.w += rhs.w;
  return lhs;
}

// = Gravitiy ======================================================================================
class Gravity {
  static constexpr double G       = 1.0;  // 6.67430e-11;  // m^3 / (kg * s^2)
  static constexpr Vector3 center = {.x = 0.0, .y = 0.0, .z = 0.0};
  double m_center;

 public:
  constexpr Gravity(double m_center) noexcept
      : m_center(m_center) {}

  constexpr void operator()(const State& u, double /*dt*/, State& dudt) const noexcept {
    const double r_sqr =
        Igor::sqr(u.x - center.x) + Igor::sqr(u.y - center.y) + Igor::sqr(u.z - center.z);
    const double F    = G * m_center / r_sqr;
    const Vector3 dir = normalize(center - Vector3{.x = u.x, .y = u.y, .z = u.z});

    dudt.x            = u.u;
    dudt.y            = u.v;
    dudt.z            = u.w;
    dudt.u            = F * dir.x;
    dudt.v            = F * dir.y;
    dudt.w            = F * dir.z;
  }

  static constexpr auto name() noexcept -> std::string { return "Gravity"; }
};

// =================================================================================================
template <typename State, typename RHS>
class LeapFrog final : public UPS::ODE::TimeIntegrator<State, RHS> {
  using TI = UPS::ODE::TimeIntegrator<State, RHS>;
  using TI::rhs;
  using TI::should_save;

 public:
  using TI::solve;
  using TI::u;
  using typename TI::Solution;

 private:
  State dudt{};

  constexpr auto do_step(double /*dt*/) noexcept -> double override { Igor::Panic("Unreachable."); }

 public:
  constexpr LeapFrog(RHS rhs, const State& u0)
      : TI(std::move(rhs), u0) {}

  [[nodiscard]] constexpr auto solve(double dt, double tend) noexcept -> bool override {
    double t = 0.0;

    // Half step for velocity
    rhs(u, dt, dudt);
    u.u += 0.5 * dt * dudt.u;
    u.v += 0.5 * dt * dudt.v;
    u.w += 0.5 * dt * dudt.w;

    while (t < tend) {
      dt = std::min(dt, tend - t);

      rhs(u, dt, dudt);
      u.u += dt * dudt.u;
      u.v += dt * dudt.v;
      u.w += dt * dudt.w;

      u.x += dt * u.u;
      u.y += dt * u.v;
      u.z += dt * u.w;

      t   += dt;
    }
    // Half step backwards for velocity
    rhs(u, dt, dudt);
    u.u -= 0.5 * dt * dudt.u;
    u.v -= 0.5 * dt * dudt.v;
    u.w -= 0.5 * dt * dudt.w;

    return true;
  }

  [[nodiscard]] constexpr auto
  solve(double dt, double tend, double dt_write, std::vector<Solution>& solution) noexcept
      -> bool override {
    double t = 0.0;

    // Half step for velocity
    rhs(u, dt, dudt);
    u.u += 0.5 * dt * dudt.u;
    u.v += 0.5 * dt * dudt.v;
    u.w += 0.5 * dt * dudt.w;

    while (t < tend) {
      dt = std::min(dt, tend - t);

      rhs(u, dt, dudt);
      u.u += dt * dudt.u;
      u.v += dt * dudt.v;
      u.w += dt * dudt.w;

      u.x += dt * u.u;
      u.y += dt * u.v;
      u.z += dt * u.w;

      t   += dt;
      if (should_save(t, dt, dt_write, tend)) { solution.emplace_back(t, u); }
    }
    // Half step backwards for velocity
    rhs(u, dt, dudt);
    u.u -= 0.5 * dt * dudt.u;
    u.v -= 0.5 * dt * dudt.v;
    u.w -= 0.5 * dt * dudt.w;

    return true;
  }

  [[nodiscard]] constexpr auto name() const noexcept -> std::string override { return "LeapFrog"; }
};

// =================================================================================================
template <typename State, typename RHS>
class SymplecticEuler final : public UPS::ODE::TimeIntegrator<State, RHS> {
  using TI = UPS::ODE::TimeIntegrator<State, RHS>;
  using TI::rhs;
  using TI::should_save;

 public:
  using TI::solve;
  using TI::u;
  using typename TI::Solution;

 private:
  State dudt{};

  constexpr auto do_step(double /*dt*/) noexcept -> double override { Igor::Panic("Unreachable."); }

 public:
  constexpr SymplecticEuler(RHS rhs, const State& u0)
      : TI(std::move(rhs), u0) {}

  [[nodiscard]] constexpr auto solve(double dt, double tend) noexcept -> bool override {
    double t = 0.0;
    while (t < tend) {
      dt = std::min(dt, tend - t);

      rhs(u, dt, dudt);
      u.u += dt * dudt.u;
      u.v += dt * dudt.v;
      u.w += dt * dudt.w;

      u.x += dt * u.u;
      u.y += dt * u.v;
      u.z += dt * u.w;

      t   += dt;
    }

    return true;
  }

  [[nodiscard]] constexpr auto
  solve(double dt, double tend, double dt_write, std::vector<Solution>& solution) noexcept
      -> bool override {
    double t = 0.0;
    while (t < tend) {
      dt = std::min(dt, tend - t);

      rhs(u, dt, dudt);
      u.u += dt * dudt.u;
      u.v += dt * dudt.v;
      u.w += dt * dudt.w;

      u.x += dt * u.u;
      u.y += dt * u.v;
      u.z += dt * u.w;

      t   += dt;
      if (should_save(t, dt, dt_write, tend)) { solution.emplace_back(t, u); }
    }

    return true;
  }

  [[nodiscard]] constexpr auto name() const noexcept -> std::string override {
    return "SymplecticEuler";
  }
};

// =================================================================================================
template <typename Solution>
auto save_solution(const std::string& filename, const std::vector<Solution>& sol) -> bool {
  static_assert(sizeof(Solution) == 7 * sizeof(double));
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
  return Igor::mdspan_to_npy(std::mdspan(data.data(), sol.size(), 7), filename);
}

template <template <class State, class RHS> class TI>
void run(Gravity rhs, State u0, double tend, double dt) {
  TI solver(rhs, u0);
  const double dt_write = -1.0;  // std::max(1e-1, dt);
  std::vector<typename TI<State, Gravity>::Solution> sol{};
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
    Igor::Info(
        "  pos({:.1e}) = ({:+.6e}, {:+.6e}, {:+.6e})", tend, solver.u.x, solver.u.y, solver.u.z);
    Igor::Info(
        "  vel({:.1e}) = ({:+.6e}, {:+.6e}, {:+.6e})", tend, solver.u.u, solver.u.v, solver.u.w);

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
  State u0{
      .x = 10.0,
      .y = 10.0,
      .z = 10.0,
      .u = 5.0,
      .v = 0.0,
      .w = 0.0,
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
    run<LeapFrog>(rhs, u0, tend, dt);
#pragma omp task
    run<SymplecticEuler>(rhs, u0, tend, dt);
  }
}
