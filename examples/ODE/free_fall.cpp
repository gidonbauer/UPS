#include <Igor/Logging.hpp>

#include "UPS.hpp"

struct Vector2 {
  double x, v;
};

[[nodiscard]] constexpr auto operator*(double lhs, Vector2 rhs) noexcept -> Vector2 {
  rhs.x *= lhs;
  rhs.v *= lhs;
  return rhs;
}

[[nodiscard]] constexpr auto operator/(Vector2 lhs, double rhs) noexcept -> Vector2 {
  lhs.x /= rhs;
  lhs.v /= rhs;
  return lhs;
}

constexpr auto operator-(Vector2 lhs, const Vector2& rhs) noexcept -> Vector2 {
  lhs.x -= rhs.x;
  lhs.v -= rhs.v;
  return lhs;
}

constexpr auto operator+(Vector2 lhs, const Vector2& rhs) noexcept -> Vector2 {
  lhs.x += rhs.x;
  lhs.v += rhs.v;
  return lhs;
}

constexpr auto operator+=(Vector2& lhs, const Vector2& rhs) noexcept -> Vector2& {
  lhs.x += rhs.x;
  lhs.v += rhs.v;
  return lhs;
}

// namespace std {
// auto isinf(const Vector2& u) noexcept -> bool { return isinf(u.x) || isinf(u.v); }
// auto isnan(const Vector2& u) noexcept -> bool { return isnan(u.x) || isnan(u.v); }
// }  // namespace std

// = Analytical solution ===========================================================================
constexpr double G = 9.80665;
constexpr auto u_analytical(double t) noexcept -> Vector2 {
  return {
      .x = 0.5 * G * t * t,
      .v = G * t,
  };
}
// = Analytical solution ===========================================================================

// =================================================================================================
class FreeFall {
 public:
  static constexpr void operator()(Vector2 u, double /*dt*/, Vector2& dudt) noexcept {
    dudt.x = u.v;
    dudt.v = G;
  }

  static constexpr auto name() noexcept -> std::string { return "FreeFall"; }
};

#define RUN(TI)                                                                                    \
  {                                                                                                \
    TI solver(rhs, Vector2{.x = 0.0, .v = 0.0});                                                   \
    if (!solver.solve(dt, t_end)) {                                                                \
      Igor::Error("{}-{} failed.", solver.name(), rhs.name());                                     \
      return 1;                                                                                    \
    }                                                                                              \
    Igor::Info("{}", solver.name());                                                               \
    Igor::Info("x({:.1f}) = {:.4f}", t_end, solver.u.x);                                           \
    Igor::Info("v({:.1f}) = {:.4f}", t_end, solver.u.v);                                           \
    std::cout << '\n';                                                                             \
  }

auto main() -> int {
  FreeFall rhs{};
  const double dt    = 0.5;
  const double t_end = 3.0;

  RUN(UPS::ODE::ExplicitEuler);
  RUN(UPS::ODE::SemiImplicitCrankNicolson);
  RUN(UPS::ODE::RungeKutta2);
  RUN(UPS::ODE::RungeKutta4);
  RUN(UPS::ODE::AdamsBashforth);
  Igor::Info("Analytical:");
  Igor::Info("x({:.1f}) = {:.4f}", t_end, u_analytical(t_end).x);
  Igor::Info("v({:.1f}) = {:.4f}", t_end, u_analytical(t_end).v);
}
