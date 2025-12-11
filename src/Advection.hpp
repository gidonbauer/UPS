#ifndef UPS_ADVECTION_HPP_
#define UPS_ADVECTION_HPP_

#include <numeric>

#include "Grid.hpp"

namespace UPS::Advection {

constexpr auto f(double u, double a) noexcept -> double { return a * u; }

// =================================================================================================
class AdjustTimestep {
  double a;
  double CFL_safety_factor;

 public:
  constexpr AdjustTimestep(double a, double CFL_safety_factor = 0.5)
      : a(a),
        CFL_safety_factor(CFL_safety_factor) {
    IGOR_ASSERT(0.0 < CFL_safety_factor && CFL_safety_factor <= 1.0,
                "Invalid CFL_safety_factor, must be in (0, 1] but is {:.6e}",
                CFL_safety_factor);
  }

  constexpr auto operator()(const Grid& grid, const Vector<double>& /*u*/) const noexcept
      -> double {
    return CFL_safety_factor * grid.dx / std::abs(a);
  }
};

// =================================================================================================
class FV_Godunov {
  double a;

  // From LeVeque: Numerical Methods for Conservation Laws 2nd edition (13.24)
  constexpr auto godunov_flux(double u_left, double u_right) const noexcept -> double {
    if (u_left <= u_right) {
      // min u in [u_left, u_right]
      return f(std::min(u_left, u_right), a);
    }
    // max u in [u_right, u_left]
    return f(std::max(u_left, u_right), a);
  }

 public:
  constexpr FV_Godunov(double a) noexcept
      : a(a) {}

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double /*dt*/,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      const auto F_minus = godunov_flux(u[i - 1], u[i]);
      const auto F_plus  = godunov_flux(u[i], u[i + 1]);
      dudt[i]            = -(1.0 / grid.dx) * (F_plus - F_minus);
    }
  }

  static constexpr auto name() noexcept -> std::string { return "FV_Godunov"; }
};

// =================================================================================================
enum class Limiter { MINMOD, SUPERBEE, VANLEER, KOREN };
class FV_HighResolution {
  double a;
  Limiter m_limiter;

  // Godunov Flux
  [[nodiscard]] constexpr auto low_order_flux(double u_left, double u_right) const noexcept
      -> double {
    if (u_left <= u_right) {
      // min u in [u_left, u_right]
      return f(std::min(u_left, u_right), a);
    }
    // max u in [u_right, u_left]
    return f(std::max(u_left, u_right), a);
  }

  // High order flux: Central finite differences (2nd order)
  [[nodiscard]] constexpr auto high_order_flux(double u_left, double u_right) const noexcept
      -> double {
    return (f(u_right, a) + f(u_left, a)) / 2.0;
  }

  [[nodiscard]] constexpr auto limiter(double U_minus, double U, double U_plus) const noexcept
      -> double {
    const double r = (U - U_minus) / (U_plus - U);

    switch (m_limiter) {
      case Limiter::MINMOD:   return std::max(0.0, std::min(1.0, r));
      case Limiter::SUPERBEE: return std::max({0.0, std::min(2.0 * r, 1.0), std::min(r, 2.0)});
      case Limiter::VANLEER:  return (r + std::abs(r)) / (1.0 + std::abs(r));
      case Limiter::KOREN:    return std::max(0.0, std::min({2.0 * r, (1.0 + 2.0 * r) / 3.0, 2.0}));
    }
  }

 public:
  constexpr FV_HighResolution(double a, Limiter limiter = Limiter::MINMOD) noexcept
      : a(a),
        m_limiter(limiter) {}

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double /*dt*/,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      const auto lf_minus = low_order_flux(u[i - 1], u[i]);
      const auto hf_minus = high_order_flux(u[i - 1], u[i]);
      const auto lf_plus  = low_order_flux(u[i], u[i + 1]);
      const auto hf_plus  = high_order_flux(u[i], u[i + 1]);

      const auto F_minus  = lf_minus + limiter(u[i - 2], u[i - 1], u[i]) * (hf_minus - lf_minus);
      const auto F_plus   = lf_plus + limiter(u[i - 1], u[i], u[i + 1]) * (hf_plus - lf_plus);
      dudt[i]             = -(F_plus - F_minus) / grid.dx;
    }
  }

  [[nodiscard]] constexpr auto name() const noexcept -> std::string {
    switch (m_limiter) {
      case Limiter::MINMOD:   return "FV_HighResolution-Minmod";
      case Limiter::SUPERBEE: return "FV_HighResolution-Superbee";
      case Limiter::VANLEER:  return "FV_HighResolution-VanLeer";
      case Limiter::KOREN:    return "FV_HighResolution-Koren";
    }
  }
};

// =================================================================================================
class FD_Upwind {
  double a;

 public:
  constexpr FD_Upwind(double a) noexcept
      : a(a) {}

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double /*dt*/,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      // 2nd order central FD -> unconditionally unstable, don't use
      // dudt[i] = -(f(u[i + 1]) - f(u[i - 1])) / (2.0 * grid.dx);

      // 1st order upwind FD
      if (a >= 0.0) {
        dudt[i] = -(f(u[i], a) - f(u[i - 1], a)) / grid.dx;
      } else {
        dudt[i] = -(f(u[i + 1], a) - f(u[i], a)) / grid.dx;
      }
    }
  }

  static constexpr auto name() noexcept -> std::string { return "FD_Upwind"; }
};

// =================================================================================================
class FD_Upwind2 {
  double a;

 public:
  constexpr FD_Upwind2(double a) noexcept
      : a(a) {}

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double /*dt*/,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      // 2st order upwind FD
      if (a >= 0.0) {
        // 1/2, −4/2, 3/2
        dudt[i] = -(3.0 * f(u[i], a) - 4.0 * f(u[i - 1], a) + f(u[i - 2], a)) / (2.0 * grid.dx);
      } else {
        // −3/2, 4/2, −1/2
        dudt[i] = -(-3.0 * f(u[i], a) + 4.0 * f(u[i + 1], a) - f(u[i + 2], a)) / (2.0 * grid.dx);
      }
    }
  }

  static constexpr auto name() noexcept -> std::string { return "FD_Upwind2"; }
};

// =================================================================================================
class LaxWendroff {
  double a;

 public:
  constexpr LaxWendroff(double a) noexcept
      : a(a) {}

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double dt,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      dudt[i] = -(f(u[i + 1], a) - f(u[i - 1], a)) / (2.0 * grid.dx) +  // 2nd order central FD
                dt / (2.0 * grid.dx * grid.dx) *                        // Numerical diffusion
                    (a * (f(u[i + 1], a) - f(u[i], a)) - a * (f(u[i], a) - f(u[i - 1], a)));
    }
  }

  static constexpr auto name() noexcept -> std::string { return "LaxWendroff"; }
};

}  // namespace UPS::Advection

#endif  // UPS_ADVECTION_HPP_
