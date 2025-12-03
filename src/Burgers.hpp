#ifndef UPS_BURGERS_HPP_
#define UPS_BURGERS_HPP_

#include <numeric>

#include "Grid.hpp"

namespace UPS::Burgers {

constexpr auto f(double u) noexcept -> double { return 0.5 * u * u; }

// =================================================================================================
class AdjustTimestep {
  double CFL_safety_factor;

 public:
  constexpr AdjustTimestep(double CFL_safety_factor = 0.5)
      : CFL_safety_factor(CFL_safety_factor) {
    IGOR_ASSERT(0.0 < CFL_safety_factor && CFL_safety_factor <= 1.0,
                "Invalid CFL_safety_factor, must be in (0, 1] but is {:.6e}",
                CFL_safety_factor);
  }

  constexpr auto operator()(const Grid& grid, const Vector<double>& u) const noexcept -> double {
    const auto abs_max_u = std::transform_reduce(
        u.data(),
        u.data() + u.size(),
        0.0,
        [](double a, double b) { return std::max(a, b); },
        [](double ui) { return std::abs(ui); });
    IGOR_ASSERT(!std::isnan(abs_max_u) && !std::isinf(abs_max_u),
                "Bad solution: abs_max_u = {}",
                abs_max_u);
    return CFL_safety_factor * grid.dx / abs_max_u;
  }
};

// =================================================================================================
class FV_Godunov {
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
  static constexpr void operator()(const Grid& grid,
                                   const Vector<double>& u,
                                   double /*dt*/,
                                   Vector<double>& dudt) noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      const auto F_minus = godunov_flux(u[i - 1], u[i]);
      const auto F_plus  = godunov_flux(u[i], u[i + 1]);
      dudt[i]            = -(1.0 / grid.dx) * (F_plus - F_minus);
    }
  }
};

// =================================================================================================
enum class Limiter { MINMOD, SUPERBEE, VANLEER, KOREN };
class FV_HighResolution {
  Limiter m_limiter;

  // Godunov Flux
  static constexpr auto low_order_flux(double u_left, double u_right) noexcept -> double {
    if (u_left <= u_right) {
      // min u in [u_left, u_right] f(u) = 0.5 * u^2
      if (u_left <= 0.0 && u_right >= 0.0) { return f(0.0); }
      return f(std::min(std::abs(u_left), std::abs(u_right)));
    }
    // max u in [u_right, u_left] f(u) = 0.5 * u^2
    return f(std::max(std::abs(u_left), std::abs(u_right)));
  }

  // Lax-Wendroff Flux
  static constexpr auto
  high_order_flux(double u_left, double u_right, double dt, double dx) noexcept -> double {
    const auto u_mid = (f(u_right) + f(u_left)) / 2.0;
    return u_mid - dt / (2.0 * dx) * (u_mid * (f(u_right) - f(u_left)));
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
  constexpr FV_HighResolution(Limiter limiter = Limiter::MINMOD)
      : m_limiter(limiter) {}

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double dt,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      const auto lf_minus = low_order_flux(u[i - 1], u[i]);
      const auto hf_minus = high_order_flux(u[i - 1], u[i], dt, grid.dx);
      const auto lf_plus  = low_order_flux(u[i], u[i + 1]);
      const auto hf_plus  = high_order_flux(u[i], u[i + 1], dt, grid.dx);

      const auto F_minus  = lf_minus + limiter(u[i - 2], u[i - 1], u[i]) * (hf_minus - lf_minus);
      const auto F_plus   = lf_plus + limiter(u[i - 1], u[i], u[i + 1]) * (hf_plus - lf_plus);
      dudt[i]             = -(F_plus - F_minus) / grid.dx;
    }
  }
};

// =================================================================================================
class FD_Upwind {
 public:
  static constexpr void operator()(const Grid& grid,
                                   const Vector<double>& u,
                                   double /*dt*/,
                                   Vector<double>& dudt) noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      // 2nd order central FD -> unconditionally unstable, don't use
      // dudt[i] = -(f(u[i + 1]) - f(u[i - 1])) / (2.0 * grid.dx);

      // 1st order upwind FD
      if (u[i] >= 0.0) {
        dudt[i] = -(f(u[i]) - f(u[i - 1])) / grid.dx;
      } else {
        dudt[i] = -(f(u[i + 1]) - f(u[i])) / grid.dx;
      }
    }
  }
};

}  // namespace UPS::Burgers

#endif  // UPS_BURGERS_HPP_
