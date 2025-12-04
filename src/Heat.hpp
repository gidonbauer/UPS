#ifndef UPS_HEAT_HPP_
#define UPS_HEAT_HPP_

#include <Igor/Logging.hpp>

#include "Grid.hpp"

namespace UPS::Heat {

// =================================================================================================
class AdjustTimestep {
  double a;
  double CFL_safety_factor;

 public:
  constexpr AdjustTimestep(double a, double CFL_safety_factor = 0.5)
      : a(a),
        CFL_safety_factor(CFL_safety_factor) {
    IGOR_ASSERT(0.0 < a, "Diffusion coefficient must be greater than 0 but is {:.6e}", a);
    IGOR_ASSERT(0.0 < CFL_safety_factor && CFL_safety_factor <= 1.0,
                "Invalid CFL_safety_factor, must be in (0, 1] but is {:.6e}",
                CFL_safety_factor);
  }

  constexpr auto operator()(const Grid& grid, const Vector<double>& /*u*/) const noexcept
      -> double {
    return CFL_safety_factor * grid.dx * grid.dx / a;
  }
};

// =================================================================================================
class FD_Central2 {
  double a;

 public:
  constexpr FD_Central2(double a)
      : a(a) {
    IGOR_ASSERT(0.0 < a, "Diffusion coefficient must be greater than 0 but is {:.6e}", a);
  }

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double /*dt*/,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      dudt[i] = a * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (grid.dx * grid.dx);
    }
  }

  static constexpr auto name() noexcept -> std::string { return "FD_Central2"; }
};

// =================================================================================================
class FD_Central4 {
  double a;

 public:
  constexpr FD_Central4(double a)
      : a(a) {
    IGOR_ASSERT(0.0 < a, "Diffusion coefficient must be greater than 0 but is {:.6e}", a);
  }

  constexpr void operator()(const Grid& grid,
                            const Vector<double>& u,
                            double /*dt*/,
                            Vector<double>& dudt) const noexcept {
    for (Index i = 0; i < grid.N; ++i) {
      // −1/12, 16/12, −30/12, 16/12, −1/12
      dudt[i] =
          a *
          (-1.0 * u[i + 2] + 16.0 * u[i + 1] + -30.0 * u[i] + 16.0 * u[i - 1] + -1.0 * u[i - 2]) /
          (12.0 * grid.dx * grid.dx);
    }
  }

  static constexpr auto name() noexcept -> std::string { return "FD_Central4"; }
};

}  // namespace UPS::Heat

#endif  // UPS_HEAT_HPP_
