#ifndef UPS_TIME_INTEGRATOR_HPP_
#define UPS_TIME_INTEGRATOR_HPP_

#include "Grid.hpp"
#include "Vector.hpp"

namespace UPS {

// =================================================================================================
template <typename RHS, typename BCond, typename AdjustTimestep>
class TimeIntegrator {
 protected:
  RHS rhs;
  BCond bcond;
  AdjustTimestep adjust_timestep;
  Grid grid;

 public:
  Vector<double> u;

 protected:
  virtual constexpr auto do_step(double dt) noexcept -> double = 0;

 public:
  constexpr TimeIntegrator(
      Grid grid, RHS rhs, BCond bcond, AdjustTimestep adjust_timestep, const Vector<double>& u0)
      : rhs(std::move(rhs)),
        bcond(std::move(bcond)),
        adjust_timestep(std::move(adjust_timestep)),
        grid(std::move(grid)),
        u(u0) {
    IGOR_ASSERT(grid.N == u0.extent(), "Incompatible size of grid and u0");
    IGOR_ASSERT(grid.NGhost == u0.nghost(), "Incompatible size of grid and u0");
  }

  constexpr void solve(double tend) noexcept {
    double t  = 0.0;
    double dt = adjust_timestep(grid, u);
    while (t < tend) {
      dt  = adjust_timestep(grid, u);
      dt  = std::min(dt, tend - t);
      dt  = do_step(dt);
      t  += dt;
    }
  }
};

// =================================================================================================
template <typename RHS, typename BCond, typename AdjustTimestep>
class ExplicitEuler final : public TimeIntegrator<RHS, BCond, AdjustTimestep> {
  using TI = TimeIntegrator<RHS, BCond, AdjustTimestep>;
  using TI::bcond;
  using TI::grid;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  Vector<double> dudt;

  constexpr auto do_step(double dt) noexcept -> double override {
    rhs(grid, u, dudt);
    for (Index i = 0; i < u.extent(); ++i) {
      u[i] += dt * dudt[i];
    }
    bcond(u);

    return dt;
  }

 public:
  constexpr ExplicitEuler(
      Grid grid, RHS rhs, BCond bcond, AdjustTimestep adjust_timestep, const Vector<double>& u0)
      : TI(std::move(grid), std::move(rhs), std::move(bcond), std::move(adjust_timestep), u0),
        dudt(u0.extent(), u0.nghost()) {}
};

// =================================================================================================
template <typename RHS, typename BCond, typename AdjustTimestep>
class RungeKutta2 final : public TimeIntegrator<RHS, BCond, AdjustTimestep> {
  using TI = TimeIntegrator<RHS, BCond, AdjustTimestep>;
  using TI::bcond;
  using TI::grid;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  Vector<double> dudt;

  constexpr auto do_step(double dt) noexcept -> double override {
    for (Index step = 0; step < 2; ++step) {
      rhs(grid, u, dudt);
      for (Index i = 0; i < u.extent(); ++i) {
        u[i] += 0.5 * dt * dudt[i];
      }
      bcond(u);
    }

    return dt;
  }

 public:
  constexpr RungeKutta2(
      Grid grid, RHS rhs, BCond bcond, AdjustTimestep adjust_timestep, const Vector<double>& u0)
      : TI(std::move(grid), std::move(rhs), std::move(bcond), std::move(adjust_timestep), u0),
        dudt(u0.extent(), u0.nghost()) {}
};

// =================================================================================================
template <typename RHS, typename BCond, typename AdjustTimestep>
class SemiImplicitCrankNicolson final : public TimeIntegrator<RHS, BCond, AdjustTimestep> {
  using TI = TimeIntegrator<RHS, BCond, AdjustTimestep>;
  using TI::bcond;
  using TI::grid;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  Index num_subiter;
  Vector<double> dudt;
  Vector<double> u_old;

  constexpr auto do_step(double dt) noexcept -> double override {
    std::copy_n(u.data(), u.size(), u_old.data());

    for (Index sub_iter = 0; sub_iter < num_subiter; ++sub_iter) {
      // Calc mid-time values
      for (Index i = 0; i < u.extent(); ++i) {
        u[i] = (u[i] + u_old[i]) / 2.0;
      }

      // Calc update
      rhs(grid, u, dudt);
      // Update values
      for (Index i = 0; i < u.extent(); ++i) {
        u[i] = u_old[i] + dt * dudt[i];
      }
      // Apply boundary conditions
      bcond(u);
    }

    return dt;
  }

 public:
  constexpr SemiImplicitCrankNicolson(Grid grid,
                                      RHS rhs,
                                      BCond bcond,
                                      AdjustTimestep adjust_timestep,
                                      const Vector<double>& u0,
                                      Index num_subiter = 5)
      : TI(std::move(grid), std::move(rhs), std::move(bcond), std::move(adjust_timestep), u0),
        num_subiter(num_subiter),
        dudt(u0.extent(), u0.nghost()),
        u_old(u0.extent(), u0.nghost()) {
    IGOR_ASSERT(
        num_subiter > 0, "Number of sub-iterations must be greater than 0 but is {}", num_subiter);
  }
};

}  // namespace UPS

#endif  // UPS_TIME_INTEGRATOR_HPP_
