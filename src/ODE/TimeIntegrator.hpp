#ifndef UPS_ODE_TIME_INTEGRATOR_HPP_
#define UPS_ODE_TIME_INTEGRATOR_HPP_

#include <algorithm>

#include <Igor/Logging.hpp>

#include "Common/Vector.hpp"

namespace UPS::ODE {

// =================================================================================================
template <typename State, typename RHS>
class TimeIntegrator {
 protected:
  RHS rhs;

 public:
  State u;

 protected:
  virtual constexpr auto do_step(double dt) noexcept -> double = 0;

 public:
  constexpr TimeIntegrator(RHS rhs, const State& u0)
      : rhs(std::move(rhs)),
        u(u0) {}

  [[nodiscard]] constexpr auto solve(double dt, double tend) noexcept -> bool {
    double t = 0.0;
    while (t < tend) {
      // #ifndef IGOR_NDEBUG
      //       if (std::isnan(u) || std::isinf(u)) {
      //         Igor::Error("Bad solution: u contains NaN or Inf");
      //         return false;
      //       }
      // #endif  // IGOR_NDEBUG
      dt  = std::min(dt, tend - t);
      dt  = do_step(dt);
      t  += dt;
    }
    return true;
  }

  [[nodiscard]] virtual constexpr auto name() const noexcept -> std::string = 0;
};

// =================================================================================================
template <typename State, typename RHS>
class ExplicitEuler final : public TimeIntegrator<State, RHS> {
  using TI = TimeIntegrator<State, RHS>;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  State dudt{};

  constexpr auto do_step(double dt) noexcept -> double override {
    rhs(u, dt, dudt);
    u += dt * dudt;
    return dt;
  }

 public:
  constexpr ExplicitEuler(RHS rhs, const State& u0)
      : TI(std::move(rhs), u0) {}

  [[nodiscard]] constexpr auto name() const noexcept -> std::string override {
    return "ExplicitEuler";
  }
};

// =================================================================================================
template <typename State, typename RHS>
class RungeKutta2 final : public TimeIntegrator<State, RHS> {
  using TI = TimeIntegrator<State, RHS>;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  State dudt{};
  State u_old{};

  constexpr auto do_step(double dt) noexcept -> double override {
    u_old = u;

    rhs(u, dt, dudt);
    u = u_old + 0.5 * dt * dudt;

    rhs(u, dt, dudt);
    u = u_old + dt * dudt;

    return dt;
  }

 public:
  constexpr RungeKutta2(RHS rhs, const State& u0)
      : TI(std::move(rhs), u0) {}

  [[nodiscard]] constexpr auto name() const noexcept -> std::string override {
    return "RungeKutta2";
  }
};

// =================================================================================================
template <typename State, typename RHS>
class RungeKutta4 final : public TimeIntegrator<State, RHS> {
  using TI = TimeIntegrator<State, RHS>;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  State u_old{};
  State k1{};
  State k2{};
  State k3{};
  State k4{};

  constexpr auto do_step(double dt) noexcept -> double override {
    u_old = u;

    rhs(u, dt, k1);

    u = u_old + 0.5 * dt * k1;
    rhs(u, dt, k2);

    u = u_old + 0.5 * dt * k2;
    rhs(u, dt, k3);

    u = u_old + dt * k3;
    rhs(u, dt, k4);

    u = u_old + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    return dt;
  }

 public:
  constexpr RungeKutta4(RHS rhs, const State& u0)
      : TI(std::move(rhs), u0) {}

  [[nodiscard]] constexpr auto name() const noexcept -> std::string override {
    return "RungeKutta4";
  }
};

// =================================================================================================
template <typename State, typename RHS>
class SemiImplicitCrankNicolson final : public TimeIntegrator<State, RHS> {
  using TI = TimeIntegrator<State, RHS>;
  using TI::rhs;

 public:
  using TI::solve;
  using TI::u;

 private:
  Index num_subiter;
  State dudt{};
  State u_old{};

  constexpr auto do_step(double dt) noexcept -> double override {
    u_old = u;

    for (Index sub_iter = 0; sub_iter < num_subiter; ++sub_iter) {
      u = (u + u_old) / 2.0;
      rhs(u, dt, dudt);
      u = u_old + dt * dudt;
    }

    return dt;
  }

 public:
  constexpr SemiImplicitCrankNicolson(RHS rhs, const State& u0, Index num_subiter = 5)
      : TI(std::move(rhs), u0),
        num_subiter(num_subiter) {
    IGOR_ASSERT(
        num_subiter > 0, "Number of sub-iterations must be greater than 0 but is {}", num_subiter);
  }

  [[nodiscard]] constexpr auto name() const noexcept -> std::string override {
    return Igor::detail::format("SemiImplicitCrankNicolson-{}", num_subiter);
  }
};

}  // namespace UPS::ODE

#endif  // UPS_TIME_INTEGRATOR_HPP_
