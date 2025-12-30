#ifndef UPS_COMMON_DEFINITIONS_HPP_
#define UPS_COMMON_DEFINITIONS_HPP_

#include <cmath>

#include <Igor/Math.hpp>

namespace UPS {

// = Vector3 =======================================================================================
struct Vector3 {
  double x, y, z;
};

constexpr auto operator+(Vector3 lhs, const Vector3& rhs) noexcept -> Vector3 {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

constexpr auto operator+=(Vector3& lhs, const Vector3& rhs) noexcept -> Vector3& {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

constexpr auto operator-(Vector3 lhs, const Vector3& rhs) noexcept -> Vector3 {
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  return lhs;
}

constexpr auto operator-=(Vector3& lhs, const Vector3& rhs) noexcept -> Vector3& {
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  return lhs;
}

constexpr auto operator*(double lhs, Vector3 rhs) noexcept -> Vector3 {
  rhs.x *= lhs;
  rhs.y *= lhs;
  rhs.z *= lhs;
  return rhs;
}

constexpr auto operator*=(Vector3& lhs, double rhs) noexcept -> Vector3& {
  lhs.x *= rhs;
  lhs.y *= rhs;
  lhs.z *= rhs;
  return lhs;
}

constexpr auto operator/(Vector3 lhs, double rhs) noexcept -> Vector3 {
  lhs.x /= rhs;
  lhs.y /= rhs;
  lhs.z /= rhs;
  return lhs;
}

constexpr auto operator/=(Vector3& lhs, double rhs) noexcept -> Vector3& {
  lhs.x /= rhs;
  lhs.y /= rhs;
  lhs.z /= rhs;
  return lhs;
}

constexpr auto norm(const Vector3& vec) noexcept -> double {
  return std::sqrt(Igor::sqr(vec.x) + Igor::sqr(vec.y) + Igor::sqr(vec.z));
}

constexpr auto normalize(Vector3 vec) noexcept -> Vector3 { return vec / norm(vec); }

// = Phase State 1D ================================================================================
struct PhaseState1 {
  double pos;
  double vel;
};

[[nodiscard]] constexpr auto operator*(double lhs, PhaseState1 rhs) noexcept -> PhaseState1 {
  rhs.pos *= lhs;
  rhs.vel *= lhs;
  return rhs;
}

[[nodiscard]] constexpr auto operator/(PhaseState1 lhs, double rhs) noexcept -> PhaseState1 {
  lhs.pos /= rhs;
  lhs.vel /= rhs;
  return lhs;
}

constexpr auto operator-(PhaseState1 lhs, const PhaseState1& rhs) noexcept -> PhaseState1 {
  lhs.pos -= rhs.pos;
  lhs.vel -= rhs.vel;
  return lhs;
}

constexpr auto operator+(PhaseState1 lhs, const PhaseState1& rhs) noexcept -> PhaseState1 {
  lhs.pos += rhs.pos;
  lhs.vel += rhs.vel;
  return lhs;
}

constexpr auto operator+=(PhaseState1& lhs, const PhaseState1& rhs) noexcept -> PhaseState1& {
  lhs.pos += rhs.pos;
  lhs.vel += rhs.vel;
  return lhs;
}

// = Phase State 3D ================================================================================
struct PhaseState3 {
  Vector3 pos;
  Vector3 vel;
};

[[nodiscard]] constexpr auto operator*(double lhs, PhaseState3 rhs) noexcept -> PhaseState3 {
  rhs.pos *= lhs;
  rhs.vel *= lhs;
  return rhs;
}

[[nodiscard]] constexpr auto operator/(PhaseState3 lhs, double rhs) noexcept -> PhaseState3 {
  lhs.pos /= rhs;
  lhs.vel /= rhs;
  return lhs;
}

constexpr auto operator-(PhaseState3 lhs, const PhaseState3& rhs) noexcept -> PhaseState3 {
  lhs.pos -= rhs.pos;
  lhs.vel -= rhs.vel;
  return lhs;
}

constexpr auto operator+(PhaseState3 lhs, const PhaseState3& rhs) noexcept -> PhaseState3 {
  lhs.pos += rhs.pos;
  lhs.vel += rhs.vel;
  return lhs;
}

constexpr auto operator+=(PhaseState3& lhs, const PhaseState3& rhs) noexcept -> PhaseState3& {
  lhs.pos += rhs.pos;
  lhs.vel += rhs.vel;
  return lhs;
}

}  // namespace UPS

#endif  // UPS_COMMON_DEFINITIONS_HPP_
