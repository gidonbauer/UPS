#ifndef UPS_VECTOR_HPP_
#define UPS_VECTOR_HPP_

#include <vector>

#include <Igor/Logging.hpp>

namespace UPS {

using Index = int;

template <typename Contained>
class Vector {
  Index m_n;
  Index m_nghost;
  std::vector<Contained> m_data{};

  [[nodiscard]] constexpr auto get_idx(Index raw_idx) const noexcept -> size_t {
    return static_cast<size_t>(raw_idx + m_nghost);
  }

 public:
  Vector(Index n, Index nghost) noexcept
      : m_n(n),
        m_nghost(nghost),
        m_data(static_cast<size_t>(n + 2 * nghost)) {}
  // Vector(const Vector& other) noexcept                              = default;
  // Vector(Vector&& other) noexcept                                   = default;
  // constexpr auto operator=(const Vector& other) noexcept -> Vector& = default;
  // constexpr auto operator=(Vector&& other) noexcept -> Vector&      = default;
  // ~Vector() noexcept                                                = default;

  [[nodiscard]] constexpr auto operator[](Index idx) noexcept -> Contained& {
    IGOR_ASSERT(idx >= -m_nghost && idx < m_n + m_nghost,
                "Index {} is out of bounds for Vector with dimension {}:{}",
                idx,
                -m_nghost,
                m_n + m_nghost);
    return m_data[get_idx(idx)];
  }

  [[nodiscard]] constexpr auto operator[](Index idx) const noexcept -> const Contained& {
    IGOR_ASSERT(idx >= -m_nghost && idx < m_n + m_nghost,
                "Index {} is out of bounds for Vector with dimension {}:{}",
                idx,
                -m_nghost,
                m_n + m_nghost);
    return m_data[get_idx(idx)];
  }

  [[nodiscard]] constexpr auto data() noexcept -> Contained* { return m_data.data(); }
  [[nodiscard]] constexpr auto data() const noexcept -> const Contained* { return m_data.data(); }

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return m_n + 2 * m_nghost; }
  [[nodiscard]] constexpr auto nghost() const noexcept -> Index { return m_nghost; }
  [[nodiscard]] constexpr auto extent() const noexcept -> Index { return m_n; }
};

// template <std::floating_point Float>
// void fma(const Vector<Float>& a,
//          const Vector<Float>& b,
//          Float c,
//          Vector<Float>& res,
//          bool include_ghost = false) noexcept {
//   // res = a + c*b
//   IGOR_ASSERT(a.extent() == b.extent() && a.extent() == res.extent(),
//               "Incompatible sizes: a.extent() = {}, b.extent() = {}, res.extent() = {}",
//               a.extent(),
//               b.extent(),
//               res.extent());
//
//   if (!include_ghost) {
// #pragma omp parallel for simd
//     for (Index i = 0; i < res.extent(); ++i) {
//       res[i] = a[i] + c * b[i];
//     }
//   } else {
//     IGOR_ASSERT(a.size() == b.size() && a.size() == res.size(),
//                 "Incompatible sizes: a.size() = {}, b.size() = {}, res.size() = {}",
//                 a.size(),
//                 b.size(),
//                 res.size());
// #pragma omp parallel for simd
//     for (Index i = -res.nghost(); i < res.extent() + res.nghost(); ++i) {
//       res[i] = a[i] + c * b[i];
//     }
//   }
// }

template <std::floating_point Float>
void fma(Vector<Float>& a, const Vector<Float>& b, Float c, bool include_ghost = false) noexcept {
  // a = a + c*b
  IGOR_ASSERT(a.extent() == b.extent(),
              "Incompatible sizes: a.extent() = {}, b.extent() = {}",
              a.extent(),
              b.extent());
  IGOR_ASSERT(!include_ghost || a.size() == b.size(),
              "Incompatible sizes: a.size() = {}, b.size() = {}",
              a.size(),
              b.size());

  const Index imin = include_ghost ? -a.nghost() : 0;
  const Index imax = include_ghost ? a.extent() + a.nghost() : a.extent();
#pragma omp parallel for simd
  for (Index i = imin; i < imax; ++i) {
    a[i] += c * b[i];
  }
}

}  // namespace UPS

#endif  // UPS_VECTOR_HPP_
