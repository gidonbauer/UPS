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

}  // namespace UPS

#endif  // UPS_VECTOR_HPP_
