#ifndef UPS_BOUNDARY_CONDITIONS_HPP_
#define UPS_BOUNDARY_CONDITIONS_HPP_

#include "Vector.hpp"

namespace UPS {

// =================================================================================================
class DirichletZero {
 public:
  static constexpr void operator()(Vector<double>& u) {
    for (Index i = -u.nghost(); i < 0; ++i) {
      u[i] = 0.0;
    }
    for (Index i = u.extent(); i < u.extent() + u.nghost(); ++i) {
      u[i] = 0.0;
    }
  }
};

// =================================================================================================
class NeumannZero {
 public:
  static constexpr void operator()(Vector<double>& u) {
    for (Index i = -u.nghost(); i < 0; ++i) {
      u[i] = u[0];
    }
    for (Index i = u.extent(); i < u.extent() + u.nghost(); ++i) {
      u[i] = u[u.extent() - 1];
    }
  }
};

}  // namespace UPS

#endif  // UPS_BOUNDARY_CONDITIONS_HPP_
