#ifndef UPS_GRID_HPP_
#define UPS_GRID_HPP_

#include "Vector.hpp"

namespace UPS {

// =================================================================================================
struct Grid {
  Index N;
  Index NGhost;
  double dx;
  Vector<double> x;
  Vector<double> xm;

  constexpr Grid(double x_min, double x_max, Index N, Index NGhost) noexcept
      : N(N),
        NGhost(NGhost),
        dx{(x_max - x_min) / static_cast<double>(N)},
        x(N + 1, NGhost),
        xm(N, NGhost) {
    for (Index i = -x.nghost(); i < x.extent() + x.nghost(); ++i) {
      x[i] = static_cast<double>(i) * dx + x_min;
    }
    for (Index i = -xm.nghost(); i < xm.extent() + xm.nghost(); ++i) {
      xm[i] = (x[i] + x[i + 1]) / 2.0;
    }
  }
};

}  // namespace UPS

#endif  // UPS_GRID_HPP_
