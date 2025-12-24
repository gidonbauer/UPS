#include "UPS.hpp"

auto main() -> int {
  const UPS::Index N      = 100;
  const UPS::Index NGHOST = 1;
  UPS::Vector<double> x(N, NGHOST);
  std::fill_n(x.data(), x.size(), 2.0);
  UPS::Vector<double> y(N, NGHOST);
  std::fill_n(y.data(), y.size(), 3.0);
  UPS::fma(y, x, 1.5);
  Igor::Info("y[0] = {:.1f}", y[0]);
}
