#ifndef RANDOM_HH
#define RANDOM_HH

#include <random>

#include "scene.hh"
#include "vector.hh"

thread_local std::mt19937 prng;

// http://mathworld.wolfram.com/SpherePointPicking.html
vec random_in_unit_sphere() {
  std::uniform_real_distribution<float> dist(-1,1);
  Vector<float,4> x;
  float square_x;
  do {
    x(0) = dist(prng), x(1) = dist(prng), x(2) = dist(prng), x(3) = dist(prng);
    square_x = square(x);
  } while (square_x >= 1.f);
  // quaternions!
  return vec{2*(x(1)*x(3)+x(0)*x(2)),
             2*(x(2)*x(3)-x(0)*x(1)),
             x(0)*x(0)+x(3)*x(3)-x(1)*x(1)-x(2)*x(2)}/square_x;
}

#endif // vim:filetype=cpp
