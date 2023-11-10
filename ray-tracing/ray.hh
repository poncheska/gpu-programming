#ifndef RAY_HH
#define RAY_HH

#include "vector.hh"

/// A ray specified by the origin and the direction vector.
template <class T,int N>
class Ray {
    Vector<T,N> _origin, _direction;
public:
    Ray() = default;
    Ray(const Vector<T,N>& origin, const Vector<T,N>& direction):
    _origin(origin), _direction(direction) {}
    const Vector<T,N>& origin() const { return _origin; }
    const Vector<T,N>& direction() const { return _direction; }
    /// Return the point that lies on the ray at the specified distance from the origin.
    Vector<T,N> point_at(T t) const { return origin() + t*direction(); }
};

#endif // vim:filetype=cpp
