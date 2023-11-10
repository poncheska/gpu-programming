#ifndef OBJECTS_HH
#define OBJECTS_HH

#include <memory>
#include <vector>

#include "vector.hh"
#include "ray.hh"

using vec = Vector<float,3>;
using ray = Ray<float,3>;

/// Location of the collision between the ray and the object.
struct Hit {
    float t{};
    vec point, normal;
    bool hit() const { return t > 0; }
    explicit operator bool() const { return hit(); }
};

/// An object from the scene.
class Object {
public:
    /// \brief Determine the location of the collision of the ray segment and the object.
    /// \param[in] t_min the start of the ray segment
    /// \param[in] t_max the end of the ray segment
    virtual Hit hit(const ray& r, float t_min, float t_max) const = 0;
};

/// A group of objects from the scene.
class Object_group: public Object {

public:
    using object_ptr = std::unique_ptr<Object>;

private:
    std::vector<object_ptr> _objects;

public:
    Hit hit(const ray& r, float t_min, float t_max) const override {
        Hit result;
        for (const auto& obj : _objects) {
            if (Hit h = obj->hit(r, t_min, t_max)) {
                result = h;
                t_max = h.t;
            }
        }
        return result;
    }
    void add(object_ptr&& obj) { _objects.emplace_back(std::move(obj)); }
};

/// Three-dimensional sphere.
class Sphere: public Object {

private:
    vec _origin;
    float _radius{};

public:
    Sphere(const vec& origin, float radius):
    _origin(origin), _radius(radius) {}
    const vec& origin() const { return _origin; }
    float radius() const { return _radius; }

    Hit hit(const ray& r, float t_min, float t_max) const override {
        Hit result;
        vec oc = r.origin() - origin();
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius()*radius();
        float discriminant = b*b - a*c;
        if (discriminant > 0) {
            float d = std::sqrt(discriminant);
            float t = (-b - d)/a;
            bool success = false;
            if (t_min < t && t < t_max) {
                success = true;
            } else {
                t = (-b + d)/a;
                if (t_min < t && t < t_max) { success = true; }
            }
            if (success) {
                result.t = t;
                result.point = r.point_at(t);
                result.normal = (result.point - origin()) / radius();
            }
        }
        return result;
    }
};

/// A camera that looks at the scene from the specified viewport.
class Camera {
//    vec _lower_left_corner{-2.f,-1.f,-1.f};
//    vec _horizontal{4.f,0.f,0.f};
//    vec _vertical{0.f,2.f,0.f};
//    vec _origin{0.f,0.f,1.f};
    // Lower left corner of the viewport.
    vec _lower_left_corner{3.02374f,-1.22628f,3.4122f};
    // Horizontal offset from the lower left corner of the viewport.
    vec _horizontal{1.18946f,0.f,-5.15434f};
    // Vertical offset from the lower left corner of the viewport.
    vec _vertical{-0.509421f,3.48757f,-0.117559f};
    // The location of the eye.
    vec _origin{13.f,2.f,3.f};

public:
    inline ray make_ray(float u, float v) const {
        return ray(_origin, _lower_left_corner + u*_horizontal + v*_vertical - _origin);
    }
    inline void move(vec delta) noexcept { this->_origin += delta; }
    const vec& origin() const { return _origin; }
};

#endif // vim:filetype=cpp
