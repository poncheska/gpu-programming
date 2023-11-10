#ifndef VECTOR_HH
#define VECTOR_HH

#include <cmath>

/// N-component vector.
template <class T,int N>
class Vector {

private:
    T _data[N]{};

public:
    Vector() = default;
    ~Vector() = default;
    Vector(const Vector&) = default;
    Vector& operator=(const Vector&) = default;
    Vector(Vector&&) = default;
    Vector& operator=(Vector&&) = default;

    template <class ... Args> explicit constexpr
    Vector(Args ... args): _data{args...} {}

    explicit constexpr Vector(const T* data) {
        for (int i=0; i<N; ++i) { _data[i] = data[i]; }
    }

    T& operator[](int i) { return _data[i]; }
    T operator[](int i) const { return _data[i]; }
    T& operator()(int i) { return _data[i]; }
    T operator()(int i) const { return _data[i]; }
    T* begin() { return _data; }
    const T* begin() const { return _data; }
    T* end() { return _data+N; }
    const T* end() const { return _data+N; }
    T* data() { return _data; }
    const T* data() const { return _data; }
    static constexpr int size() { return N; }

    const Vector& operator+() const { return *this; }

    Vector operator-() const {
        Vector tmp(*this);
        for (auto& e : tmp) { e = -e; }
        return tmp;
    }

    Vector& operator=(T rhs) {
        for (int i=0; i<N; ++i) { _data[i] = rhs; }
        return *this;
    }

};

#define GPC_BINARY_OPERATOR(op) \
    template <class T,int N> \
    Vector<T,N> operator op(const Vector<T,N>& a, const Vector<T,N>& b) { \
        Vector<T,N> result(a); \
        for (int i=0; i<N; ++i) { result[i] = result[i] op b[i]; } \
        return result; \
    } \
    template <class T,int N> \
    Vector<T,N> operator op(const Vector<T,N>& a, T b) { \
        Vector<T,N> result(a); \
        for (int i=0; i<N; ++i) { result[i] = result[i] op b; } \
        return result; \
    } \
    template <class T,int N> \
    Vector<T,N> operator op(T a, const Vector<T,N>& b) { \
        Vector<T,N> result(b); \
        for (int i=0; i<N; ++i) { result[i] = a op result[i]; } \
        return result; \
    }

GPC_BINARY_OPERATOR(+);
GPC_BINARY_OPERATOR(-);
GPC_BINARY_OPERATOR(*);
GPC_BINARY_OPERATOR(/);

#define GPC_BINARY_OPERATOR_EQ(op) \
    template <class T,int N> \
    Vector<T,N>& operator op(Vector<T,N>& a, const Vector<T,N>& b) { \
        for (int i=0; i<N; ++i) { a[i] op b[i]; } \
        return a; \
    } \
    template <class T,int N> \
    Vector<T,N>& operator op(Vector<T,N>& a, T b) { \
        for (int i=0; i<N; ++i) { a[i] op b; } \
        return a; \
    }

GPC_BINARY_OPERATOR_EQ(+=);
GPC_BINARY_OPERATOR_EQ(-=);
GPC_BINARY_OPERATOR_EQ(*=);
GPC_BINARY_OPERATOR_EQ(/=);

/// Dot product.
template <class T,int N>
T dot(const Vector<T,N>& a, const Vector<T,N>& b) {
    T result{};
    for (int i=0; i<N; ++i) { result += a[i]*b[i]; }
    return result;
}

/// Raise each element of the vector to the power of two.
template <class T,int N>
T square(const Vector<T,N>& v) {
    T result{};
    for (T e : v) { result += e*e; }
    return result;
}

/// Geometric vector length.
template <class T,int N>
T length(const Vector<T,N>& v) {
    return std::sqrt(square(v));
}

/// Geometric distance between two points.
template <class T,int N>
T distance(const Vector<T,N>& a, const Vector<T,N>& b) {
    return length(b-a);
}

/// Return normalised vector.
template <class T,int N>
Vector<T,N> unit(const Vector<T,N>& v) {
    auto len = length(v);
    if (len == T{}) { return v; }
    return v/len;
}

/// Cross product of the two two-dimensional vectors.
template <class T>
T cross(const Vector<T,2>& a, const Vector<T,2>& b) {
    return a(0)*b(1)-a(1)*b(0);
}

/// Cross product of the two three-dimensional vectors.
template <class T>
Vector<T,3> cross(const Vector<T,3>& a, const Vector<T,3>& b) {
    return {a(1)*b(2)-a(2)*b(1), a(2)*b(0)-a(0)*b(2), a(0)*b(1)-a(1)*b(0)};
}

/// Raise each element of the vector to the specified power.
template <class T,int N>
Vector<T,N> pow(const Vector<T,N>& v, T p) {
    Vector<T,N> result(v);
    for (T& e : result) { e = std::pow(e,p); }
    return result;
}

#endif // vim:filetype=cpp
