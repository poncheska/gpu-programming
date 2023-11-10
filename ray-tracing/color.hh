#ifndef COLOR_HH
#define COLOR_HH

#include <ostream>

#include "vector.hh"

template <class T>
struct Color { T r{},g{},b{}; };

/// Write color in PPM format
template <class T> std::ostream&
operator<<(std::ostream& out, const Color<T>& rhs) {
    return out << int(rhs.r*255)
        << ' ' << int(rhs.g*255)
        << ' ' << int(rhs.b*255);
}

template <class T>
const Color<T>& to_color(const Vector<T,3>& v) {
    static_assert(sizeof(Vector<T,3>) == sizeof(Color<T>), "bad size");
    return *reinterpret_cast<const Color<T>*>(&v);
}

/// A matrix where each element is a color.
template <class T>
class Pixel_matrix {
    int _nx = 0, _ny = 0;
    std::vector<Color<T>> _pixels;
public:
    Pixel_matrix(int nx, int ny): _nx(nx), _ny(ny), _pixels(nx*ny) {}
    int nrows() const { return _ny; }
    int ncols() const { return _nx; }
    Color<T>& operator()(int i, int j) { return _pixels[j*ncols()+i]; }
    const Color<T>& operator()(int i, int j) const { return _pixels[j*ncols()+i]; }
    const std::vector<Color<T>>& pixels() const { return _pixels; }
    Color<T>& operator[](int i) { return _pixels[i]; }
    const Color<T>& operator[](int i) const { return _pixels[i]; }
    void to_rgba(unsigned char* rgba) const {
        const int npixels = nrows()*ncols();
        for (int i=0; i<npixels; ++i) {
            const auto& color = this->_pixels[i];
            rgba[4*i + 0] = int(color.r*255);
            rgba[4*i + 1] = int(color.g*255);
            rgba[4*i + 2] = int(color.b*255);
            rgba[4*i + 3] = 0;
        }
    }
};

template <class T> std::ostream&
operator<<(std::ostream& out, const Pixel_matrix<T>& rhs) {
    out << "P3\n" << rhs.ncols() << ' ' << rhs.nrows() << "\n255\n";
    for (int j=rhs.nrows()-1; j>=0; --j) {
        for (int i=0; i<rhs.ncols(); ++i) {
            out << rhs(i,j) << '\n';
        }
    }
    return out;
}

#endif // vim:filetype=cpp
