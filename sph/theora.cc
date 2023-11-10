#include <fstream>
#include <iostream>
#include <sstream>

#include <GL/gl.h>

#include "theora.hh"

namespace {

    void rgb_to_yuv444(unsigned char* rgba, unsigned char* y,
                       unsigned char* u, unsigned char* v,
                       int w, int h, int i0, int j0) {
        int w16 = w+i0;
        for (int j=0; j<h; ++j) {
            for (int i=0; i<w; ++i) {
                int idx = (h-j-1)*w + i, idx_out = (j+j0)*w16 + i + i0;
                unsigned short r = rgba[4*idx + 0], g = rgba[4*idx + 1], b = rgba[4*idx + 2];
                y[idx_out] = ((66*r + 129*g + 25*b + 128) >> 8) + 16;
                u[idx_out] = ((-38*r - 74*g + 112*b + 128) >> 8) + 128;
                v[idx_out] = ((112*r - 94*g - 18*b + 128) >> 8) + 128;
            }
        }
    }

}

const char*
thx::to_string(errc rhs) {
    switch (rhs) {
        case errc::fault: return "invalid pointer";
        case errc::invalid_value: return "invalid argument";
        case errc::bad_header: return "bad header";
        case errc::bad_format: return "non-theora header";
        case errc::bad_version: return "bitstream version is too high";
        case errc::not_implemented: return "not implemented";
        case errc::bad_packet: return "bad packet";
        case errc::dropped_frame: return "dropped frame";
        default: return nullptr;
    }
}

std::ostream&
thx::operator<<(std::ostream& out, const errc& rhs) {
    if (const char* s = to_string(rhs)) { out << s; }
    else { out << static_cast<int>(rhs); }
    return out;
}


thx::error_category thx::libtheora_category;

std::string
thx::error_category::message(int ev) const noexcept {
    auto cond = static_cast<errc>(ev);
    const char* str = to_string(cond);
    if (str) { return str; }
    std::stringstream msg;
    msg << static_cast<int>(cond);
    return msg.str();
}

void
thx::screen_recorder::record_frame() {
    auto width = this->_width, height = this->_height;
    if (!this->_encoder.context()) {
        bitstream info;
        info.dimensions(width, height);
        info.pixel_format(pixel_format::yuv444);
        info.color_space(color_space::unspecified);
        info.frame_rate(60, 1);
        info.aspect_ratio(1, 1);
        info.target_bitrate = 0;
        info.quality = 63;
        /*
        std::clog << "info.target_bitrate=" << info.target_bitrate << std::endl;
        std::clog << "info.quality=" << info.quality << std::endl;
        std::clog << "info.pic_x=" << info.pic_x << std::endl;
        std::clog << "info.pic_y=" << info.pic_y << std::endl;
        std::clog << "info.pic_width=" << info.pic_width << std::endl;
        std::clog << "info.pic_height=" << info.pic_height << std::endl;
        std::clog << "info.frame_width=" << info.frame_width << std::endl;
        std::clog << "info.frame_height=" << info.frame_height << std::endl;
        */
        this->_encoder.set_bitstream(info);
        auto buf = new std::filebuf;
        buf->open(this->_filename, std::ios::out);
        this->_encoder.sink().sink(buf);
        this->_encoder.flush_headers();
    }
    auto& rgba = this->_rgba;
    rgba.resize(width*height*4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    this->flush(false);
}

void
thx::screen_recorder::flush(bool last) {
    using namespace thx;
    auto& rgba = this->_rgba;
    if (rgba.empty()) { return; }
    auto width = this->_width, height = this->_height;
    int offset_x = (16-width%16)%16, offset_y = (16-height%16)%16;
    int w16 = width + offset_x, h16 = height + offset_y;
    std::vector<unsigned char> y(w16*h16), u(w16*h16), v(w16*h16);
    rgb_to_yuv444(rgba.data(), y.data(), u.data(), v.data(), width, height, offset_x, offset_y);
    image_plane yuv[3] = {
        image_plane(y.data(), w16, h16),
        image_plane(u.data(), w16, h16),
        image_plane(v.data(), w16, h16)};
    this->_encoder.encode(yuv, last);
}
