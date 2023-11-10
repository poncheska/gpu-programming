#ifndef THEORA_HH
#define THEORA_HH

#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>

#include <theora/theoraenc.h>

#include "color.hh"

/// \brief libogg wrapper
namespace ogg {

    using packet = ::ogg_packet;

    struct page: public ::ogg_page {};

    inline std::streambuf&
    operator<<(std::streambuf& out, const page& rhs) {
        out.sputn(reinterpret_cast<const char*>(rhs.header), rhs.header_len);
        out.sputn(reinterpret_cast<const char*>(rhs.body), rhs.body_len);
        return out;
    }

    struct stream: public ::ogg_stream_state {

    private:
        std::unique_ptr<std::streambuf> _sink;

    public:

        inline stream(int serial=0) { ::ogg_stream_init(this,serial); }
        inline ~stream() { ::ogg_stream_clear(this); }
        inline void flush() {
            page pg;
            int ret = 0;
            while ((ret = ::ogg_stream_flush(this, &pg)) != 0) {
                if (ret < 0) { throw std::runtime_error("ogg error"); }
                *this->_sink << pg;
            }
        }
        inline void write(packet& pc) {
            ::ogg_stream_packetin(this, &pc);
            page pg;
            int ret = 0;
            while ((ret = ::ogg_stream_pageout(this, &pg)) != 0) {
                if (ret < 0) { throw std::runtime_error("ogg error"); }
                *this->_sink << pg;
            }
        }
        inline void sink(std::ostream&& rhs) {
            auto* old = this->_sink.release();
            this->_sink.reset(rhs.rdbuf(old));
        }
        inline void sink(std::streambuf* rhs) { this->_sink.reset(rhs); }
        inline std::streambuf* sink() { return this->_sink.get(); }

    };

}

/// \brief libtheora wrapper
namespace thx {

    using u32 = ::ogg_uint32_t;
    using i64 = ::ogg_int64_t;

    enum class errc: int {
        fault=TH_EFAULT,
        invalid_value=TH_EINVAL,
        bad_header=TH_EBADHEADER,
        bad_format=TH_ENOTFORMAT,
        bad_version=TH_EVERSION,
        not_implemented=TH_EIMPL,
        bad_packet=TH_EBADPACKET,
        dropped_frame=TH_DUPFRAME,
    };

    class error_category: public std::error_category {
    public:
        inline const char* name() const noexcept override { return "libtheora"; }
        std::string message(int ev) const noexcept override;
    };

    const char* to_string(errc rhs);
    std::ostream& operator<<(std::ostream& out, const errc& rhs);

    extern error_category libtheora_category;

    inline std::error_condition
    make_error_condition(errc e) noexcept {
        return std::error_condition(static_cast<int>(e), libtheora_category);
    }

    #define LIBTHEORA_THROW(errcode) \
        throw ::std::system_error(static_cast<int>(errcode), ::thx::libtheora_category)

    #define LIBTHEORA_CHECK(errcode) \
        if (static_cast<::thx::errc>(errcode) != ::thx::errc(0)) { \
            LIBTHEORA_THROW(errcode); \
        }

    enum class pixel_format {
        yuv420 = TH_PF_420,
        yuv422 = TH_PF_422,
        yuv444 = TH_PF_444,
        size = TH_PF_NFORMATS,
    };

    enum class color_space {
        unspecified = TH_CS_UNSPECIFIED,
        ntsc = TH_CS_ITU_REC_470M,
        pal = TH_CS_ITU_REC_470BG,
        size = TH_CS_NSPACES,
    };

    using encoder_context = ::th_enc_ctx;

    struct bitstream: public ::th_info {
        inline bitstream() { ::th_info_init(this); }
        inline ~bitstream() { ::th_info_clear(this); }
        inline void pixel_format(::thx::pixel_format rhs) {
            this->pixel_fmt = static_cast<::th_pixel_fmt>(rhs);
        }
        inline ::thx::pixel_format pixel_format() const {
            return static_cast<::thx::pixel_format>(this->pixel_fmt);
        }
        inline void color_space(::thx::color_space rhs) {
            this->colorspace = static_cast<::th_colorspace>(rhs);
        }
        inline void dimensions(u32 w, u32 h) {
            this->picture((16-w%16)%16, (16-h%16)%16, w, h);
            this->frame(w + this->pic_x, h + this->pic_y);
        }
        inline void frame(u32 w, u32 h) {
            this->frame_width = w;
            this->frame_height = h;
        }
        inline void picture(u32 x, u32 y, u32 w, u32 h) {
            this->pic_x = x, this->pic_y = y, this->pic_width = w, this->pic_height = h;
        }
        inline void version(unsigned char major,
                            unsigned char minor,
                            unsigned char subminor) {
            this->version_major = major, this->version_minor = minor,
            this->version_subminor = subminor;
        }
        inline void frame_rate(u32 num, u32 den) {
            this->fps_numerator = num, this->fps_denominator = den;
        }
        inline void aspect_ratio(u32 num, u32 den) {
            this->aspect_numerator = num, this->aspect_denominator = den;
        }
    };

    struct comment: public ::th_comment {
        inline comment() { ::th_comment_init(this); }
        inline ~comment() { ::th_comment_clear(this); }
    };

    struct image_plane: public ::th_img_plane {
        image_plane() = default;
        inline image_plane(unsigned char* data, int w, int h) {
            this->width = w, this->height = h, this->stride = w, this->data = data;
        }
    };

    class theora_encoder {

    private:
        encoder_context* _context = nullptr;
        ogg::stream _sink;

    public:

        theora_encoder() = default;

        inline explicit
        theora_encoder(const bitstream& info):
        _context(::th_encode_alloc(&info)) {
            if (!this->_context) { throw std::invalid_argument("bad bitstream"); }
        }

        inline ~theora_encoder() { this->clear(); }

        inline void clear() { ::th_encode_free(this->_context); }

        inline void set_bitstream(const bitstream& info) {
            this->clear();
            this->_context = ::th_encode_alloc(&info);
            if (!this->_context) { throw std::invalid_argument("bad bitstream"); }
        }

        inline void
        flush_headers() {
            comment comm;
            ogg::packet pc;
            int ret = 0;
            while ((ret = ::th_encode_flushheader(this->_context, &comm, &pc)) != 0) {
                if (ret < 0) { LIBTHEORA_THROW(ret); }
                this->_sink.write(pc);
            }
            this->_sink.flush();
        }

        inline void
        encode(image_plane yuv[3], bool last=false) {
            LIBTHEORA_CHECK(::th_encode_ycbcr_in(this->_context, yuv));
            ::ogg_packet pc;
            int ret = 0;
            while ((ret = ::th_encode_packetout(this->_context, last, &pc)) != 0) {
                if (ret < 0) { LIBTHEORA_THROW(ret); }
                this->_sink.write(pc);
            }
        }

        inline ogg::stream& sink() { return this->_sink; }
        inline encoder_context* context() { return this->_context; }
        inline const encoder_context* context() const { return this->_context; }

    };

    class screen_recorder {

    private:
        std::string _filename = "out.ogv";
        int _width = 0, _height = 0;
        theora_encoder _encoder;
        std::vector<unsigned char> _rgba;

    public:

        inline
        screen_recorder(std::string filename, int width, int height):
        _filename(filename), _width(width), _height(height) {}

        inline ~screen_recorder() noexcept { flush(true); }

        void record_frame(const Pixel_matrix<float>& pixels);
        void flush(bool last);

    };

}

namespace std { template<> struct is_error_condition_enum<thx::errc>: true_type {}; }

#endif // vim:filetype=cpp
