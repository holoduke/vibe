#pragma once
// Little-endian byte streams for the animation asset files. The reader never reads past the
// end: a short or damaged file sets `ok` to false and yields zeros.
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace qlike::anim {

class ByteWriter {
public:
    std::vector<uint8_t> bytes;
    template <typename T> void pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void raw(const void* data, size_t len) {
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + len);
    }
    void str(const std::string& s) { pod(uint32_t(s.size())); raw(s.data(), s.size()); }
    void blob(const std::vector<uint8_t>& b) { pod(uint64_t(b.size())); raw(b.data(), b.size()); }
    template <typename T> void array(const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        pod(uint64_t(v.size()));
        raw(v.data(), v.size() * sizeof(T));
    }
};

class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : p_(data), end_(data + size) {}
    bool ok = true;
    size_t left() const { return size_t(end_ - p_); }
    template <typename T> T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T v{};
        if (left() < sizeof(T)) { ok = false; p_ = end_; return v; }
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }
    void raw(void* dst, size_t len) {
        if (left() < len) { ok = false; p_ = end_; std::memset(dst, 0, len); return; }
        std::memcpy(dst, p_, len);
        p_ += len;
    }
    std::string str() {
        const uint32_t n = pod<uint32_t>();
        if (!ok || left() < n) { ok = false; p_ = end_; return {}; }
        std::string s(reinterpret_cast<const char*>(p_), n);
        p_ += n;
        return s;
    }
    std::vector<uint8_t> blob() {
        const uint64_t n = pod<uint64_t>();
        if (!ok || left() < n) { ok = false; p_ = end_; return {}; }
        std::vector<uint8_t> b(p_, p_ + n);
        p_ += n;
        return b;
    }
    template <typename T> std::vector<T> array() {
        static_assert(std::is_trivially_copyable_v<T>);
        const uint64_t n = pod<uint64_t>();
        if (!ok || n > left() / sizeof(T)) { ok = false; p_ = end_; return {}; }
        std::vector<T> v(static_cast<size_t>(n));
        if (n) std::memcpy(v.data(), p_, static_cast<size_t>(n) * sizeof(T));
        p_ += n * sizeof(T);
        return v;
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

}  // namespace qlike::anim
