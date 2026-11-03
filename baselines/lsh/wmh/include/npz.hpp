// Minimal reader for uncompressed .npz files as written by
// scipy.sparse.save_npz(..., compressed=False) / numpy.savez.
// Supports ZIP_STORED entries with or without zip64 extra fields, and
// .npy format versions 1-3 with little-endian i4/i8/f8/f4 arrays.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace wmh {

struct NpyArray {
    std::string descr;            // e.g. "<f8"
    std::vector<int64_t> shape;
    std::vector<char> bytes;      // raw little-endian payload
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
    template <class T> std::vector<T> as() const {
        auto conv = [&](auto tag) {
            using S = decltype(tag);
            if ((int64_t)bytes.size() != numel() * (int64_t)sizeof(S)) throw std::runtime_error("npy: size mismatch");
            std::vector<T> out(numel());
            const S* p = reinterpret_cast<const S*>(bytes.data());
            for (int64_t i = 0; i < numel(); i++) out[i] = (T)p[i];
            return out;
        };
        if (descr == "<i4") return conv(int32_t{});
        if (descr == "<i8") return conv(int64_t{});
        if (descr == "<f8") return conv(double{});
        if (descr == "<f4") return conv(float{});
        throw std::runtime_error("npy: unsupported dtype " + descr);
    }
};

inline uint16_t rd16(const unsigned char* p) { return p[0] | (p[1] << 8); }
inline uint32_t rd32(const unsigned char* p) { return (uint32_t)rd16(p) | ((uint32_t)rd16(p + 2) << 16); }
inline uint64_t rd64(const unsigned char* p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

inline NpyArray parse_npy(const unsigned char* p, uint64_t len) {
    if (len < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic");
    int major = p[6];
    uint64_t hlen, off;
    if (major == 1) { hlen = rd16(p + 8); off = 10; }
    else { hlen = rd32(p + 8); off = 12; }
    std::string h(reinterpret_cast<const char*>(p + off), hlen);
    NpyArray a;
    auto field = [&](const char* key) {
        auto k = h.find(key);
        if (k == std::string::npos) throw std::runtime_error(std::string("npy: missing ") + key);
        return k + std::strlen(key);
    };
    size_t k = field("'descr':");
    size_t q1 = h.find('\'', k), q2 = h.find('\'', q1 + 1);
    a.descr = h.substr(q1 + 1, q2 - q1 - 1);
    k = field("'fortran_order':");
    if (h.find("True", k) == h.find_first_not_of(' ', k)) throw std::runtime_error("npy: fortran order unsupported");
    k = field("'shape':");
    size_t l = h.find('(', k), r = h.find(')', l);
    std::string dims = h.substr(l + 1, r - l - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        size_t c = dims.find(',', pos);
        std::string tok = dims.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        if (tok.find_first_not_of(' ') != std::string::npos) a.shape.push_back(std::stoll(tok));
        if (c == std::string::npos) break;
        pos = c + 1;
    }
    uint64_t data_off = off + hlen;
    a.bytes.assign(reinterpret_cast<const char*>(p + data_off), reinterpret_cast<const char*>(p + len));
    return a;
}

inline std::map<std::string, NpyArray> load_npz(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(size);
    if (std::fread(buf.data(), 1, size, f) != (size_t)size) { std::fclose(f); throw std::runtime_error("short read " + path); }
    std::fclose(f);

    std::map<std::string, NpyArray> out;
    uint64_t pos = 0;
    while (pos + 30 <= (uint64_t)size && rd32(&buf[pos]) == 0x04034b50u) {
        const unsigned char* h = &buf[pos];
        uint16_t flags = rd16(h + 6), method = rd16(h + 8);
        uint64_t csize = rd32(h + 18), usize = rd32(h + 22);
        uint16_t fnlen = rd16(h + 26), exlen = rd16(h + 28);
        std::string name(reinterpret_cast<const char*>(h + 30), fnlen);
        const unsigned char* ex = h + 30 + fnlen;
        if (csize == 0xFFFFFFFFu || usize == 0xFFFFFFFFu) {
            bool found = false;
            for (uint16_t e = 0; e + 4 <= exlen;) {
                uint16_t id = rd16(ex + e), sz = rd16(ex + e + 2);
                if (id == 1) {
                    const unsigned char* z = ex + e + 4;
                    if (usize == 0xFFFFFFFFu) { usize = rd64(z); z += 8; }
                    if (csize == 0xFFFFFFFFu) { csize = rd64(z); }
                    found = true;
                }
                e += 4 + sz;
            }
            if (!found) throw std::runtime_error(path + ": zip64 sizes missing");
        }
        if (method != 0) throw std::runtime_error(path + ": compressed npz not supported (save with compressed=False)");
        if (flags & 8) throw std::runtime_error(path + ": zip data descriptors not supported");
        if (csize != usize) throw std::runtime_error(path + ": stored entry size mismatch");
        uint64_t data = pos + 30 + fnlen + exlen;
        if (data + csize > (uint64_t)size) throw std::runtime_error(path + ": truncated entry " + name);
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            std::string key = name.substr(0, name.size() - 4);
            if (key != "format") out[key] = parse_npy(&buf[data], csize);  // 'format' is a unicode scalar
        }
        pos = data + csize;
    }
    if (out.empty()) throw std::runtime_error(path + ": no arrays found");
    return out;
}

}  // namespace wmh
