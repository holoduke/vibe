#include "engine/site_store.h"

#include <fstream>
#include <set>
#include <sstream>

namespace vibe {

namespace {
std::string strip_comment(const std::string& line) {
    const size_t h = line.find('#');
    return h == std::string::npos ? line : line.substr(0, h);
}
bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}
} // namespace

bool SiteStore::parse_index_line(const std::string& raw, SiteHeader& out) {
    std::istringstream is(strip_comment(raw));
    std::string id, east, south;
    if (!(is >> id >> east >> south)) return false;
    SiteHeader h;
    h.id = id;
    if (east == "auto") {
        h.auto_place = true;
    } else {
        try { h.east_m = std::stod(east) * 1000.0; h.south_m = std::stod(south) * 1000.0; }
        catch (...) { return false; }
    }
    if (!(is >> h.flat >> h.slope >> h.rise >> h.clearing)) return false;
    if (h.flat < 10.0f || h.slope <= h.flat) return false;
    std::string word;
    while (is >> word) h.name += (h.name.empty() ? "" : " ") + word;
    if (h.name.empty()) h.name = id;
    out = h;
    return true;
}

std::vector<SiteHeader> SiteStore::load_index() const {
    std::vector<SiteHeader> all;
    std::set<std::string> seen;
    for (const std::string& dir : { local_dir, cache_dir }) {
        std::ifstream f(dir + "/index.txt");
        std::string line;
        while (std::getline(f, line)) {
            SiteHeader h;
            if (!parse_index_line(line, h) || !seen.insert(h.id).second) continue;
            all.push_back(std::move(h));
        }
    }
    return all;
}

bool SiteStore::parse_site(const std::string& text, SiteData& out) {
    std::istringstream all(text);
    std::string raw;
    bool any = false;
    while (std::getline(all, raw)) {
        std::istringstream is(strip_comment(raw));
        std::string kind;
        if (!(is >> kind)) continue;
        if (kind == "road") {
            float x, z;
            while (is >> x >> z) out.road.push_back(glm::vec2(x, z));
            any = true;
            continue;
        }
        SiteObject o;
        o.kind = kind;
        if (!(is >> o.x >> o.z)) continue;
        if (kind == "tower") {
            std::string seed;
            is >> o.a >> seed;
            try { o.seed = seed.empty() ? 0u : uint32_t(std::stoul(seed, nullptr, 16)); } catch (...) { o.seed = 0; }
        } else if (kind == "lamp") {
            is >> o.a >> o.b;
        } else if (kind != "house" && kind != "chapel") {
            continue;                   // (a kind this build does not know: left out, the rest still stands)
        }
        out.objects.push_back(o);
        any = true;
    }
    return any;
}

bool SiteStore::load(const std::string& id, SiteData& out) const {
    if (id.empty() || id.find_first_of("/\\.:") != std::string::npos) return false;     // (an id is a plain name, never a path)
    for (const std::string& dir : { local_dir, cache_dir }) {
        std::string text;
        if (!read_file(dir + "/" + id + ".site", text)) continue;
        SiteData d;
        d.id = id;
        if (!parse_site(text, d)) continue;
        out = std::move(d);
        return true;
    }
    return false;
}

} // namespace vibe
