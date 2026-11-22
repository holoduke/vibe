#pragma once
// Where the descriptions of the planet's hand-made places ("sites") come from.
//
// The planet is procedural everywhere; here and there somebody built something - a town, a
// fort, a harbour. There may be thousands of these, and they are not part of the program:
// each is a small text file, and the engine only ever holds the one the player is at
// (vk_engine/sites.cpp). This class is the engine's only door to that data:
//
//   the index      one line per site - its id, where on the planet it lies, the shape of
//                  its ground, its name. Small (some 60 bytes a site), read whole at
//                  start-up: it is what lets the landscape, the view from orbit and the
//                  travel menu know about a place that is not loaded.
//   a site file    <id>.site - what stands there (towers, houses, a road ...). Read when the
//                  player comes near, forgotten when they leave.
//
// Both are looked for in two folders: `local` (shipped with the game: assets/sites) and
// `cache` (cache/sites: what was fetched from elsewhere). A server client plugs in WITHOUT
// the engine changing: it sets `request`, which is called (on the main thread, at most every
// few seconds per id) when a site file is wanted and in neither folder - and it has only to
// put <id>.site, or a newer index.txt, into the cache folder. The engine picks files up from
// there on its next look. Nothing here does any networking.
//
// File formats (text, '#' starts a comment):
//
//   index.txt   <id> <east_km> <south_km> <flat_m> <slope_m> <rise_m> <clearing_m> <name ...>
//               <id> auto      -          <flat_m> ...          (auto: the engine picks open sea)
//               east / south: the place in home coordinates (km east and south of home, over
//               the ground). flat / slope: the ground is level within `flat` of the centre
//               and meets the natural ground at `slope`; rise: its height above the water;
//               clearing: no woods within this radius.
//
//   <id>.site   tower     <x> <z> <style 0..3> <seed>
//               house     <x> <z>
//               chapel    <x> <z>          (with its graveyard)
//               lamp      <x> <z> <arm x> <arm z>
//               road      <x> <z> <x> <z> ...        (one line; may be given more than once: joined)
//               x east, z south of the site's centre, in metres, on its level ground.

#include <glm/vec2.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vibe {

struct SiteHeader {
    std::string id, name;
    bool   auto_place = false;          // no coordinates given: the engine chooses (sites.cpp)
    double east_m = 0.0, south_m = 0.0; // home coordinates
    float  flat = 300.0f, slope = 1500.0f, rise = 14.0f, clearing = 330.0f;
};

struct SiteObject {
    std::string kind;                   // "tower", "house", "chapel", "lamp"
    float x = 0.0f, z = 0.0f;
    float a = 0.0f, b = 0.0f;           // tower: style, -; lamp: the arm's direction
    uint32_t seed = 0;
};

struct SiteData {
    std::string id;
    std::vector<SiteObject> objects;
    std::vector<glm::vec2> road;        // the street's line (x, z)
};

class SiteStore {
public:
    std::string local_dir = "assets/sites";
    std::string cache_dir = "cache/sites";
    // A site file that is in neither folder is wanted: for a server client to fetch it into
    // `cache_dir` (see the top of this file). Unset: such a site simply stays empty land.
    std::function<void(const std::string& id)> request;

    // Every site known: the local index, then the cache's (ids not seen before).
    std::vector<SiteHeader> load_index() const;
    // The description of one site. False: not there (yet).
    bool load(const std::string& id, SiteData& out) const;

    // (for tests and tools) the same from text
    static bool parse_index_line(const std::string& line, SiteHeader& out);
    static bool parse_site(const std::string& text, SiteData& out);
};

} // namespace vibe
