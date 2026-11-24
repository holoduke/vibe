#pragma once
// The terrain's temporal upscaler (shaders/terrain_rm_resolve.frag): where the low-resolution
// terrain is offset to from frame to frame. Header-only and free of the renderer, so the
// rule has a unit test (tests/test_terrain_keep.cpp).
//
// The terrain is drawn with fewer pixels than the picture has and put together over the
// frames in a kept picture that is larger than both. Each low-resolution pixel covers several
// kept pixels each way; every frame the whole low-resolution picture is moved inside its own
// pixel, so that over a round of frames its samples fall on every one of them.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

namespace vibe {

// How many places each way the low-resolution picture visits inside its pixel: as many as it
// has kept pixels under it, four at most (at a quarter size a round of 8 x 8 = 64 frames would
// never be finished - the samples then count for their neighbours too), two at least.
inline int terrain_keep_places(uint32_t keep_width, uint32_t lowres_width) {
    const float per = float(keep_width) / float(std::max(lowres_width, 1u));
    return std::clamp(int(std::ceil(per - 0.01f)), 2, 4);
}

// The offset in frame `f`: one of n x n places - the centres of an n x n grid over the pixel -
// in units of that pixel (-0.5 .. 0.5). A round is n x n frames and visits every place once.
// In an order that spreads them: any few frames in a row lie well apart, so a view that has
// just come to rest gains evenly over the whole pixel instead of a row at a time.
inline glm::vec2 terrain_keep_phase(int n, uint64_t f) {
    int x = 0, y = 0;
    if (n >= 4) {
        // (ordered-dither order: frame v goes to the cell holding v in the 4 x 4 Bayer matrix -
        //  the first four frames take one place in each quarter of the pixel)
        static const uint8_t kX[16] = { 0, 2, 2, 0, 1, 3, 3, 1, 1, 3, 3, 1, 0, 2, 2, 0 };
        static const uint8_t kY[16] = { 0, 2, 0, 2, 1, 3, 1, 3, 0, 2, 0, 2, 1, 3, 1, 3 };
        n = 4; x = kX[f & 15u]; y = kY[f & 15u];
    } else if (n == 3) {
        const int i = int((f % 9u) * 4u % 9u);      // (4 and 9 share no factor: all nine, a diagonal step at a time)
        x = i % 3; y = i / 3;
    } else {
        static const uint8_t kX[4] = { 0, 1, 1, 0 };
        static const uint8_t kY[4] = { 0, 1, 0, 1 };
        n = 2; x = kX[f & 3u]; y = kY[f & 3u];
    }
    return (glm::vec2(float(x), float(y)) + 0.5f) / float(n) - 0.5f;
}

}  // namespace vibe
