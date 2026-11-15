#pragma once
// Painted vegetation.
//
// A paint PRESET says what grows: trees (how dense, pine vs broadleaf, how
// big), bushes (how dense) and grass tufts (how dense, how tall, how dry).
// Each of the three is optional, so a preset can be "pine forest with
// undergrowth", "tall grass only" or "nothing at all".
//
// The paint MAP holds one layer per kind (trees / bushes / grass): a byte
// per 2 m cell with the id of the preset painted there, 0 = automatic (the
// procedural world). Painting a preset writes its id only into the layers
// it uses, so grass painted over a painted forest keeps the forest.
// Cells store the preset ID, not its values: editing a preset changes every
// area painted with it at the next replant.

#include <cmath>
#include <cstdint>
#include <vector>

namespace qlike {

struct PaintPreset {
    char    name[32] = "preset";
    uint8_t trees = 0, bushes = 0, grass = 0, pad = 0;   // which layers it paints
    float   tree_density = 1.0f;     // 0 = no trees .. 1 = closed forest
    float   pine_share = 0.5f;       // 0 = all broadleaf .. 1 = all pine
    float   tree_size = 1.0f;        // scale multiplier
    float   bush_density = 0.4f;     // 0 = none .. 1 = thicket
    float   grass_density = 1.0f;    // 0 = none .. 2 = very lush
    float   grass_height = 1.0f;     // scale multiplier
    float   grass_dry = 0.0f;        // 0 = fresh green .. 1 = straw
};

enum { kPaintTrees = 0, kPaintBushes = 1, kPaintGrass = 2, kPaintLayers = 3 };
constexpr int   kPaintDim = 1024;            // cells per side
constexpr float kPaintSide = 2048.0f;        // metres covered, centred on the origin
constexpr int   kPaintMaxPresets = 32;       // ids 1..31 (0 = automatic)

struct PaintMap {
    std::vector<uint8_t>     layer[kPaintLayers];   // kPaintDim * kPaintDim each (row = z)
    std::vector<PaintPreset> presets;               // [0] is a placeholder for "automatic"

    bool empty() const { return layer[0].empty(); }
    static float cell() { return kPaintSide / float(kPaintDim); }
    static int index(float x, float z) {
        const int i = int(std::floor((x + 0.5f * kPaintSide) / cell()));
        const int j = int(std::floor((z + 0.5f * kPaintSide) / cell()));
        if (i < 0 || j < 0 || i >= kPaintDim || j >= kPaintDim) return -1;
        return j * kPaintDim + i;
    }
    // The preset painted on layer `l` at (x, z), or nullptr where the layer
    // is automatic there.
    const PaintPreset* at(int l, float x, float z) const {
        if (layer[l].empty()) return nullptr;
        const int k = index(x, z);
        if (k < 0) return nullptr;
        const uint8_t id = layer[l][size_t(k)];
        if (id == 0 || id >= presets.size()) return nullptr;
        return &presets[id];
    }
};

}  // namespace qlike
