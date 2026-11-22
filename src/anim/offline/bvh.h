#pragma once
// BVH motion capture files (offline only).
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace vibe::anim::offline {

struct Bvh {
    struct Joint {
        std::string              name;
        int                      parent = -1;
        glm::dvec3               offset{0.0};        // from the parent, at rest (file units)
        std::vector<std::string> channels;           // "Xposition", "Zrotation", ... in file order
        int                      first_channel = 0;  // column of its first channel in a frame
        bool                     has_end = false;
        glm::dvec3               end{0.0};           // "End Site" offset
    };
    std::vector<Joint>  joints;                      // parents before children
    int                 frames = 0;
    int                 columns = 0;                 // channels per frame
    double              frame_time = 0.0;            // seconds
    std::vector<double> data;                        // frames x columns

    int    find(const std::string& name) const;
    double at(int frame, int column) const { return data[size_t(frame) * size_t(columns) + size_t(column)]; }
};

bool parse_bvh(const std::string& text, Bvh& out, std::string* why = nullptr);
bool load_bvh(const std::string& path, Bvh& out, std::string* why = nullptr);

// Forward kinematics of one frame: world rotation and position of every joint (file units,
// the file's own axes). Rotation channels are applied in file order.
void bvh_fk(const Bvh& b, int frame, std::vector<glm::dmat3>& rot, std::vector<glm::dvec3>& pos);

}  // namespace vibe::anim::offline
