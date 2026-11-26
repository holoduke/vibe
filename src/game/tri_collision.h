#pragma once
// Collision of a walker against a triangle mesh (the rock formations: rock_formations_gpu.cpp).
//
// A walker is a capsule: a segment from just above the feet to just below the head, and a
// radius. Against a mesh it is resolved the way a character controller does it:
//   * every triangle the capsule cuts into pushes it out along the line from the nearest point
//     on the triangle to the nearest point on the segment;
//   * a push that points up steeply enough (a face one can stand on) lifts the walker straight
//     up instead - no sliding down a walkable slope, and it counts as standing on the ground;
//   * any other push (a steep face, an overhang, a ceiling) moves the walker along it, and the
//     velocity loses what went into the face: a steep face is slid down, a ledge stops a jump.
// floor_below() finds the highest walkable face under a point - the "sticky" snap that keeps a
// walker on the rock while walking down it, as the terrain clamp does for the ground.
//
// Pure geometry, no engine: unit-tested in tests/test_tri_collision.cpp.

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace vibe::tricol {

// Triangles in world space with a grid over x / z for the queries.
struct Mesh {
    std::vector<glm::vec3> v;               // three per triangle
    std::vector<glm::vec3> n;               // one per triangle (unit; 0 for a degenerate one)
    glm::vec3 lo{ 1e30f }, hi{ -1e30f };    // the box of all triangles
    float     cell = 2.0f;
    glm::ivec2 dim{ 0 };
    std::vector<uint32_t> start;            // per cell: first entry in `items` (dim.x * dim.y + 1)
    std::vector<uint32_t> items;            // triangle numbers

    // Adds a triangle (before build()).
    void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
    // Makes the grid (call once after the last add()).
    void build(float cell_size = 2.0f);
    size_t triangles() const { return n.size(); }
};

struct Contact {
    glm::vec3 push{ 0.0f };                 // how far the capsule was moved
    bool      ground = false;               // it stands on a walkable face
    glm::vec3 ground_n{ 0.0f, 1.0f, 0.0f }; // that face's normal
    bool      touched = false;              // anything was touched
};

// The capsule: segment a (bottom sphere's centre) .. b (top sphere's centre), radius r.
// walk_ny: the least normal.y of a face one can stand on. Moves `a`, `b` out of every triangle
// of the meshes (up to `iters` rounds) and takes the pushed-into part out of `vel`.
Contact resolve_capsule(const std::vector<const Mesh*>& meshes, glm::vec3& a, glm::vec3& b, float r,
                        glm::vec3& vel, float walk_ny, int iters = 4);

// The highest walkable face (normal.y >= walk_ny) crossed by the vertical line at (x, z)
// between y_hi and y_lo. False when there is none.
bool floor_below(const std::vector<const Mesh*>& meshes, float x, float z, float y_hi, float y_lo,
                 float walk_ny, float& y, glm::vec3& normal);

// The nearest point on triangle (a, b, c) to p.
glm::vec3 closest_on_triangle(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
// The nearest points between segment p0..p1 and triangle (a, b, c): on the segment (s) and
// on the triangle (t). Returns the squared distance.
float segment_triangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& a, const glm::vec3& b,
                       const glm::vec3& c, glm::vec3& s, glm::vec3& t);

}  // namespace vibe::tricol
