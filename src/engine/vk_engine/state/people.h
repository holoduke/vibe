#pragma once
// The people the NPC engine shows (player_body.cpp "PEOPLE"): their meshes and what decides how
// each looks. Private to player_body.cpp and npc_world.cpp - vk_engine.h only holds a pointer.
//
// Everything is on the soldier's skeleton and in the soldier's texture atlas
// (tools/character/make_people.py): one bone palette, one set of animations, one texture.

#include "engine/vk_engine.h"

#include <glm/glm.hpp>
#include <cstdint>

namespace vibe {

// How one person looks: which meshes, and the six rows of the look block (shaders/people.glsl).
struct PersonLook {
    int   outfit = 0;          // kPeopleOutfits: 0 the soldier's kit (the player's own body mesh)
    int   head = 0;            // kPeopleHeads
    bool  hair = false;        // the head's hair cards (only heads that have them)
    int   gear = 0;            // 0 nothing on the head, 1 the helmet, 2 a cap
    float height = 1.0f, width = 1.0f;   // the body drawn this much taller / broader
    glm::vec4 row[6]{};
};

// The default look: the player's soldier as the atlas has him (shaders/people.glsl rows).
PersonLook people_soldier_look();
// Who somebody is, from a number that is the same on every machine (the NPC agent and its look
// byte) and how they are dressed (npc::Dress).
PersonLook people_look(uint32_t person, uint8_t dress);

constexpr int kPeopleOutfits = 6;        // soldier, fatigues, tee, civil, work, jacket
constexpr int kPeopleHeads = 8;
constexpr uint32_t kLookRow = 62 * 3;    // the look block's first row in a character's palette region

struct VulkanEngine::PeopleState {
    SkinMesh bodies[kPeopleOutfits][kBodyLods];      // [0] unused: the soldier is body_lod_
    SkinMesh heads[kPeopleHeads][kBodyLods];
    SkinMesh hair[kPeopleHeads][kBodyLods];
    SkinMesh helmet[kBodyLods], cap[kBodyLods];
    bool     ok = false;
};

} // namespace vibe
