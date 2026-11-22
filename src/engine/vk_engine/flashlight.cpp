// Hand flashlight (F).
//
// Held low on the right in place of the weapon. It does not sit rigidly in
// the view: the aim eases after the camera (the hand follows the head a
// moment late), and while walking the lamp bobs and rolls through the step
// cycle, so the beam swings a little over the ground ahead. Standing still
// leaves a slow breathing drift.
//
// The model is a machined tactical torch built on a lathe at start-up:
// anodised body with a diamond-knurled grip, grooves and cooling fins, a
// flared head, a crenellated steel bezel, a chrome reflector around the
// emitter, a rubber tail switch and a steel pocket clip.
//
// The light itself is a spot light in the scene UBO (flash_pos / flash_dir /
// flash_col); every lit shader traces its own shadow ray for it (objects,
// terrain, lawn), it bounces through the path-traced GI, and the beam in
// the air is integrated per pixel along the view ray (scene_ubo.glsl
// flashlight_beam), so it is a clean cone at full resolution.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/audio.h"
#include "engine/log.h"
#include "engine/window.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace qlike {

namespace {

// ---- lathe ----------------------------------------------------------
// Profile points run from the tail (+z) to the front (-z); the lamp's
// axis is local Z, the beam leaves along -Z.
struct ProfPt {
    float z, r;
    bool  crown = false;     // front edge of the strike bezel (scalloped)
};
struct LatheOpts {
    int   segs = 72;
    bool  inward = false;    // normals toward the axis (reflector)
    float knurl_z0 = 0.0f, knurl_z1 = 0.0f;   // z range with the diamond knurl (z0 > z1)
};

float knurl(float z, float th, const LatheOpts& o) {
    if (!(o.knurl_z0 > o.knurl_z1) || z > o.knurl_z0 || z < o.knurl_z1) return 0.0f;
    // Two opposite helices crossing = diamond pattern; faded at both ends.
    const float n = 30.0f, k = 1500.0f;
    const float m = std::cos(n * th + k * z) * std::cos(n * th - k * z);
    const float edge = glm::smoothstep(0.0f, 0.004f, o.knurl_z0 - z) * glm::smoothstep(0.0f, 0.004f, z - o.knurl_z1);
    return 0.00045f * std::max(m, 0.0f) * edge;
}

glm::vec3 lathe_pos(const ProfPt& a, const ProfPt& b, float t, float th, const LatheOpts& o) {
    // Scallops of the strike bezel: six notches cut back from the front edge.
    const float notch = 0.0032f * glm::smoothstep(0.25f, 0.75f, 0.5f + 0.5f * std::cos(6.0f * th));
    const float za = a.z + (a.crown ? notch : 0.0f), zb = b.z + (b.crown ? notch : 0.0f);
    const float z = glm::mix(za, zb, t);
    const float r = glm::mix(a.r, b.r, t) + knurl(z, th, o);
    return glm::vec3(std::cos(th) * r, std::sin(th) * r, z);
}

// One strip per profile segment (hard edges between segments, as on a
// machined part); normals from the surface itself, so the knurl and the
// scallops shade correctly.
void lathe(std::vector<Vertex>& v, std::vector<uint32_t>& ix, const std::vector<ProfPt>& prof, const LatheOpts& o) {
    const float two_pi = 6.2831853f;
    for (size_t s = 0; s + 1 < prof.size(); ++s) {
        const ProfPt& a = prof[s];
        const ProfPt& b = prof[s + 1];
        const float len = std::sqrt((a.z - b.z) * (a.z - b.z) + (a.r - b.r) * (a.r - b.r));
        if (len < 1e-6f) continue;
        const bool knurled = o.knurl_z0 > o.knurl_z1 && std::min(a.z, b.z) < o.knurl_z0 && std::max(a.z, b.z) > o.knurl_z1;
        const int rows = knurled ? std::max(2, int(len / 0.0007f)) : 1;
        const int segs = knurled ? o.segs * 3 : o.segs;
        // Reference normal in the (z, r) plane: (dr, -dz) points outward for a
        // tail-to-front profile.
        glm::vec2 ref(b.r - a.r, -(b.z - a.z));
        ref = glm::normalize(ref) * (o.inward ? -1.0f : 1.0f);
        const uint32_t base = uint32_t(v.size());
        for (int i = 0; i <= rows; ++i)
            for (int j = 0; j <= segs; ++j) {
                const float t = float(i) / float(rows), th = two_pi * float(j) / float(segs);
                const glm::vec3 p = lathe_pos(a, b, t, th, o);
                const float e = 1e-3f;
                const glm::vec3 dth = lathe_pos(a, b, t, th + e, o) - lathe_pos(a, b, t, th - e, o);
                const glm::vec3 dt = lathe_pos(a, b, std::min(t + 0.02f, 1.0f), th, o) - lathe_pos(a, b, std::max(t - 0.02f, 0.0f), th, o);
                glm::vec3 n = glm::cross(dth, dt);
                const glm::vec3 refn(std::cos(th) * ref.y, std::sin(th) * ref.y, ref.x);
                if (glm::dot(n, n) < 1e-16f) n = refn; else n = glm::normalize(n);
                if (glm::dot(n, refn) < 0.0f) n = -n;
                v.push_back(Vertex{ p, n, glm::vec2(float(j) / float(segs), t) });
            }
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < segs; ++j) {
                const uint32_t i0 = base + uint32_t(i * (segs + 1) + j), i1 = i0 + 1;
                const uint32_t i2 = i0 + uint32_t(segs + 1), i3 = i2 + 1;
                // Both windings: the shell is thin and seen from every side (a
                // single winding left parts of it culled - see-through holes).
                ix.insert(ix.end(), { i0, i2, i1, i1, i2, i3, i0, i1, i2, i1, i3, i2 });
            }
    }
}

}  // namespace

// ---------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------

void VulkanEngine::init_flashlight_meshes() {
    auto build = [&](Mesh& out, const std::vector<ProfPt>& prof, const LatheOpts& o) {
        std::vector<Vertex> v;
        std::vector<uint32_t> ix;
        lathe(v, ix, prof, o);
        out = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                    v.data(), uint32_t(v.size()), ix.data(), uint32_t(ix.size()));
    };
    // 0: body + head, anodised aluminium (metres; tail at +z, front at -z).
    {
        LatheOpts o; o.knurl_z0 = 0.040f; o.knurl_z1 = -0.036f;
        build(flash_mesh_[0], {
            { 0.0760f, 0.0000f }, { 0.0760f, 0.0108f }, { 0.0745f, 0.0126f }, { 0.0700f, 0.0134f },   // tail cap
            { 0.0560f, 0.0134f }, { 0.0548f, 0.0122f }, { 0.0520f, 0.0122f }, { 0.0508f, 0.0134f },   // groove
            { 0.0460f, 0.0134f }, { 0.0448f, 0.0126f }, { 0.0420f, 0.0126f },
            { 0.0400f, 0.0126f }, { -0.0360f, 0.0126f },                                              // knurled grip
            { -0.0380f, 0.0126f }, { -0.0392f, 0.0118f }, { -0.0430f, 0.0118f }, { -0.0442f, 0.0131f },  // groove
            { -0.0560f, 0.0131f },
            { -0.0572f, 0.0148f }, { -0.0590f, 0.0148f }, { -0.0598f, 0.0134f }, { -0.0612f, 0.0134f },  // cooling fins
            { -0.0620f, 0.0148f }, { -0.0638f, 0.0148f }, { -0.0646f, 0.0134f }, { -0.0660f, 0.0134f },
            { -0.0668f, 0.0148f }, { -0.0686f, 0.0148f }, { -0.0694f, 0.0136f }, { -0.0720f, 0.0136f },
            { -0.0900f, 0.0182f },                                                                    // flare
            { -0.0915f, 0.0188f }, { -0.1110f, 0.0188f }, { -0.1118f, 0.0180f }, { -0.1118f, 0.0160f },   // head
        }, o);
    }
    // 1: strike bezel, stainless steel, scalloped front edge.
    build(flash_mesh_[1], {
        { -0.1100f, 0.0160f }, { -0.1100f, 0.0194f }, { -0.1118f, 0.0200f }, { -0.1205f, 0.0200f },
        { -0.1222f, 0.0194f, true }, { -0.1222f, 0.0166f, true }, { -0.1150f, 0.0160f },
    }, LatheOpts{});
    // 2: reflector, chrome, a deep dish from the bezel down to the emitter.
    {
        LatheOpts o; o.inward = true;
        build(flash_mesh_[2], {
            { -0.1150f, 0.0160f }, { -0.1100f, 0.0146f }, { -0.1040f, 0.0124f }, { -0.0985f, 0.0098f },
            { -0.0945f, 0.0070f }, { -0.0920f, 0.0042f }, { -0.0912f, 0.0030f },
        }, o);
    }
    // 3: emitter (the LED die), a small disc at the bottom of the dish.
    build(flash_mesh_[3], { { -0.0914f, 0.0030f }, { -0.0914f, 0.0000f } }, LatheOpts{});
    // 4: front glass, a disc just inside the bezel.
    build(flash_mesh_[4], { { -0.1160f, 0.0160f }, { -0.1160f, 0.0000f } }, LatheOpts{});
    // 6: unit sphere (finger joints, palm, wrist).
    {
        std::vector<ProfPt> pr;
        for (int i = 0; i <= 14; ++i) {
            const float a = 3.14159265f * float(i) / 14.0f;
            pr.push_back({ std::cos(a), std::sin(a) });
        }
        LatheOpts o; o.segs = 28;
        build(flash_mesh_[6], pr, o);
    }
    // 7: forearm in a jacket sleeve: from the elbow end (+z, open, off screen)
    // down to the cuff at the wrist (z = 0).
    {
        LatheOpts o; o.segs = 40;
        build(flash_mesh_[7], {
            { 0.340f, 0.052f }, { 0.240f, 0.050f }, { 0.120f, 0.043f }, { 0.050f, 0.037f }, { 0.022f, 0.0345f },
            { 0.020f, 0.0365f }, { 0.004f, 0.0365f }, { 0.000f, 0.0345f }, { 0.000f, 0.0270f },
        }, o);
    }
    // 5: tail switch boot, a rubber dome.
    build(flash_mesh_[5], {
        { 0.0760f, 0.0098f }, { 0.0790f, 0.0094f }, { 0.0812f, 0.0080f }, { 0.0826f, 0.0054f }, { 0.0832f, 0.0000f },
    }, LatheOpts{});
}

void VulkanEngine::destroy_flashlight_meshes() {
    for (Mesh& m : flash_mesh_)
        if (m.vertex_buffer) { destroy_mesh(allocator_, m); m = Mesh{}; }
}

void VulkanEngine::draw_flashlight_model(VkCommandBuffer cmd, const glm::mat4& vp) {
    if (!flash_mesh_[0].vertex_buffer) return;
    const glm::mat4 base(glm::vec4(flash_right_, 0.0f), glm::vec4(flash_up_, 0.0f),
                         glm::vec4(-flash_fwd_, 0.0f), glm::vec4(flash_pos_, 1.0f));
    // material: -1 plain, -2 brushed steel, -3 anodised metal, -4 chrome (cube.frag surface_rough_metal)
    auto part = [&](const Mesh& m, const glm::mat4& local, glm::vec4 color, float material, glm::vec3 emissive, bool full) {
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, m.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        PushConstants pc{};
        const glm::mat4 model = base * local;
        pc.mvp = vp * model;
        pc.model = model;
        pc.prev_mvp = pc.mvp;                    // hand-held: no motion vectors
        if (rt_.fsr2_enabled) {
            glm::mat4 J(1.0f);
            const glm::vec2 dj = current_frame_view_.jitter - prev_jitter_;
            J[3][0] = dj.x;
            J[3][1] = dj.y;
            pc.prev_mvp = J * pc.mvp;
        }
        pc.color = color;
        pc.emissive = glm::vec4(emissive, full ? 1.0f : 0.0f);
        pc.tex_params = glm::vec4(material, -1.0f, 1.0f, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, m.index_count, 1, 0, 0, 0);
    };
    const glm::mat4 I(1.0f);
    const glm::vec3 none(0.0f);
    const glm::vec3 warm(1.0f, 0.95f, 0.84f);
    part(flash_mesh_[0], I, glm::vec4(0.060f, 0.062f, 0.068f, 1.0f), -3.0f, none, false);       // body: black anodised
    part(flash_mesh_[1], I, glm::vec4(0.62f, 0.63f, 0.65f, 1.0f), -2.0f, none, false);           // bezel: steel
    part(flash_mesh_[2], I, glm::vec4(0.92f, 0.92f, 0.90f, 1.0f), -4.0f, warm * 1.2f, false);    // reflector: lit chrome
    // The emitter and the glowing front glass are fully emissive, and that
    // material draws on top of everything: from behind the lamp (the only way
    // the player ever sees it) they showed through the head as a white disc.
    // Only drawn when the lamp actually faces the eye.
    if (glm::dot(flash_fwd_, glm::vec3(current_frame_view_.inv_view[2])) > 0.2f) {
        part(flash_mesh_[3], I, glm::vec4(warm, 1.0f), -1.0f, warm * 30.0f, true);               // emitter
        part(flash_mesh_[4], I, glm::vec4(warm, 1.0f), -1.0f, warm * 2.2f, true);                // glass glow
    }
    part(flash_mesh_[5], I, glm::vec4(0.020f, 0.020f, 0.021f, 1.0f), -1.0f, none, false);        // rubber switch
    // Pocket clip: a steel strip along the top, with its mounting ring.
    auto box = [&](glm::vec3 c, glm::vec3 half, glm::vec4 col, float mat) {
        part(cube_mesh_, glm::translate(I, c) * glm::scale(I, half * 2.0f), col, mat, none, false);
    };
    const glm::vec4 steel(0.60f, 0.61f, 0.63f, 1.0f);
    // (on the left side, clear of the thumb)
    box(glm::vec3(-0.0146f, 0.0f, 0.0250f), glm::vec3(0.0005f, 0.0032f, 0.0300f), steel, -2.0f);
    box(glm::vec3(-0.0142f, 0.0f, 0.0535f), glm::vec3(0.0012f, 0.0040f, 0.0016f), steel, -2.0f);
    box(glm::vec3(-0.0152f, 0.0f, -0.0040f), glm::vec3(0.0010f, 0.0032f, 0.0020f), steel, -2.0f);

    if (body_drawn_) return;          // the soldier's own hand holds it (player_body.cpp)
    // ---- The hand that holds it: a gloved right fist around the grip
    // (palm on the right, fingers wrapped under and round to the left,
    // thumb along the top), and the forearm in a jacket sleeve running
    // back, down and out of the picture. Joints are spheres, bones are
    // cylinders. ----
    const glm::vec4 glove(0.030f, 0.031f, 0.034f, 1.0f), sleeve(0.085f, 0.105f, 0.075f, 1.0f);
    auto ball = [&](glm::vec3 c, glm::vec3 radii, glm::vec4 col) {
        part(flash_mesh_[6], glm::translate(I, c) * glm::scale(I, radii), col, -1.0f, none, false);
    };
    // Matrix whose Y axis runs from a to b (cylinder_mesh_: radius 1, half height 1 along Y).
    auto bone = [&](glm::vec3 a, glm::vec3 b, float r, glm::vec4 col) {
        const glm::vec3 d = b - a;
        const float len = glm::length(d);
        if (len < 1e-5f) return;
        const glm::vec3 y = d / len;
        const glm::vec3 x = glm::normalize(glm::cross(y, std::abs(y.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
        const glm::vec3 z = glm::cross(x, y);
        const glm::mat4 R(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f), glm::vec4(z, 0.0f), glm::vec4((a + b) * 0.5f, 1.0f));
        part(cylinder_mesh_, R * glm::scale(I, glm::vec3(r, len * 0.5f, r)), col, -1.0f, none, false);
    };
    const float kGrip = 0.0126f + 0.0058f;                 // torch radius + half a finger
    auto ring = [&](float deg, float zc) {
        const float a = glm::radians(deg);
        return glm::vec3(std::cos(a) * kGrip, std::sin(a) * kGrip, zc);
    };
    // Four fingers: knuckle on the right, wrapped under the torch, tips on the left.
    for (int k = 0; k < 4; ++k) {
        const float zc = 0.0040f + 0.0118f * float(k);
        const float fr = (k == 3) ? 0.0050f : 0.0057f;      // the little finger is thinner
        const float curl = (k == 3) ? -188.0f : -202.0f;
        const glm::vec3 j0 = ring(24.0f, zc), j1 = ring(-52.0f, zc), j2 = ring(-128.0f, zc), j3 = ring(curl, zc);
        bone(j0, j1, fr, glove); bone(j1, j2, fr * 0.96f, glove); bone(j2, j3, fr * 0.90f, glove);
        ball(j0, glm::vec3(fr * 1.05f), glove); ball(j1, glm::vec3(fr * 1.0f), glove);
        ball(j2, glm::vec3(fr * 0.96f), glove); ball(j3, glm::vec3(fr * 0.90f), glove);
    }
    // Thumb along the top, pointing forward.
    {
        const glm::vec3 t0(0.0175f, 0.0100f, 0.0400f), t1(0.0085f, 0.0188f, 0.0190f), t2(0.0015f, 0.0196f, -0.0010f);
        bone(t0, t1, 0.0068f, glove); bone(t1, t2, 0.0061f, glove);
        ball(t0, glm::vec3(0.0078f), glove); ball(t1, glm::vec3(0.0067f), glove); ball(t2, glm::vec3(0.0060f), glove);
    }
    // Palm / back of the hand, heel of the hand, wrist.
    ball(glm::vec3(0.0225f, -0.0030f, 0.0235f), glm::vec3(0.0150f, 0.0245f, 0.0270f), glove);
    ball(glm::vec3(0.0260f, -0.0120f, 0.0440f), glm::vec3(0.0200f, 0.0230f, 0.0220f), glove);
    const glm::vec3 wrist(0.0330f, -0.0200f, 0.0640f);
    ball(wrist, glm::vec3(0.0265f, 0.0255f, 0.0265f), glove);
    // Forearm: its local +Z runs from the cuff at the wrist toward the elbow.
    {
        const glm::vec3 az = glm::normalize(glm::vec3(0.40f, -0.62f, 1.0f));
        const glm::vec3 ax = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), az));
        const glm::vec3 ay = glm::cross(az, ax);
        const glm::mat4 A(glm::vec4(ax, 0.0f), glm::vec4(ay, 0.0f), glm::vec4(az, 0.0f), glm::vec4(wrist + az * 0.010f, 1.0f));
        part(flash_mesh_[7], A, sleeve, -1.0f, none, false);
    }
}

// ---------------------------------------------------------------------
// Sway
// ---------------------------------------------------------------------

void VulkanEngine::tick_flashlight(float dt) {
    // F toggles (edge), not while typing in a text field or in the menu.
    const bool down = in_flashlight_;      // (F, or the controller's button: menu, Controls)
    const bool typing = imgui_initialized_ && ImGui::GetIO().WantTextInput;
    static const bool kTestOn = std::getenv("QLIKE_TEST_FLASHLIGHT") != nullptr;
    if (kTestOn && frame_number_ == 5) flashlight_on_ = true;
    if (down && !flash_key_prev_ && state_ == State::Playing && !typing) {
        flashlight_on_ = !flashlight_on_;
        flash_init_ = false;
        if (audio_) audio_->play_local(ClipID::Jump, 0.25f, 0.02f, 1.9f);   // a small click
        log::infof("[flashlight] %s", flashlight_on_ ? "on" : "off");
    }
    flash_key_prev_ = down;
    if (!flashlight_on_) return;

    dt = glm::clamp(dt, 0.0f, 0.05f);
    if (!flash_init_) {
        flash_yaw_ = player_.yaw; flash_pitch_ = player_.pitch;
        flash_phase_ = 0.0f; flash_move_ = 0.0f;
        flash_init_ = true;
    }
    // The hand follows the head a moment late.
    const float follow = 1.0f - std::exp(-dt * 13.0f);
    flash_yaw_   += (player_.yaw - flash_yaw_) * follow;
    flash_pitch_ += (player_.pitch - flash_pitch_) * follow;

    // Step cycle: advances with the distance walked; fades out in the air
    // and when standing.
    const float speed = glm::length(glm::vec2(player_.velocity.x, player_.velocity.z));
    const float want_move = (player_.on_ground || player_swimming_) ? glm::clamp(speed / 6.0f, 0.0f, 1.3f) : 0.15f;
    flash_move_ += (want_move - flash_move_) * (1.0f - std::exp(-dt * 7.0f));
    flash_phase_ += speed * dt * 1.25f;
    const float t = static_cast<float>(glfwGetTime());
    const float s1 = std::sin(flash_phase_), s2 = std::sin(flash_phase_ * 2.0f);
    // Position bob (m) and aim bob (rad): a sideways sway once per two steps,
    // a vertical dip every step, plus a slow idle drift.
    const float bob_x = 0.014f * s1 * flash_move_ + 0.0020f * std::sin(t * 0.9f);
    // (a smooth dip per step: |sin| has a kink at every footfall, which read as a jerk)
    const float bob_y = -0.009f * (0.5f - 0.5f * std::cos(2.0f * flash_phase_)) * flash_move_ + 0.0025f * std::sin(t * 1.3f + 1.0f);
    const float aim_yaw   = 0.016f * s1 * flash_move_ + 0.0030f * std::sin(t * 0.7f + 2.0f);
    const float aim_pitch = 0.011f * s2 * flash_move_ + 0.0030f * std::sin(t * 1.1f);
    const float roll      = 0.035f * s1 * flash_move_;

    auto basis = [](float yaw, float pitch, glm::vec3& f, glm::vec3& r, glm::vec3& u) {
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        f = glm::vec3(std::sin(yaw) * cp, sp, -std::cos(yaw) * cp);
        r = glm::vec3(std::cos(yaw), 0.0f, std::sin(yaw));
        u = glm::cross(r, f);
    };
    glm::vec3 cf, cr, cu;                                   // where the head looks
    basis(player_.yaw, player_.pitch, cf, cr, cu);
    glm::vec3 f, r, u;                                      // where the lamp points
    // (held on the right, so it is toed in a touch toward what the eye looks at)
    basis(flash_yaw_ + aim_yaw - 0.014f, flash_pitch_ + aim_pitch + 0.006f, f, r, u);
    // Roll about the beam axis.
    const float cr_ = std::cos(roll), sr_ = std::sin(roll);
    const glm::vec3 r2 = r * cr_ + u * sr_, u2 = u * cr_ - r * sr_;

    flash_fwd_ = glm::normalize(f);
    flash_right_ = r2;
    flash_up_ = u2;
    // Held low on the right, a forearm's length ahead.
    // Stored relative to the eye: the camera position is interpolated between
    // physics ticks, and the lamp must ride on THAT position (anchored to the
    // raw tick position it shook against the view while walking). The frame
    // re-anchors it once the camera is known (begin_frame_commands).
    flash_off_ = cr * (0.118f + bob_x) + cu * (-0.118f + bob_y) + cf * 0.275f;
    flash_pos_ = player_.eye_position() + flash_off_;
    // Test hook: QLIKE_TEST_FLASHVIEW=<deg> holds the lamp in front of the eye, turned by that angle (model inspection).
    static const char* kView = std::getenv("QLIKE_TEST_FLASHVIEW");
    if (kView) {
        basis(player_.yaw + glm::radians(float(std::atof(kView))), player_.pitch - 0.25f, f, r, u);
        flash_fwd_ = f; flash_right_ = r; flash_up_ = u;
        flash_off_ = cf * 0.24f - cu * 0.01f;
        flash_pos_ = player_.eye_position() + flash_off_;
    }
    // Test hook: QLIKE_TEST_FLASHSIDE=<m> moves the lamp that far to the right (shadow check).
    static const char* kSide = std::getenv("QLIKE_TEST_FLASHSIDE");
    if (kSide) { flash_off_ += cr * float(std::atof(kSide)); flash_pos_ = player_.eye_position() + flash_off_; }
}

}  // namespace qlike
