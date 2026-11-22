// Sites: the hand-made places of the planet other than home.
//
// The planet is procedural everywhere; a site is a spot where something was built on it - a
// town, a fort. There may be thousands, and none of them is in the program: each is a line in
// an index (where it is, the shape of its ground) and a small text file (what stands there),
// read through engine/site_store.h - from the game's own folder, or from a cache folder that
// a server client fills.
//
// What makes thousands cost nothing:
//   * The engine holds ONE site's content at a time - the one the player is at. Every site
//     has a chart of its own in which its content lies at the same small engine coordinates,
//     (0, kSiteOffZ) (engine/globe.h), so "the site" is one slot: site_meshes_ / site_cols_ /
//     site_road_ / its lanterns. It is filled within kSiteLoad of a site (load_site) and
//     emptied beyond kSiteUnload (unload_site); it is drawn, traced and collided with only
//     while the engine is in that site's chart (site_here).
//   * The landscape knows a site only by its shape (flat top, slope, rise), and only the few
//     near the player: globe::sites_nearby(), republished when the set changes.
//   * What is scanned every frame is the index alone: one dot product a site.
//
//   init_sites()      the index -> globe's table (start-up)
//   update_sites()    which sites are near; load / unload the slot (every frame)
//   load_site()       a site file -> buildings, colliders, street, lanterns
//   travel_to_site()  the menu's "travel to"
//   site_uniform()    what the shaders need of the site in sight

#include "engine/vk_engine/internal.h"
#include "engine/globe.h"
#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/window.h"
#include "game/level.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace vibe {

namespace {
// A site given as "auto" in the index: the first spot of open sea found on a fixed walk
// outward from home (600 km and on), with nothing but water for 2.5 km round it - so that an
// island stands clear - and a coast within 60 km, so there is land to see from it.
glm::dvec3 find_open_sea(float wl, bool& found) {
    const globe::Chart home;
    auto nat = [&](const glm::dvec3& G) { return world_natural_height(G, wl, 6); };
    found = false;
    for (double dist = 600000.0; dist <= 1500000.0; dist += 40000.0) {
        for (int b = 0; b < 32; ++b) {
            const double ang = 0.35 + 6.283185307 * double(b) / 32.0;
            const double hx = dist * std::sin(ang), hz = dist * std::cos(ang);
            const glm::dvec3 G = globe::to_planet(home, hx, hz);
            if (nat(G) > wl - 6.0f) continue;
            const globe::Chart at = globe::far_chart(globe::frame_at(home, hx, hz));
            bool open = true, coast = false;
            for (int k = 0; k < 12 && open; ++k) {
                const double a2 = 6.283185307 * double(k) / 12.0;
                if (nat(globe::to_planet(at, globe::kFarX + 2500.0 * std::cos(a2), 2500.0 * std::sin(a2))) > wl - 3.0f) open = false;
            }
            if (!open) continue;
            for (int k = 0; k < 24 && !coast; ++k) {
                const double a2 = 6.283185307 * double(k) / 24.0;
                for (double r : { 15000.0, 30000.0, 60000.0 })
                    if (nat(globe::to_planet(at, globe::kFarX + r * std::cos(a2), r * std::sin(a2))) > wl + 8.0f) coast = true;
            }
            if (!coast) continue;
            found = true;
            return G;
        }
    }
    return globe::to_planet(home, 700000.0 * std::sin(1.0), 700000.0 * std::cos(1.0));
}
} // namespace

// The index -> the table of sites (engine/globe.h). Needs the water level; must run before
// any ground height is asked for.
void VulkanEngine::init_sites() {
    if (globe::site_count() > 1) return;
    std::error_code ec;
    std::filesystem::create_directories(site_store_.cache_dir, ec);
    const globe::Chart home;
    int autos = 0;
    for (const SiteHeader& h : site_store_.load_index()) {
        globe::SiteShape sh;
        sh.flat = h.flat; sh.slope = h.slope; sh.rise = h.rise; sh.clearing = h.clearing;
        if (h.auto_place) {
            if (autos++ > 0) { log::warnf("[site] %s: only one site may be placed by \"auto\" - skipped", h.id.c_str()); continue; }
            bool found = false;
            sh.G = find_open_sea(rt_.water_level, found);
            if (!found) log::warnf("[site] %s: no open water found, the fallback spot", h.id.c_str());
        } else {
            sh.G = globe::to_planet(home, h.east_m, h.south_m);
        }
        const int k = globe::add_site(h.id, h.name, sh);
        const glm::dvec2 hz = globe::home_az(globe::site(k).G);
        if (globe::site_count() <= 12)
            log::infof("[site] %s (%s): site %d, %.0f km from home (%.3f km east, %.3f km south)", h.name.c_str(), h.id.c_str(), k,
                       glm::length(hz) * 0.001, hz.x * 0.001, hz.y * 0.001);
    }
    log::infof("[site] %d sites in the index (%s, %s)", globe::site_count() - 1, site_store_.local_dir.c_str(), site_store_.cache_dir.c_str());
}

// Every frame: which sites are near (for the landscape), which is nearest (for the chart, the
// view from high up), and whether the slot holds the right one.
void VulkanEngine::update_sites() {
    // Test hook: VIBE_TEST_SITE_HOP=<frame>,<site> travels to that site at that frame (to see
    // one site's content leave the slot and another's come in, in one run).
    if (static const char* hop = std::getenv("VIBE_TEST_SITE_HOP")) {
        static int hop_frame = -1, hop_site = 0;
        if (hop_frame < 0) { hop_frame = 0; std::sscanf(hop, "%d,%d", &hop_frame, &hop_site); }
        if (int(frame_number_) == hop_frame) travel_to_site(hop_site);
    }
    const globe::Chart& c = globe::current();
    const glm::dvec3 G = globe::to_planet(c, double(player_.position.x), double(player_.position.z));
    constexpr double kNear = 300000.0;
    const double cos_near = std::cos(kNear / globe::kR);
    std::vector<int> close_by;
    int best = -1; double best_dot = -2.0;
    for (int k = 1; k < globe::site_count(); ++k) {
        const double d = glm::dot(G, globe::site(k).G);
        if (d > best_dot) { best_dot = d; best = k; }
        if (d > cos_near) close_by.push_back(k);
    }
    site_near_ = best;
    site_near_dist_ = best >= 0 ? globe::ground_distance(G, globe::site(best).G) : 1e12;
    if (close_by != site_nearby_idx_) {
        std::vector<globe::SiteShape> shapes;
        for (int k : close_by) shapes.push_back(globe::site(k));
        globe::set_sites_nearby(std::move(shapes));
        site_nearby_idx_ = std::move(close_by);
    }
    // The slot. (The retired content of the site before waits a few frames: the frames in
    // flight and their ray-tracing structure may still refer to it.)
    for (auto it = site_retired_.begin(); it != site_retired_.end();) {
        if (--it->first > 0) { ++it; continue; }
        destroy_mesh_parts(it->second);
        it = site_retired_.erase(it);
    }
    int want = -1;
    if (site_near_ > 0 && site_near_dist_ < globe::kSiteLoad) want = site_near_;
    else if (site_loaded_ > 0 && globe::ground_distance(G, globe::site(site_loaded_).G) < globe::kSiteUnload) want = site_loaded_;
    if (want == site_loaded_) return;
    if (site_loaded_ > 0) unload_site();
    // (one site's content at a time, strictly: the next is not built until the last one's GPU
    // resources are gone - a dozen frames. Building the new ray-tracing structures while the
    // old ones were still waiting to be freed made the validation layer reject the new ones.)
    if (want > 0 && !site_retired_.empty()) return;
    if (want > 0) {
        const double now = glfwGetTime();
        if (now < site_retry_at_) return;
        if (!load_site(want)) {
            // Not on this machine (yet): ask for it - a server client fetches it into the cache
            // folder (engine/site_store.h) - and look again in a while. Until then: empty land.
            site_retry_at_ = now + 3.0;
            if (site_store_.request) site_store_.request(globe::site(want).id);
            if (site_missing_logged_ != want) {
                log::warnf("[site] %s: no description (%s/%s.site) - %s", globe::site(want).name.c_str(), site_store_.cache_dir.c_str(),
                           globe::site(want).id.c_str(), site_store_.request ? "requested" : "nothing to fetch it with");
                site_missing_logged_ = want;
            }
        }
    }
}

// What stands at site k, into the slot. All of it on the level top (y = water + rise), laid
// out round the centre: x east, z south.
bool VulkanEngine::load_site(int k) {
    const globe::Site& S = globe::site(k);
    SiteData d;
    if (!site_store_.load(S.id, d)) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const float H = rt_.water_level + S.rise;
    const glm::vec3 c(float(S.at.x), H, float(S.at.y));
    auto at = [&](float dx, float dz) { return c + glm::vec3(dx, 0.0f, dz); };
    auto ground = [H](float, float) { return H; };
    game::Level lv;
    std::vector<glm::vec3> lamps;
    site_shelters_.clear();
    int towers = 0, houses = 0, chapels = 0, graves = 0;
    for (const SiteObject& o : d.objects) {
        if (o.kind == "tower") {
            game::append_outpost_tower(lv, at(o.x, o.z), glm::clamp(int(o.a + 0.5f), 0, 3), o.seed ? o.seed : 0x5EA7000u + uint32_t(towers) * 977u);
            ++towers;
        } else if (o.kind == "house") {
            place_house(lv, at(o.x, o.z) + glm::vec3(0.0f, -0.05f, 0.0f));
            ++houses;
        } else if (o.kind == "chapel") {
            if (game::append_church(lv, at(o.x, o.z))) { graves += game::append_graveyard(lv, at(o.x, o.z), ground); ++chapels; }
        } else if (o.kind == "lamp") {
            glm::vec3 arm(o.a, 0.0f, o.b);
            arm = glm::dot(arm, arm) > 1e-6f ? glm::normalize(arm) : glm::vec3(1.0f, 0.0f, 0.0f);
            lamps.push_back(game::append_lamp_post(lv, at(o.x, o.z), arm, 0x5EA7A00u + uint32_t(lamps.size()) * 131u));
        }
    }
    // The street. One line, as the shaders know one road at a time (scene.road_pts).
    site_road_ = Road{};
    if (d.road.size() >= 2) {
        Road r;
        for (const glm::vec2& p : d.road) r.pts.push_back(at(p.x, p.y) + glm::vec3(0.0f, 0.02f, 0.0f));
        const float m = r.half_width + r.shoulder + r.blend + 2.0f;
        r.bb_min = glm::vec2(1e9f); r.bb_max = glm::vec2(-1e9f);
        for (const auto& p : r.pts) {
            r.bb_min = glm::min(r.bb_min, glm::vec2(p.x, p.z) - m);
            r.bb_max = glm::max(r.bb_max, glm::vec2(p.x, p.z) + m);
        }
        site_road_ = std::move(r);
    }
    // Onto the GPU: mesh parts and their ray-tracing structures, as home's are made.
    size_t tris = 0;
    site_meshes_.clear();
    for (const auto& lm : lv.meshes) {
        if (lm.verts.empty() || lm.idx.empty()) continue;
        site_meshes_.push_back(make_mesh_part(lm));
        tris += lm.idx.size() / 3;
    }
    build_mesh_part_blas(site_meshes_);
    // Colliders: the player's (site_cols_, taken with the rest near the player) and the
    // physics world's (bullets, bodies).
    site_cols_ = lv.aabbs;
    site_phys_xz_.clear();
    if (physics_ && !site_cols_.empty()) {
        std::vector<PhysicsWorld::StaticBox> vb;
        for (const auto& a : site_cols_) {
            vb.push_back({ (a.min + a.max) * 0.5f, (a.max - a.min) * 0.5f });
            site_phys_xz_.push_back(glm::vec2(vb.back().center.x, vb.back().center.z));
        }
        physics_->add_static_box_compounds(vb.data(), vb.size());
    }
    // Lanterns: at the end of the list of them, so that they can be taken off again.
    site_lamp0_ = lamp_lights_.size();
    lamp_lights_.insert(lamp_lights_.end(), lamps.begin(), lamps.end());
    site_center_ = c;
    site_loaded_ = k;
    site_missing_logged_ = -1;
    ++site_gen_;
    near_static_src_[0] = SIZE_MAX;             // the player's colliders: taken again
    wv_nf_invalidate();                         // the near ground's bake: the street and floors cut out of the grass
    log::infof("[site] %s loaded in %.0f ms: %d towers, %d houses, %d chapels, %d gravestones, %zu lanterns, %zu road points (%zu mesh parts, %zu tris, %zu colliders)",
               S.name.c_str(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               towers, houses, chapels, graves, lamps.size(), site_road_.pts.size(), site_meshes_.size(), tris, site_cols_.size());
    return true;
}

void VulkanEngine::unload_site() {
    if (site_loaded_ <= 0) return;
    log::infof("[site] %s unloaded", globe::site(site_loaded_).name.c_str());
    if (!site_meshes_.empty()) site_retired_.emplace_back(12, std::move(site_meshes_));
    site_meshes_.clear();
    if (physics_ && !site_phys_xz_.empty()) physics_->remove_compound_boxes_at(site_phys_xz_.data(), site_phys_xz_.size(), 0.02f);
    site_phys_xz_.clear();
    site_cols_.clear();
    site_shelters_.clear();
    if (site_lamp0_ <= lamp_lights_.size()) lamp_lights_.resize(site_lamp0_);
    lamp_map_slot_ = -1;
    site_road_ = Road{};
    site_loaded_ = -1;
    ++site_gen_;
    near_static_src_[0] = SIZE_MAX;
}

void VulkanEngine::destroy_sites() {
    for (auto& r : site_retired_) destroy_mesh_parts(r.second);
    site_retired_.clear();
    destroy_mesh_parts(site_meshes_);
}

// The slot's content exists where the engine is now: in the chart of the site it holds.
bool VulkanEngine::site_here() const { return site_loaded_ > 0 && globe::current().site == site_loaded_; }

// The menu's "travel to": the player is set down at the site, in its chart. (Its content is
// loaded by update_sites on the next frame.)
void VulkanEngine::travel_to_site(int k) {
    if (k < 0 || k >= globe::site_count()) return;
    const globe::Site& S = globe::site(k);
    GlobeChartRef ref; ref.c = S.chart;
    globe::set_current(ref.c);
    if (k == 0) {
        reset_player();
    } else {
        // 120 m south of the centre, facing it
        player_.position = glm::vec3(float(S.at.x), rt_.water_level + S.rise + 1.0f, float(S.at.y) + 120.0f);
        player_.yaw = 0.0f; player_.pitch = 0.0f;
    }
    player_.velocity = glm::vec3(0.0f);
    prev_player_position_ = player_.position;
    fly_mode_ = false;
    globe_moved_player_ = true;
    wv_reset();
    near_static_src_[0] = SIZE_MAX;             // the player's colliders: taken again
    rocket_.cam_yaw = player_.yaw;
    site_retry_at_ = 0.0;
    log::infof("[site] travelled to %s", S.name.c_str());
}

// Home and the sites nearest to the player, nearest first: for the travel menu and the labels
// (with thousands of sites neither can show them all).
std::vector<int> VulkanEngine::sites_by_distance(int max_n) const {
    const globe::Chart& c = globe::current();
    const glm::dvec3 G = globe::to_planet(c, double(player_.position.x), double(player_.position.z));
    std::vector<std::pair<double, int>> order;
    for (int k = 1; k < globe::site_count(); ++k) order.push_back({ -glm::dot(G, globe::site(k).G), k });
    const size_t n = std::min(order.size(), size_t(std::max(max_n, 0)));
    std::partial_sort(order.begin(), order.begin() + ptrdiff_t(n), order.end());
    std::vector<int> out{ 0 };
    for (size_t i = 0; i < n; ++i) out.push_back(order[i].second);
    return out;
}

// The sites' names on the screen when seen from high up (above 1.5 km, and from space): a
// small ring where the place is, its name and how far it is. From orbit a site is a few
// pixels across - this is what tells which is which. (Menu: Game > Travel.)
void VulkanEngine::site_labels_hud() {
    if (!rt_.site_labels || !imgui_initialized_ || state_ != State::Playing) return;
    const globe::Chart& c = globe::current();
    const glm::vec3 eye = player_.eye_position();
    const float ground = sample_terrain_height(eye.x, eye.z);
    if (eye.y - std::max(ground, rt_.water_level) < 1500.0f) return;
    const glm::dvec3 Gc = globe::to_planet(c, double(eye.x), double(eye.z));
    const glm::dmat3 fc = globe::frame_at(c, double(eye.x), double(eye.z));          // east, up, south at the camera
    const double rc = globe::kR + double(eye.y);
    const glm::dvec3 Pc = Gc * rc;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const float k = disp.y / 1080.0f;
    for (int i : sites_by_distance(10)) {
        const globe::Site& s = globe::site(i);
        if (glm::dot(Gc, s.G) < globe::kR / rc - 0.002) continue;                     // behind the planet's edge
        const glm::dvec3 d = s.G * (globe::kR + 40.0) - Pc;
        const double dist = glm::length(d);
        if (dist < 4000.0) continue;
        const glm::vec3 dir(float(glm::dot(d, fc[0]) / dist), float(glm::dot(d, fc[1]) / dist), float(glm::dot(d, fc[2]) / dist));
        const glm::vec4 cl = current_frame_view_.vp * glm::vec4(eye + dir * 1000.0f, 1.0f);
        if (cl.w < 0.3f) continue;
        const ImVec2 sp((cl.x / cl.w * 0.5f + 0.5f) * disp.x, (cl.y / cl.w * 0.5f + 0.5f) * disp.y);
        if (sp.x < 0.0f || sp.y < 0.0f || sp.x > disp.x || sp.y > disp.y) continue;
        char tag[128];
        const double ground_km = globe::ground_distance(Gc, s.G) * 0.001;
        std::snprintf(tag, sizeof(tag), ground_km < 100.0 ? "%s  %.1f km" : "%s  %.0f km", s.name.c_str(), ground_km);
        const ImU32 col = IM_COL32(255, 235, 150, 235);
        dl->AddCircle(sp, 9.0f * k, IM_COL32(0, 0, 0, 160), 24, 3.0f * k);
        dl->AddCircle(sp, 9.0f * k, col, 24, 1.5f * k);
        dl->AddLine(ImVec2(sp.x, sp.y - 9.0f * k), ImVec2(sp.x, sp.y - 26.0f * k), col, 1.5f * k);
        const ImVec2 ts = ImGui::CalcTextSize(tag);
        const ImVec2 tp(sp.x - ts.x * 0.5f, sp.y - 28.0f * k - ts.y);
        dl->AddText(ImVec2(tp.x + 1.0f, tp.y + 1.0f), IM_COL32(0, 0, 0, 210), tag);
        dl->AddText(tp, col, tag);
    }
}

// For the scene buffer (scene_ubo.glsl site_ground / site_shape): the nearest site, if it is
// within reach of what can be seen - its centre in the chart the engine is in (xy; z: 1 =
// there is one; w: its number) and the shape of its ground (flat, slope, rise, clearing).
glm::vec4 VulkanEngine::site_uniform(glm::vec4* shape) const {
    if (shape) *shape = glm::vec4(300.0f, 1500.0f, 14.0f, 330.0f);
    const globe::Chart& c = globe::current();
    if (c.home || site_near_ <= 0 || site_near_dist_ > 150000.0) return glm::vec4(0.0f);
    const globe::Site& s = globe::site(site_near_);
    if (shape) *shape = glm::vec4(s.flat, s.slope, s.rise, s.clearing);
    const glm::dvec2 at = c.site == site_near_ ? s.at : globe::to_chart(c, s.G);
    return glm::vec4(float(at.x), float(at.y), 1.0f, float(site_near_));
}

} // namespace vibe
