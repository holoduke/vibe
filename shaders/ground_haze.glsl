// The haze of distance near the ground: one formula for everything that stands in the
// landscape. The raymarched ground (terrain_rm_main_gi.glsl) and the lit geometry on it -
// trees, buildings, rocks (deferred_lighting.frag) - both call ground_haze(), so a far tree
// fades into the same air as the hill it stands on. Needs scene_ubo.glsl before it.

// The haze is tuned for the 1500 m seen from the ground. Beyond that (only reached from
// the air, looking down through thinner air) it grows slowly. And only the part of the ray
// inside the hazy air near the ground (2.5 km) counts: from high up most of the way down
// is through clear air.
float ground_haze_dist(float t) {
    float tl = t * min(1.0, 2500.0 / max(scene.camera_pos.y, 1.0));
    return (tl > 1500.0) ? 1500.0 + (tl - 1500.0) * 0.004 : tl;
}

// col seen through t metres of air along rd. Blue fades about twice as fast as red; the
// air itself is grey by day (dark at night: crater_info.w) and warm toward the sun.
vec3 ground_haze(vec3 col, float t, vec3 rd, vec3 sun_dir) {
    vec3 extinction = exp(-ground_haze_dist(t) * 0.00025 * vec3(1.0, 1.3, 2.0));
    float sundot = clamp(dot(rd, sun_dir), 0.0, 1.0);
    float sn2 = sundot * sundot;
    float sn4 = sn2 * sn2;
    float sn8 = sn4 * sn4;
    vec3 fogCol = scene.crater_info.w * mix(vec3(0.55, 0.55, 0.58), vec3(1.0, 0.7, 0.3), 0.3 * sn8);
    return col * extinction + fogCol * (1.0 - extinction);
}
