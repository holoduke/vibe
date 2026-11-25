// The haze of distance near the ground: one formula for everything that stands in the
// landscape. The raymarched ground (terrain_rm_main_gi.glsl) and the lit geometry on it -
// trees, buildings, rocks (deferred_lighting.frag) - both call ground_haze(), so a far tree
// fades into the same air as the hill it stands on. Needs scene_ubo.glsl before it, and a
// `sampler2D u_skyview` (the atmosphere's sky-view LUT) declared by the file that includes it.

// The haze is tuned for the 1500 m seen from the ground. Beyond that (only reached from
// the air, looking down through thinner air) it grows slowly. And only the part of the ray
// inside the hazy air near the ground (2.5 km) counts: from high up most of the way down
// is through clear air.
float ground_haze_dist(float t) {
    float tl = t * min(1.0, 2500.0 / max(scene.camera_pos.y, 1.0));
    return (tl > 1500.0) ? 1500.0 + (tl - 1500.0) * 0.004 : tl;
}

// The colour of fully hazed air along rd: the sky that is seen just over the horizon that way.
// What is far enough away has this colour - so it has to be the colour of what lies behind
// it, or a far forest ends in a hard outline against the sky. It was a fixed grey (dark at
// night by a factor): by day a grey-blue band of trees in front of a brighter, bluer sky; at
// night a band LIGHTER than the sky, since the night sky is darker than the grey times that
// factor. The sky itself is right at every hour: bright and pale by day, warm toward a low
// sun, dark at night.
// (a little over the horizon and no higher than a far ridge reaches: the row of the LUT at
//  the horizon itself holds the ground's own bounce, a bright white)
vec3 ground_haze_air(vec3 rd, vec3 sun_dir) {
    if (scene.atmo_params.x > 0.5) {
// Looking DOWN (from a hill, from the air) the air between the eye and the ground is not
        // the bright band over the horizon - that band is bright because the line of sight
        // runs through hundreds of kilometres of air. A line of sight that goes down at an
        // angle crosses about as much air as one that goes up at that angle: it takes the
        // sky's colour from there (mirrored). With the horizon's colour for every downward ray
        // the land seen from 600 m up was washed out to white (2026-11-24).
        float e_rd = asin(clamp(rd.y, -1.0, 1.0));
        float elev = (e_rd >= 0.0 ? clamp(e_rd, 0.0, 0.30) : clamp(-e_rd, 0.0, 1.30)) + 0.025;
        float az = 0.0;
        if (dot(rd.xz, rd.xz) > 1e-8 && dot(sun_dir.xz, sun_dir.xz) > 1e-8)
            az = acos(clamp(dot(normalize(rd.xz), normalize(sun_dir.xz)), -1.0, 1.0));
        // Not the glow round the sun itself. That glow is light scattered in the air in FRONT
        // of the sun - and a mountain that stands between the eye and a low sun has the sun
        // behind it: the air on this side of it lies in its shadow. With the sky's colour taken
        // straight toward the sun, an evening mountain was washed orange as if the sun shone
        // through it. So the haze on things takes the sky no nearer than 40 degrees round from
        // the sun: the glow stays in the sky, above and beside the mountain.
        az = max(az, 0.70);
        // (the LUT's own mapping, atmosphere.glsl skyview_uv: x = angle from the sun round the
        //  horizon, y = elevation with the horizon stretched)
        float v = elev / 1.5707963;
        vec2 uv = vec2(az / 3.14159265, 0.5 + 0.5 * sqrt(v));
vec3 air = textureLod(u_skyview, uv, 0.0).rgb * scene.sun_color.rgb * (scene.sun_color.a * scene.atmo_params.y);
        // ... and that sky is the light of the whole depth of the atmosphere, while the air
        // under the eye is only what lies below it: the share of the air's light that a
        // downward view gets grows with the camera's height (about a fifth at 600 m, two
        // fifths at 1500 m, all of it from high up). Without it the land from 1500 m was a
        // sheet of the sky's blue.
        float below = 1.0 - exp(-max(scene.camera_pos.y - scene.water_params.y, 0.0) / 3000.0);
        return air * mix(1.0, below, smoothstep(0.0, 0.25, -rd.y));
    }
    // No sky model (the plain sky): the grey it was, warm toward the sun.
    float sundot = clamp(dot(rd, sun_dir), 0.0, 1.0);
    float sn2 = sundot * sundot;
    float sn4 = sn2 * sn2;
    float sn8 = sn4 * sn4;
    return scene.crater_info.w * mix(vec3(0.55, 0.55, 0.58), vec3(1.0, 0.7, 0.3), 0.3 * sn8);
}

// col seen through t metres of air along rd. Blue fades about twice as fast as red.
vec3 ground_haze(vec3 col, float t, vec3 rd, vec3 sun_dir) {
    vec3 extinction = exp(-ground_haze_dist(t) * 0.00025 * vec3(1.0, 1.3, 2.0));
    return col * extinction + ground_haze_air(rd, sun_dir) * (1.0 - extinction);
}
