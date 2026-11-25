// Gusts: the wind's strength as fronts that travel down the wind, so a field of grass and a
// line of trees move in waves - a front arrives, bends what stands there and passes on, with
// lulls between - instead of everything swaying at once. (Ours: the reference game's shaders take
// one strength for the whole view from the CPU plus the wind field near the camera; what its
// CPU does with that strength over time is not in the shaders.)
//   xz: the plant's place (m), wd: the wind's direction (unit, world x and z), t: time (s).
// Returns the factor on the wind's strength: about 0.3 in a lull .. 1.5 in a front.
float wind_gust(vec2 xz, vec2 wd, float t) {
    float a = dot(xz, wd), c = dot(xz, vec2(-wd.y, wd.x));
    float u = a - t * 6.0;                                                  // the fronts run at 6 m/s
    float g = sin(u * 0.20 + 1.4 * sin(c * 0.043 + t * 0.09)) * 0.5 + 0.5;  // ~30 m apart, bent along their length
    g *= 0.55 + 0.45 * sin(u * 0.053 + c * 0.029 + 2.0);                    // in groups, with calmer stretches
    float f = sin(u * 0.85 + c * 0.33 + 2.5 * sin(c * 0.12 + t * 0.2)) * 0.5 + 0.5;   // the small ripples on them
    return 0.3 + 1.0 * g * g + 0.25 * f * g;
}
