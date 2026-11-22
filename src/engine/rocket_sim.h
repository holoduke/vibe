#pragma once
// The rocket's flight, from the pad to orbit and back (no engine dependencies).
//
// A point mass in the plane of its orbit, in double precision, planet-centred and
// non-rotating: inverse-square gravity, an exponential atmosphere (drag and the loss of
// thrust at sea level), a rocket engine that burns propellant, and guidance that flies
// it: a pitch-over into a gravity turn, a closed-loop climb to a circular orbit, one
// revolution, a retrograde burn timed so the fall ends at the pad, an engines-first entry
// and a powered landing (zero-effort-miss / zero-effort-velocity guidance) back on the mount.
//
// Coordinates: the planet's centre is the origin, the pad stands at (0, rs) and the
// rocket leaves toward +x. The step is fixed (kDt), so a copy of the state run ahead of
// time predicts the flight exactly - that is how the retrograde burn is timed.

namespace vibe {

struct RocketSim {
    static constexpr double kPlanetR = 6360000.0;        // m (the atmosphere model's planet)
    static constexpr double kMu      = 3.986004418e14;   // m^3 / s^2
    static constexpr double kG0      = 9.80665;
    static constexpr double kDt      = 0.05;             // s, fixed step
    static constexpr double kDry     = 170.0e3;          // kg
    static constexpr double kProp    = 3430.0e3;         // kg
    static constexpr double kIspSl   = 335.0, kIspVac = 372.0;
    static constexpr double kThrustSl = 1.5 * (kDry + kProp) * kG0;      // N at sea level, all engines
    static constexpr double kOrbitAlt = 180.0e3;         // m

    enum Phase { Pad, Ascent, Coast, Deorbit, Entry, Landing, Landed, Crashed };

    Phase  phase = Pad;
    double rs = kPlanetR;            // the pad's distance from the centre (rocket at rest)
    double x = 0.0, y = kPlanetR, vx = 0.0, vy = 0.0;
    double ux = 0.0, uy = 1.0;       // the rocket's axis (nose direction); thrust acts along it
    double mass = kDry + kProp;
    double t = 0.0;                  // s since lift-off
    double throttle = 0.0;           // 0..1 of all engines
    double swept = 0.0;              // central angle travelled since lift-off (rad, unwrapped)
    double burn_at = 0.0;            // swept angle at which the retrograde burn starts
    double burn_dv = 0.0;            // m/s still to shed in the retrograde burn
    bool   guide_lateral = true;     // landing burn steers to the pad (off in the planner's first pass)
    // read-outs (updated by step)
    double alt = 0.0, speed = 0.0, vr = 0.0, vt = 0.0, q = 0.0, gload = 0.0, mach = 0.0, heat = 0.0;

    void reset(double pad_radius);
    void liftoff();                  // Pad -> Ascent (engines are already at full thrust)
    void step();                     // one kDt
    double propellant() const { return (mass - kDry) / kProp; }      // 0..1 left
    double angle() const;            // central angle from the pad (rad, -pi..pi; + = downrange)
    double downrange() const;        // ground distance from the pad along the surface (m, signed)

private:
    void plan_return();              // times the retrograde burn (at engine cut-off in orbit)
    double prev_ang_ = 0.0;
    double td_swept_ = 0.0;          // swept angle at touch-down (the planner reads it)
};

}  // namespace vibe
