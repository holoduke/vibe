#include "engine/rocket_sim.h"

#include <algorithm>
#include <cmath>

namespace qlike {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kScaleH = 7200.0;                 // m, the atmosphere's scale height
constexpr double kArea = 63.6;                     // m^2 (9 m across)
constexpr double kMdotFull = RocketSim::kThrustSl / (RocketSim::kG0 * RocketSim::kIspSl);
constexpr double kLandEngines = 13.0 / 33.0;       // the share of the engines that relight for the landing

// Turns the unit vector (ux, uy) toward (cx, cy) by at most `max_rad`.
void turn_toward(double& ux, double& uy, double cx, double cy, double max_rad) {
    const double cl = std::hypot(cx, cy);
    if (cl < 1e-9) return;
    cx /= cl; cy /= cl;
    const double ang = std::atan2(ux * cy - uy * cx, ux * cx + uy * cy);      // signed angle u -> c
    const double a = std::clamp(ang, -max_rad, max_rad);
    const double c = std::cos(a), s = std::sin(a);
    const double nx = ux * c - uy * s, ny = ux * s + uy * c;
    const double l = std::hypot(nx, ny);
    ux = nx / l; uy = ny / l;
}
}  // namespace

void RocketSim::reset(double pad_radius) {
    *this = RocketSim{};
    rs = pad_radius;
    x = 0.0; y = rs;
}

void RocketSim::liftoff() {
    if (phase != Pad && phase != Landed) return;
    const double keep = rs;
    reset(keep);
    phase = Ascent;
    throttle = 1.0;
}

double RocketSim::angle() const { return std::atan2(x, y); }
double RocketSim::downrange() const { return angle() * kPlanetR; }

void RocketSim::step() {
    if (phase == Pad || phase == Landed || phase == Crashed) { throttle = 0.0; gload = 1.0; return; }
    const double r = std::hypot(x, y);
    const double upx = x / r, upy = y / r;               // local up
    const double tgx = upy, tgy = -upx;                  // local horizontal, downrange
    alt = r - rs;
    vr = vx * upx + vy * upy;
    vt = vx * tgx + vy * tgy;
    speed = std::hypot(vx, vy);
    const double g = kMu / (r * r);
    const double gx = -g * upx, gy = -g * upy;
    const double h = std::max(alt, 0.0);
    const double pr = h < 150.0e3 ? std::exp(-h / kScaleH) : 0.0;        // pressure / density ratio
    const double rho = 1.225 * pr;
    const double isp = kIspVac - (kIspVac - kIspSl) * pr;
    const double a_full = kMdotFull * kG0 * isp / mass;                  // all engines, full throttle
    // drag (nose first on the way up, engines first on the way down)
    const double cda = (phase == Ascent ? 0.38 : 1.15) * kArea;
    const double ad = 0.5 * rho * speed * cda / mass;                    // a_drag = -ad * v
    const double adx = -ad * vx, ady = -ad * vy;

    // ---- guidance: the commanded axis and throttle ----
    double cx = ux, cy = uy, thr = 0.0, rate = 2.0;                      // rate: deg / s
    switch (phase) {
    case Ascent: {
        thr = std::min(1.0, 3.5 * kG0 / a_full);
        double s;                                                        // sine of the elevation above the horizon
        if (alt < 300.0) s = 1.0;
        else if (alt < 45.0e3) s = std::cos(0.5 * kPi * 0.78 * std::pow((alt - 300.0) / (45.0e3 - 300.0), 0.62));
        else {
            // Closed loop: the thrust along the horizon builds orbital speed, which takes a known
            // time; the thrust's vertical part is what brings the rocket to the orbit's height,
            // level, in exactly that time (the same zero-effort-miss law the landing uses), on top
            // of carrying the weight that speed does not yet carry.
            const double tgo = std::max((std::sqrt(kMu / r) - vt) / (a_full * thr * 0.97), 5.0);
            const double dh = kOrbitAlt - alt;
            const double a_r = 6.0 * (dh - vr * tgo) / (tgo * tgo) + 2.0 * vr / tgo + (g - vt * vt / r);
            s = std::clamp(a_r / (a_full * thr), -0.5, 0.8);
        }
        const double c = std::sqrt(std::max(0.0, 1.0 - s * s));
        cx = upx * s + tgx * c; cy = upy * s + tgy * c;
        rate = alt < 45.0e3 ? 3.0 : 1.2;
        if (vt * vt >= kMu / r && alt > 100.0e3) {                       // orbital speed: engines off
            thr = 0.0;
            phase = Coast;
            plan_return();
        }
        break;
    }
    case Coast:
        cx = -vx; cy = -vy;                                              // turn round: engines forward
        if (swept >= burn_at) { phase = Deorbit; burn_dv = 200.0; }
        break;
    case Deorbit: {
        cx = -vx; cy = -vy;
        thr = std::min(1.0, 6.0 / a_full);
        burn_dv -= a_full * thr * kDt;
        if (burn_dv <= 0.0) { thr = 0.0; phase = Entry; }
        break;
    }
    case Entry:
        cx = -vx; cy = -vy;
        rate = 4.0;
        // Relight when what is left of the fall is what the engines need to stop the sink rate
        // at a gentle 12 m/s^2 (the burn also has to kill the speed over the ground).
        if (alt < 15.0e3 && vr < 0.0 && alt <= 1.15 * vr * vr / (2.0 * 12.0) + 100.0) phase = Landing;
        break;
    case Landing: {
        rate = 14.0;
        const double a_max = a_full * kLandEngines, a_min = a_full * (1.0 / 33.0) * 0.4;
        double ax, ay;                                                   // wanted acceleration besides gravity
        // the pad, along the horizon (m; the guidance works in the local up / along frame)
        const double ex = std::clamp((0.0 - x) * tgx + (rs - y) * tgy, -4000.0, 4000.0);
        if (alt > 30.0) {
            const double tgo = std::max(2.0 * alt / std::max(-vr, 2.0), 1.0);
            const double zm_u = -alt - vr * tgo + 0.5 * g * tgo * tgo, zv_u = -vr + g * tgo;
            const double a_u = 6.0 * zm_u / (tgo * tgo) - 2.0 * zv_u / tgo;
            // (the planner's pass only stops the drift over the ground - where that ends is the
            // point the retrograde burn is timed by)
            const double a_h = guide_lateral ? 6.0 * (ex - vt * tgo) / (tgo * tgo) + 2.0 * vt / tgo : -2.0 * vt / tgo;
            ax = upx * a_u + tgx * a_h; ay = upy * a_u + tgy * a_h;
        } else {
            // The last metres: a steady sink rate, held over the mount.
            const double a_r = g + (-std::max(1.0, alt * 0.30) - vr) * 1.8;
            const double a_h = (guide_lateral ? 0.8 * ex : 0.0) - 1.6 * vt;
            ax = upx * a_r + tgx * a_h; ay = upy * a_r + tgy * a_h;
        }
        // keep the thrust near the vertical
        double au = ax * upx + ay * upy, ah = ax * tgx + ay * tgy;
        au = std::max(au, a_min * 0.5);
        const double lim = std::tan((alt < 400.0 ? 22.0 : 45.0) * kPi / 180.0) * au;
        ah = std::clamp(ah, -lim, lim);
        ax = upx * au + tgx * ah; ay = upy * au + tgy * ah;
        const double mag = std::clamp(std::hypot(ax, ay), a_min, a_max);
        thr = mag / a_full;
        cx = ax; cy = ay;
        break;
    }
    default: break;
    }
    turn_toward(ux, uy, cx, cy, rate * kPi / 180.0 * kDt);

    // ---- dynamics ----
    double mdot = thr * kMdotFull;
    if (mass - mdot * kDt < kDry) { mdot = 0.0; thr = 0.0; }             // tanks dry
    throttle = thr;
    const double at = thr * a_full;
    const double ax = gx + adx + at * ux, ay = gy + ady + at * uy;
    vx += ax * kDt; vy += ay * kDt;
    x += vx * kDt; y += vy * kDt;
    mass -= mdot * kDt;
    t += kDt;
    const double ang = std::atan2(x, y);
    double da = ang - prev_ang_;
    if (da > kPi) da -= 2.0 * kPi; else if (da < -kPi) da += 2.0 * kPi;
    swept += da; prev_ang_ = ang;

    gload = std::hypot(adx + at * ux, ady + at * uy) / kG0;
    q = 0.5 * rho * speed * speed;
    mach = speed / 300.0;
    heat = std::min(std::sqrt(rho) * speed * speed * speed / 4.0e9, 1.5);
    const double r2 = std::hypot(x, y);
    alt = r2 - rs;
    if (alt <= 0.0 && phase != Ascent) {
        td_swept_ = swept;
        if (std::hypot(vx, vy) < 6.0) {
            phase = Landed;
            if (guide_lateral) { x = 0.0; y = rs; }
            vx = vy = 0.0; ux = 0.0; uy = 1.0; alt = 0.0; speed = 0.0; throttle = 0.0; gload = 1.0; q = 0.0; heat = 0.0;
        } else {
            phase = Crashed; throttle = 0.0;
        }
    } else if (alt < 0.0) {                                              // (still on the mount)
        x = 0.0; y = rs; vx = vy = 0.0;
    }
}

// The retrograde burn is timed so that the fall ends at the pad after exactly one revolution.
// The planet does not turn and the orbit is close to a circle, so starting the burn an angle
// later moves the touch-down by nearly the same angle: run a copy of the flight ahead (same
// step, same guidance, the landing burn flown straight down), see where it comes down, shift
// the burn by the miss, repeat. Three or four rounds bring the miss to centimetres; what is
// left the landing guidance steers out.
void RocketSim::plan_return() {
    burn_at = 2.0 * kPi - 0.30;
    for (int it = 0; it < 8; ++it) {
        RocketSim c = *this;
        c.guide_lateral = false;
        c.phase = Coast;
        for (int i = 0; i < 400000 && c.phase != Landed && c.phase != Crashed; ++i) c.step();
        if (c.phase != Landed) break;                                    // (no orbit to come back from)
        const double err = c.td_swept_ - 2.0 * kPi;
        burn_at -= err;
        if (std::abs(err) < 2.0e-7) break;
    }
}

}  // namespace qlike
