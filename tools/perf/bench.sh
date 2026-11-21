#!/bin/bash
# GPU benchmark: the same three views, every time, at the played resolution.
#
#   tools/perf/bench.sh [-e build/vibe.exe] [-s 3440x1400] [-n label] [key=value ...]
#
# Runs the game three times (close ground / far view / forest), 400 frames each, with a COPY of
# vibe_settings.cfg in which the time of day, weather and season are pinned and the camera is
# held in place; `key=value` arguments override settings in that copy (the user's file is not
# touched). Prints one line per view with the cost of the main passes (ms, from the log's
# `gpu-passes` line) and the sum, then the three sums added up.
#
# Before trusting a number:
#   * nothing else may be using the GPU (another vibe.exe, a game): the script warns;
#   * after a terrain shader change the first run compiles it for a minute or two - the script
#     makes a throw-away run first and waits until the log says the full shader is in use.
#
# A/B: run it twice with different overrides, e.g.
#   tools/perf/bench.sh -n off terrain_rm_prepass=0
#   tools/perf/bench.sh -n on  terrain_rm_prepass=1
set -u
cd "$(dirname "$0")/../.." || exit 1
EXE=build/vibe.exe; SIZE=3440x1400; LABEL=run
while getopts "e:s:n:" o; do case $o in e) EXE=$OPTARG;; s) SIZE=$OPTARG;; n) LABEL=$OPTARG;; *) exit 2;; esac; done
shift $((OPTIND - 1))
OUT=build/bench; mkdir -p "$OUT"
OV=("$@")

if tasklist 2>/dev/null | grep -qi "vibe"; then echo "WARNING: a vibe process is running - the numbers will be too high and unsteady" >&2; fi

make_cfg() {   # $1 = file, rest = key=value
    local f=$1; shift
    cp vibe_settings.cfg "$f"
    for kv in time_of_day=14 weather_preset=0 season=1 "$@"; do
        local k="${kv%%=*}" v="${kv#*=}"
        grep -v "^$k = " "$f" > "$f.tmp"; echo "$k = $v" >> "$f.tmp"; mv "$f.tmp" "$f"
    done
}
run() {        # $1 = name, rest = key=value
    local name=$1; shift
    make_cfg "$OUT/$name.cfg" "$@"
    QLIKE_CFG="$(cygpath -w "$OUT/$name.cfg")" QLIKE_TEST_HOLD=1 QLIKE_TEST_GROUND=1 QLIKE_NO_NET=1 QLIKE_TEST_SIZE=$SIZE \
        "$EXE" --frames ${FRAMES:-400} --log "$(cygpath -w "$OUT/$name.log")" > /dev/null 2>&1
}
pass() { echo "$1" | grep -o "$2=[0-9.]*" | cut -d= -f2 | awk '{s+=$1} END {printf "%.2f", s+0}'; }

# Throw-away run: lets the driver finish compiling the terrain shader.
for try in 1 2 3; do
    FRAMES=200 run warm "${OV[@]}"
    grep -q "full shader is compiled" "$OUT/warm.log" && [ "$(grep -n 'full shader is compiled' "$OUT/warm.log" | head -1 | cut -d: -f1)" -lt "$(grep -n 'gpu-passes' "$OUT/warm.log" | head -1 | cut -d: -f1)" ] && break
done

total=0
printf "%-8s %-7s | %7s %7s %7s %7s %7s %7s %7s %7s | %7s\n" "$LABEL" view terrain march_lo deferred veg grass scatter upscale other sum
for view in "close player_pos_x=26.6 player_pos_z=123.8 player_yaw=-71.6895 player_pitch=-0.32" \
            "far player_pos_x=26.6 player_pos_z=123.8 player_yaw=0.35 player_pitch=-0.05" \
            "forest player_pos_x=139.4 player_pos_z=-3.2 player_yaw=0 player_pitch=-0.05"; do
    set -- $view
    name=$1; shift
    run "$LABEL-$name" "$@" "${OV[@]}"
    L=$(grep "gpu-passes" "$OUT/$LABEL-$name.log" | tail -1)
    t=$(pass "$L" w_terrain_raymarch); lo=$(pass "$L" terrain_march_lo); d=$(pass "$L" deferred_lighting)
    v=$(echo "$L" | grep -o "[dw]_veg=[0-9.]*" | cut -d= -f2 | awk '{s+=$1} END {printf "%.2f", s+0}')
    g=$(pass "$L" w_grass_rm); sc=$(pass "$L" w_scatter); u=$(pass "$L" fsr_taau); s=$(echo "$L" | grep -o "sum=[0-9.]*" | cut -d= -f2)
    o=$(awk -v s="$s" -v a="$t" -v b="$lo" -v c="$d" -v e="$v" -v f="$g" -v h="$sc" -v i="$u" 'BEGIN {printf "%.2f", s-a-b-c-e-f-h-i}')
    printf "%-8s %-7s | %7s %7s %7s %7s %7s %7s %7s %7s | %7s\n" "" "$name" "$t" "$lo" "$d" "$v" "$g" "$sc" "$u" "$o" "$s"
    total=$(awk -v a="$total" -v b="$s" 'BEGIN {printf "%.2f", a+b}')
done
echo "$LABEL total (3 views): $total ms"
