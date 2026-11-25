add_executable(vibe_tests
    test_physics.cpp
    test_collision.cpp
    test_frustum.cpp
    test_site_store.cpp
    test_globe.cpp
    test_terrain_keep.cpp
    test_npc.cpp
)

target_link_libraries(vibe_tests PRIVATE
    vibe_core
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(vibe_tests)

# Character animation (src/anim). Separate binary: it links ozz and the offline importers.
add_executable(vibe_anim_tests
    anim/test_skeleton.cpp
    anim/test_runtime.cpp
    anim/test_import.cpp
    anim/test_motion.cpp
    anim/test_animator.cpp
    anim/test_net.cpp
)
target_link_libraries(vibe_anim_tests PRIVATE
    vibe_anim_offline
    Catch2::Catch2WithMain
)
target_compile_definitions(vibe_anim_tests PRIVATE VIBE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
catch_discover_tests(vibe_anim_tests)
