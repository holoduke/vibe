add_executable(qlike_tests
    test_physics.cpp
    test_collision.cpp
    test_frustum.cpp
)

target_link_libraries(qlike_tests PRIVATE
    qlike_core
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(qlike_tests)

# Character animation (src/anim). Separate binary: it links ozz and the offline importers.
add_executable(qlike_anim_tests
    anim/test_skeleton.cpp
    anim/test_runtime.cpp
    anim/test_import.cpp
    anim/test_motion.cpp
    anim/test_animator.cpp
)
target_link_libraries(qlike_anim_tests PRIVATE
    qlike_anim_offline
    Catch2::Catch2WithMain
)
target_compile_definitions(qlike_anim_tests PRIVATE QLIKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
catch_discover_tests(qlike_anim_tests)
