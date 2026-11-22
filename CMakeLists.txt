cmake_minimum_required(VERSION 3.24)
project(vibe CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:preprocessor)
    # /Zi + /DEBUG so the StackWalk-based crash dumper can resolve symbols
    # to file:line in Release. Without these we still get module+offset, but
    # not function names — useless for "where did this throw come from".
    add_compile_options(/Zi)
    add_link_options(/DEBUG)

    # Release-only: aggressive optimisation. /O2 is the default but we
    # explicitly opt into the additional flags below.
    #   /Ob3       max inlining (above /Ob2 default)
    #   /Oi        intrinsic functions (memcpy, sin, etc → inline asm)
    #   /Ot        favour speed over size
    #   /GL        whole-program optimisation (paired with /LTCG link)
    #   /Gy        function-level linking (lets /OPT:REF strip unused)
    #   /Gw        same for global data
    #   /arch:AVX2 enable AVX2 instructions (every x64 CPU since ~2013)
    #   /Qpar      auto-parallelise simple loops
    #
    # /fp:fast removed — bisected to be the cause of a sky-coloured
    # cyan-wash flicker on the user's GPU. /fp:fast lets MSVC reorder
    # FP ops + treat denormals as zero + assume operands are not NaN;
    # somewhere in the per-frame uniform / scene UBO compute, that
    # assumption produces a wrong value that scales the lit term
    # screen-wide. Keep IEEE-conformant FP for now; the perf delta vs
    # /fp:precise is marginal in this codebase.
    add_compile_options(
        $<$<CONFIG:Release>:/Ob3>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/Qpar>
    )
    # /DEBUG turns OPT:REF/ICF off by default; re-enable both so unused
    # functions + identical COMDATs get folded out.
    add_link_options(
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # -ffast-math symmetrically dropped (matches MSVC /fp:fast removal).
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-flto>
    )
    # AVX2 baseline only on x86_64 — Apple Silicon (arm64) doesn't have it.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        add_compile_options($<$<CONFIG:Release>:-march=x86-64-v3>)
    endif()
    add_link_options($<$<CONFIG:Release>:-flto>)
endif()

find_package(Vulkan REQUIRED COMPONENTS glslc)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE)

FetchContent_Declare(vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG v1.3.295
    GIT_SHALLOW TRUE)

FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE)

FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)

# Jolt Physics — used for non-player rigid bodies (the player is custom Quake-style).
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(CROSS_PLATFORM_DETERMINISTIC OFF CACHE BOOL "" FORCE)
set(DOUBLE_PRECISION OFF CACHE BOOL "" FORCE)
# Skip Jolt's install/export plumbing — it tries to write JoltConfig-$config.cmake
# with a literal $config when its export() runs from inside FetchContent, which
# trips a "$<CONFIG:$Config>" generator-expression error during Generate.
set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG v5.2.0
    SOURCE_SUBDIR Build
    GIT_SHALLOW TRUE)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
    GIT_SHALLOW TRUE)

# tinygltf — header-only glTF 2.0 loader. Used to load the optional viewmodel
# asset (drop a .glb at assets/gun.glb to activate). Opted out of its loader
# for stb_image_write / json header generation; we only need the parser.
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_VALIDATOR_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_BUILDER_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.3
    GIT_SHALLOW TRUE)

# tinyexr — OpenEXR loader. Used to load the HDR equirect skybox (.exr) so
# the sun pixel keeps its real radiance and blooms naturally instead of
# being a clamped LDR disc + procedural overlay. tinyexr ships its own CMake
# target that bundles miniz — we link it directly rather than re-instantiate
# TINYEXR_IMPLEMENTATION ourselves (which would also need miniz on our include
# path).
set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG v1.0.8
    GIT_SHALLOW TRUE)

# miniaudio — single-header audio engine (MIT-0). Built-in 3D spatial
# sources with WAV/MP3/FLAC decoding; Vorbis comes via the bundled
# stb_vorbis. We pull just the source tree and wire a header-only
# INTERFACE target ourselves; the implementation goes in vma_impl.cpp's
# sibling (audio.cpp) so MINIAUDIO_IMPLEMENTATION lives in exactly one TU.
FetchContent_Declare(miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG 0.11.21
    GIT_SHALLOW TRUE)

# FastNoiseLite — single-header noise (MIT). Used by the procedural
# terrain generator (src/engine/terrain.cpp).
FetchContent_Declare(fastnoiselite
    GIT_REPOSITORY https://github.com/Auburn/FastNoiseLite.git
    GIT_TAG v1.1.1
    GIT_SHALLOW TRUE)

# ENet - reliable UDP for games (MIT). LAN multiplayer transport (src/engine/net).
FetchContent_Declare(enet
    GIT_REPOSITORY https://github.com/lsalzman/enet.git
    GIT_TAG v1.3.18
    GIT_SHALLOW TRUE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# meshoptimizer - index / vertex order for the GPU, simplification, meshlets (MIT).
# Used by src/engine/mesh_opt.h; the base for cluster LOD work.
FetchContent_Declare(meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG v1.0
    GIT_SHALLOW TRUE)

# ozz-animation - skeletons, compressed clips, sampling, blending, local-to-model, IK jobs (MIT).
# The character animation library (src/anim) is built on it; it is hidden behind
# anim::IAnimationRuntime. Only the runtime and the offline builders are built.
set(ozz_build_tools OFF CACHE BOOL "" FORCE)
set(ozz_build_fbx OFF CACHE BOOL "" FORCE)
set(ozz_build_gltf OFF CACHE BOOL "" FORCE)
set(ozz_build_data OFF CACHE BOOL "" FORCE)
set(ozz_build_samples OFF CACHE BOOL "" FORCE)
set(ozz_build_howtos OFF CACHE BOOL "" FORCE)
set(ozz_build_tests OFF CACHE BOOL "" FORCE)
set(ozz_build_simd_ref OFF CACHE BOOL "" FORCE)
set(ozz_build_postfix OFF CACHE BOOL "" FORCE)
set(ozz_build_msvc_rt_dll ON CACHE BOOL "" FORCE)
FetchContent_Declare(ozz
    GIT_REPOSITORY https://github.com/guillaumeblanc/ozz-animation.git
    GIT_TAG 0.17.0
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(glfw glm vk_bootstrap vma Catch2 JoltPhysics imgui tinygltf tinyexr miniaudio fastnoiselite enet meshoptimizer ozz)

# AMD FidelityFX-FSR2 — vendored as a git submodule under
# external/FidelityFX-FSR2 (MIT license; LICENSE.txt preserved). Use as-is
# via the SDK's public API; no port, no clean-room reimplementation.
#
# Phase 2 (build wiring):
#   - DX12 backend disabled; we only need Vulkan.
#   - FSR2's CMakeLists hard-FATAL_ERRORs unless CMAKE_GENERATOR_PLATFORM
#     or CMAKE_EXE_LINKER_FLAGS picks x64/Win32 — set the linker-flag
#     hint temporarily to satisfy that check under Ninja, then restore.
#   - EXCLUDE_FROM_ALL so we only build it when something links to it.
#   - Auto-shader-compilation calls FidelityFX_SC.exe (bundled at
#     external/FidelityFX-FSR2/tools/sc/), which then invokes the
#     glslangValidator next to it — both are in the submodule, no
#     external DXC dependency required for the Vulkan path.
if(EXISTS "${CMAKE_SOURCE_DIR}/external/FidelityFX-FSR2/src/ffx-fsr2-api/CMakeLists.txt")
    set(FFX_FSR2_API_DX12 OFF CACHE BOOL "" FORCE)
    set(FFX_FSR2_API_VK   ON  CACHE BOOL "" FORCE)
    set(FSR2_BUILD_AS_DLL OFF CACHE BOOL "" FORCE)

    # Satisfy FSR2's platform check (FATAL_ERROR on non-x64) — we use
    # Ninja so CMAKE_GENERATOR_PLATFORM is unset; spoof via linker flag.
    set(_qlike_orig_exe_linker_flags "${CMAKE_EXE_LINKER_FLAGS}")
    if(NOT CMAKE_GENERATOR_PLATFORM)
        set(CMAKE_EXE_LINKER_FLAGS "/machine:x64")
    endif()

    add_subdirectory(external/FidelityFX-FSR2/src/ffx-fsr2-api EXCLUDE_FROM_ALL)

    # Restore — the FSR2 CMake only reads CMAKE_EXE_LINKER_FLAGS at
    # configure-time inside its top-level if() block.
    set(CMAKE_EXE_LINKER_FLAGS "${_qlike_orig_exe_linker_flags}")
else()
    message(STATUS "FidelityFX-FSR2 submodule not present — run "
                   "`git submodule update --init --recursive` to enable.")
endif()

# AMD FidelityFX SDK (FSR3 + Frame Generation) — vendored at
# external/FidelityFX-SDK (MIT, LICENSE.txt preserved). The v1.1.4 SDK
# ships a stable C ABI behind a prebuilt DLL (amd_fidelityfx_vk.dll +
# amd_fidelityfx_vk.lib in PrebuiltSignedDLL/). We link the import lib,
# include the ffx-api headers, and copy the DLL alongside the exe at
# build time. No add_subdirectory needed — the sample-framework CMake
# in the SDK isn't designed for drop-in use.
#
# The DLL is the FIRST runtime dependency in the project; everything
# else is statically linked. This is the cost of FSR3's chosen ABI.
set(FFX_SDK_ROOT "${CMAKE_SOURCE_DIR}/external/FidelityFX-SDK")
if(EXISTS "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.lib")
    add_library(ffx_sdk_vk SHARED IMPORTED)
    set_target_properties(ffx_sdk_vk PROPERTIES
        IMPORTED_LOCATION "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.dll"
        IMPORTED_IMPLIB   "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFX_SDK_ROOT}/ffx-api/include")
    message(STATUS "FidelityFX SDK 1.1.4 prebuilt VK lib + headers found")
else()
    message(STATUS "FidelityFX SDK not present at external/FidelityFX-SDK — "
                   "FSR3 disabled. Run `git submodule update --init --recursive` "
                   "and check out v1.1.4 in the submodule.")
endif()

# tinyexr enables -Weverything -Werror; clang on Apple Silicon flags
# /usr/local/include via -Wpoison-system-directories. Disable warnings for it.
if(TARGET tinyexr)
    target_compile_options(tinyexr PRIVATE -w)
endif()
if(TARGET miniz)
    target_compile_options(miniz PRIVATE -w)
endif()

# Dear ImGui ships without CMakeLists; build our own target from its sources.
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# qlike_core: pure logic with no Vulkan/GLFW deps. Tests link this directly.
add_library(qlike_core STATIC
    src/game/physics.cpp
    src/game/collision.cpp
    src/game/level.cpp
    src/game/player.cpp
    src/engine/frustum.cpp
)
target_include_directories(qlike_core PUBLIC src)
target_link_libraries(qlike_core PUBLIC glm::glm)
target_compile_definitions(qlike_core PUBLIC
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_INLINE             # aggressive inline for hot paths
    # GLM_FORCE_INTRINSICS intentionally NOT defined — it routes vec
    # construction through SIMD intrinsics that MSVC can't evaluate at
    # constexpr (breaks `static constexpr glm::vec3` in player.h etc.).
    # The scalar path with /arch:AVX2 + /fp:fast gets auto-vectorised
    # by the compiler anyway.
)

# qlike_anim: character animation (skeleton + clip runtime on ozz, motion matching, layers, IK).
# No Vulkan, no engine headers; tests and the cook tool link it directly.
add_library(qlike_anim STATIC
    src/anim/types.cpp
    src/anim/skeleton_def.cpp
    src/anim/asset.cpp
    src/anim/ozz_runtime.cpp
    src/anim/motion_db.cpp
    src/anim/trajectory.cpp
    src/anim/motion_matcher.cpp
    src/anim/inertializer.cpp
    src/anim/airborne.cpp
    src/anim/character_animator.cpp
)
target_link_libraries(qlike_anim PUBLIC qlike_core PRIVATE ozz_animation ozz_base)

# Offline half: importers, retargeting, cooking (never linked into the game).
add_library(qlike_anim_offline STATIC
    src/anim/offline/bvh.cpp
    src/anim/offline/raw_clip.cpp
    src/anim/offline/retarget.cpp
    src/anim/offline/cook.cpp
    src/anim/offline/motion_build.cpp
)
target_link_libraries(qlike_anim_offline PUBLIC qlike_anim PRIVATE ozz_animation_offline ozz_animation ozz_base)

add_executable(vibe_animcook tools/anim/cook_main.cpp)
target_link_libraries(vibe_animcook PRIVATE qlike_anim_offline)

# What animation costs for 1 .. 1000 characters on this machine (docs/animation/performance.md).
add_executable(vibe_animbench tools/anim/bench_main.cpp)
target_link_libraries(vibe_animbench PRIVATE qlike_anim)

# qlike_engine: rendering, windowing, Vulkan. Depends on qlike_core.
add_library(qlike_engine STATIC
    src/engine/window.cpp
    src/engine/vk_initializers.cpp
    src/engine/vk_engine.cpp
    src/engine/vk_engine/frame.cpp
    src/engine/vk_engine/frame_upload.cpp
    src/engine/frame_pacer.cpp
    src/engine/vk_engine/rr_guides.cpp
    src/engine/vk_engine/loader_screen.cpp
    src/engine/vk_engine/pipelines_setup.cpp
    src/engine/vk_engine/helpers.cpp
    src/engine/vk_engine/settings.cpp
    src/engine/vk_engine/descriptors.cpp
    src/engine/vk_engine/rt.cpp
    src/engine/vk_engine/castle_mesh.cpp
    src/engine/vk_engine/fx.cpp
    src/engine/vk_engine/vegetation.cpp
    src/engine/vk_engine/veg_impostor.cpp
    src/engine/vk_engine/clutter_shadow.cpp
    src/engine/vk_engine/aa.cpp
    src/engine/vk_engine/rocket.cpp
    src/engine/vk_engine/post.cpp
    src/engine/vk_engine/atmosphere.cpp
    src/engine/vk_engine/radiance_cache.cpp
    src/engine/vk_engine/craters.cpp
    src/engine/vk_engine/water_fx.cpp
    src/engine/vk_engine/environment.cpp
    src/engine/vk_engine/sun_horizon.cpp
    src/engine/vk_engine/veg_zones.cpp
    src/engine/vk_engine/veg_stream.cpp
    src/engine/vk_engine/ssdm.cpp
    src/engine/vk_engine/flashlight.cpp
    src/engine/vk_engine/player_body.cpp
    src/engine/vk_engine/ragdoll.cpp
    src/engine/vk_engine/body_rt.cpp
    src/engine/net/net.cpp
    src/engine/cluster_lod.cpp
    src/engine/vk_engine/cluster_draw.cpp
    src/engine/vk_engine/plant_gpu.cpp
    src/engine/vk_engine/multiplayer.cpp
    src/engine/vk_engine/ui_gamehud.cpp
    src/engine/cpu_prof.cpp
    src/engine/jobs.cpp
    src/engine/vk_engine/dlss.cpp
    src/engine/vk_engine/bloom.cpp
    src/engine/vk_engine/taa.cpp
    src/engine/vk_engine/fsr2.cpp
    src/engine/vk_engine/fsr3.cpp
    src/engine/vk_engine/restir.cpp
    src/engine/vk_engine/sun_shadow.cpp
    src/engine/vk_engine/combat.cpp
    src/engine/vk_engine/screenshot.cpp
    src/engine/vk_engine/setup.cpp
    src/engine/vk_engine/svgf.cpp
    src/engine/vk_engine/voxel.cpp
    src/engine/vk_engine/world.cpp
    src/engine/vk_engine/world_render.cpp
    src/engine/vk_engine/world_textures.cpp
    src/engine/vk_engine/terrain_edit.cpp
    src/engine/vk_engine/terrain_shadow_bake.cpp
    src/engine/vk_engine/terrain_textures.cpp
    src/engine/vk_engine/props.cpp
    src/engine/vk_engine/deferred_lighting.cpp
    src/engine/voxel/voxel_world.cpp
    src/engine/vk_engine/ui_setup.cpp
    src/engine/vk_engine/ui_hud.cpp
    src/engine/vk_engine/ui_menu.cpp
    src/engine/vk_pipelines.cpp
    src/engine/mesh.cpp
    src/engine/gltf_loader.cpp
    src/engine/skybox.cpp
    src/engine/texture.cpp
    src/engine/physics_world.cpp
    src/engine/vma_impl.cpp
    src/engine/log.cpp
    src/engine/crash_handler.cpp
    src/engine/audio.cpp
    src/engine/terrain.cpp
    src/engine/terrain_clod.cpp
    src/engine/grass.cpp
    src/engine/scatter.cpp
    src/engine/road.cpp
    src/engine/eztree.cpp
    src/engine/world_props.cpp
    src/engine/aa_vehicle.cpp
    src/engine/rocket_sim.cpp
    src/engine/globe.cpp
)
target_include_directories(qlike_engine PUBLIC src)
target_include_directories(qlike_engine PRIVATE
    ${tinygltf_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${miniaudio_SOURCE_DIR}
    ${fastnoiselite_SOURCE_DIR}/Cpp)
target_link_libraries(qlike_engine PUBLIC
    qlike_core
    qlike_anim
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    Jolt
    imgui
    tinyexr
    enet
    meshoptimizer
    ws2_32
    winmm
    iphlpapi
)
target_include_directories(qlike_engine PRIVATE ${enet_SOURCE_DIR}/include)
target_include_directories(qlike_engine PRIVATE ${meshoptimizer_SOURCE_DIR}/demo)   # clusterlod.h

# FSR2 — link only if the submodule was processed above.
if(TARGET ffx_fsr2_api_x64)
    target_link_libraries(qlike_engine PUBLIC ffx_fsr2_api_x64)
    target_include_directories(qlike_engine PRIVATE
        ${CMAKE_SOURCE_DIR}/external/FidelityFX-FSR2/src/ffx-fsr2-api)
    target_compile_definitions(qlike_engine PUBLIC QLIKE_HAVE_FSR2=1)
endif()
if(TARGET ffx_fsr2_api_vk_x64)
    target_link_libraries(qlike_engine PUBLIC ffx_fsr2_api_vk_x64)
endif()
if(TARGET ffx_sdk_vk)
    # Runtime LoadLibrary path — the SDK's headers hardcode FFX_API_ENTRY
    # as __declspec(dllexport), so static linkage to the import lib
    # produces NULL function pointers in the consumer. AMD's recommended
    # consumer integration is ffx_api_loader.h + GetProcAddress at
    # runtime. So we DON'T link the .lib here, only pull in the headers
    # and rely on the staged DLL alongside the exe.
    target_include_directories(qlike_engine PUBLIC
        $<TARGET_PROPERTY:ffx_sdk_vk,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(qlike_engine PUBLIC QLIKE_HAVE_FSR3=1)
    # Compile fsr3.cpp with /EHa so __try/__except can catch hardware
    # AVs (NULL function calls, bad ptr derefs) coming back from the
    # SDK DLL. /EHsc (the project default) only catches synchronous
    # C++ exceptions and would let the SDK's AVs kill the process
    # before our SEH handler can flip the fatal flag.
    set_source_files_properties(src/engine/vk_engine/fsr3.cpp
        PROPERTIES COMPILE_FLAGS "/EHa")
endif()
# NVIDIA Streamline (DLSS SR / DLAA) - downloaded by scripts/fetch_streamline.ps1
# into external/streamline (not committed: ~280 MB). Manual-hooking
# integration: we link sl.interposer.lib only for the sl* API; it comes
# AFTER vulkan-1.lib on the link line, so every vk* call still resolves
# to the regular Vulkan loader.
set(SL_ROOT "${CMAKE_SOURCE_DIR}/external/streamline")
if(WIN32 AND EXISTS "${SL_ROOT}/lib/x64/sl.interposer.lib")
    target_include_directories(qlike_engine PUBLIC "${SL_ROOT}/include")
    target_compile_definitions(qlike_engine PUBLIC QLIKE_HAVE_STREAMLINE=1)
    # vulkan-1.lib again right before the interposer: libraries linked
    # after the first vulkan-1.lib (the FSR2 Vulkan backend) would otherwise
    # resolve vk* imports to sl.interposer.dll's stubs, which crash when
    # Streamline is not initialised (non-NVIDIA GPUs).
    # (CMake de-duplicates the same path, hence a renamed copy.)
    configure_file("${Vulkan_LIBRARY}" "${CMAKE_BINARY_DIR}/vulkan-1-before-sl.lib" COPYONLY)
    target_link_libraries(qlike_engine PUBLIC "${CMAKE_BINARY_DIR}/vulkan-1-before-sl.lib" "${SL_ROOT}/lib/x64/sl.interposer.lib")
    set(QLIKE_HAVE_SL ON)
    message(STATUS "NVIDIA Streamline SDK found - DLSS available (fsr_backend = 2)")
else()
    message(STATUS "NVIDIA Streamline SDK not present (run scripts/fetch_streamline.ps1) - DLSS disabled")
endif()
if(WIN32)
    target_compile_definitions(qlike_engine PUBLIC VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(qlike_engine PUBLIC VK_USE_PLATFORM_METAL_EXT)
elseif(UNIX)
    target_compile_definitions(qlike_engine PUBLIC VK_USE_PLATFORM_XLIB_KHR)
endif()

add_executable(vibe src/main.cpp)
target_link_libraries(vibe PRIVATE qlike_engine)

# Stage the FSR3 DLL next to vibe.exe so it loads at process start.
# Must be on the exe target (not qlike_engine, which is a static lib —
# its POST_BUILD doesn't carry over to the exe's output dir).
if(TARGET ffx_sdk_vk)
    add_custom_command(TARGET vibe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.dll"
            "$<TARGET_FILE_DIR:vibe>/amd_fidelityfx_vk.dll"
        COMMENT "Staging amd_fidelityfx_vk.dll alongside vibe.exe")
endif()
if(QLIKE_HAVE_SL)
    foreach(dll sl.interposer.dll sl.common.dll sl.dlss.dll nvngx_dlss.dll sl.dlss_d.dll nvngx_dlssd.dll NvLowLatencyVk.dll
                sl.dlss_g.dll nvngx_dlssg.dll sl.reflex.dll sl.pcl.dll)
        add_custom_command(TARGET vibe POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SL_ROOT}/bin/x64/${dll}" "$<TARGET_FILE_DIR:vibe>/${dll}"
            COMMENT "Staging ${dll} (Streamline / DLSS)")
    endforeach()
endif()
target_compile_definitions(qlike_engine PRIVATE
    QLIKE_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)

# --- Shader compilation ---
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.tesc"
    "${CMAKE_SOURCE_DIR}/shaders/*.tese"
    "${CMAKE_SOURCE_DIR}/shaders/*.comp")
# (grass_raymarch.frag picked up by the glob above)
# Shared GLSL headers (#include "xxx.glsl"): every shader depends on all of
# them so editing a header rebuilds its users.
file(GLOB SHADER_INCLUDES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/shaders/*.glsl")

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(COMPILED_SHADERS)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_OUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV_OUT}
        # -O = SPIR-V optimisation (perf preset). Enables dead-code
        # elimination, common-subexpression elim, loop unrolling.
        # Equivalent to running spirv-opt -O on the output.
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O ${SHADER} -o ${SPV_OUT}
        DEPENDS ${SHADER} ${SHADER_INCLUDES}
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM)
    list(APPEND COMPILED_SHADERS ${SPV_OUT})
endforeach()

# G-buffer-only variant of cube.frag: all forward lighting (RT shadows,
# path-traced GI, AO, fog) compiled out. Used for brushes / props / mesh
# terrain whenever the deferred lighting pass owns their shading — lower
# register pressure, no side-effect stores, far cheaper per fragment.
set(CUBE_GBUF_SPV "${SHADER_OUTPUT_DIR}/cube_gbuf.frag.spv")
add_custom_command(
    OUTPUT ${CUBE_GBUF_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DQLIKE_GBUFFER_ONLY=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/cube.frag -o ${CUBE_GBUF_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/cube.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader cube_gbuf.frag (G-buffer-only variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${CUBE_GBUF_SPV})

# Terrain-only G-buffer variant of cube.frag: is_terrain is a compile-time
# constant, so the brush / prop / castle-mesh paths are compiled out of the
# mesh-terrain pipeline (lower register pressure on the biggest raster pass).
set(CUBE_GBUF_TERRAIN_SPV "${SHADER_OUTPUT_DIR}/cube_gbuf_terrain.frag.spv")
add_custom_command(
    OUTPUT ${CUBE_GBUF_TERRAIN_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DQLIKE_GBUFFER_ONLY=1 -DQLIKE_TERRAIN_ONLY=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/cube.frag -o ${CUBE_GBUF_TERRAIN_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/cube.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader cube_gbuf_terrain.frag (terrain-only G-buffer variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${CUBE_GBUF_TERRAIN_SPV})

# Sun-shaft variant of compose.frag: the light shafts through silhouettes are
# added to the render-resolution scene colour BEFORE the upscaler, so their
# edges are resolved over time with everything else (after the upscaler they
# followed the jittered low-resolution depth: shaking, blocky outlines).
set(SUN_SHAFTS_SPV "${SHADER_OUTPUT_DIR}/sun_shafts.frag.spv")
add_custom_command(
    OUTPUT ${SUN_SHAFTS_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DSUN_SHAFTS=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/compose.frag -o ${SUN_SHAFTS_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/compose.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader sun_shafts.frag (compose sun-shaft variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${SUN_SHAFTS_SPV})

# Clouds in front of the land (camera inside or above the cloud layer), also
# before the upscaler: the same clouds, as sharp and steady as against the sky.
set(CLOUDS_OVER_SPV "${SHADER_OUTPUT_DIR}/clouds_over.frag.spv")
add_custom_command(
    OUTPUT ${CLOUDS_OVER_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DCLOUDS_OVER=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/compose.frag -o ${CLOUDS_OVER_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/compose.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader clouds_over.frag (compose clouds-over-land variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${CLOUDS_OVER_SPV})

# Screen-space displacement mapping: the three full-screen passes of ssdm_resolve.frag.
set(SSDM_GBUFFER_SPV "${SHADER_OUTPUT_DIR}/ssdm_gbuffer.frag.spv")
add_custom_command(
    OUTPUT ${SSDM_GBUFFER_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DSSDM_GBUFFER=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag -o ${SSDM_GBUFFER_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader ssdm_gbuffer.frag (ssdm_resolve: the whole G-buffer)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${SSDM_GBUFFER_SPV})
set(SSDM_COLOR_SPV "${SHADER_OUTPUT_DIR}/ssdm_color.frag.spv")
add_custom_command(
    OUTPUT ${SSDM_COLOR_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DSSDM_COLOR=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag -o ${SSDM_COLOR_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader ssdm_color.frag (ssdm_resolve: colour + resolve map)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${SSDM_COLOR_SPV})
set(SSDM_DEPTH_SPV "${SHADER_OUTPUT_DIR}/ssdm_depth.frag.spv")
add_custom_command(
    OUTPUT ${SSDM_DEPTH_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DSSDM_DEPTH=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag -o ${SSDM_DEPTH_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader ssdm_depth.frag (ssdm_resolve: displaced depth)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${SSDM_DEPTH_SPV})
set(SSDM_VELOCITY_SPV "${SHADER_OUTPUT_DIR}/ssdm_velocity.frag.spv")
add_custom_command(
    OUTPUT ${SSDM_VELOCITY_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DSSDM_VELOCITY=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag -o ${SSDM_VELOCITY_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/ssdm_resolve.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader ssdm_velocity.frag (ssdm_resolve: motion vectors)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${SSDM_VELOCITY_SPV})

# Sky-fill variant of compose.frag: writes the real sky into the render-res
# scene colour BEFORE DLSS (exact colour/alpha alignment), so DLSS upscales
# a finished sky instead of compose rebuilding it at silhouettes afterwards.
set(SKY_FILL_SPV "${SHADER_OUTPUT_DIR}/sky_fill.frag.spv")
add_custom_command(
    OUTPUT ${SKY_FILL_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DSKY_FILL=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/compose.frag -o ${SKY_FILL_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/compose.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader sky_fill.frag (compose sky-fill variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${SKY_FILL_SPV})

# Ray-marched terrain without the winter code (snow layer, footprints, ice):
# what is bound whenever there is no snow. The same source with QLIKE_SUMMER.
set(TR_SUMMER_SPV "${SHADER_OUTPUT_DIR}/terrain_raymarch_summer.frag.spv")
add_custom_command(
    OUTPUT ${TR_SUMMER_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DQLIKE_SUMMER=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag -o ${TR_SUMMER_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader terrain_raymarch_summer.frag (no winter code)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${TR_SUMMER_SPV})

# Water surface pass: same source as the ray-marched terrain, water branch.
# The "first start" build of the terrain shader: small, compiled by the driver in a moment; the
# game runs on it while the full shader compiles in the background (see QLIKE_LITE in the shader).
set(TR_LITE_SPV "${SHADER_OUTPUT_DIR}/terrain_raymarch_lite.frag.spv")
add_custom_command(
    OUTPUT ${TR_LITE_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DQLIKE_LITE=1 -DQLIKE_LITE_MASK=65023 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag -o ${TR_LITE_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader terrain_raymarch_lite.frag (first-start variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${TR_LITE_SPV})

# The low-resolution march of the terrain (QLIKE_MARCH_ONLY in the shader): the same source,
# cut down to "how far can a ray go before the ground comes near" - one value per block of pixels.
set(TR_MARCH_SPV "${SHADER_OUTPUT_DIR}/terrain_march_lo.frag.spv")
add_custom_command(
    OUTPUT ${TR_MARCH_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DQLIKE_MARCH_ONLY=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag -o ${TR_MARCH_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader terrain_march_lo.frag (low-resolution march variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${TR_MARCH_SPV})

set(WATER_SPV "${SHADER_OUTPUT_DIR}/water.frag.spv")
add_custom_command(
    OUTPUT ${WATER_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O
            -DQLIKE_WATER=1 -fshader-stage=frag
            ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag -o ${WATER_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/terrain_raymarch.frag ${SHADER_INCLUDES}
    COMMENT "Compiling shader water.frag (terrain_raymarch water variant)"
    VERBATIM)
list(APPEND COMPILED_SHADERS ${WATER_SPV})

if(COMPILED_SHADERS)
    add_custom_target(shaders ALL DEPENDS ${COMPILED_SHADERS})
    add_dependencies(vibe shaders)
endif()

enable_testing()
add_subdirectory(tests)
