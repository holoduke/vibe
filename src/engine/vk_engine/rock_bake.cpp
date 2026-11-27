// The rock kit's bake (tools/rocks/make_formations.py -> assets/rocks/kit_bake.bin) on
// the GPU: one array texture of two-channel block-compressed layers with mips, at scene
// binding 63 (shaders/rock_breaker.glsl u_rock_bake):
//   three layers a piece  - (normal x, normal z), (normal y, height), (occlusion, the ground's
//                           weight on tops) - the piece's own axes, its own unwrap (the mesh's uv);
//   then two layers       - the BREAKER pattern every steep rock face shares, formation or
//                           cliff: (strata tone, stain tone), (relief, cracks); a tile is 24 m.
// What the shaders may use is told in the ground effects' block (gfx.rock: ground_fx.cpp
// reads rock_bake_block()). Switches, each takes one part away:
//   VIBE_NO_ROCK_TOPS=1     the ground's material on tops and ledges
//   VIBE_NO_ROCK_BAKE=1     the baked normal and height (occlusion and the tops' weight stay)
//   VIBE_NO_ROCK_BREAKER=1  the breaker pattern and the colour strip by altitude
//   VIBE_NO_ROCK_ACCUM=1    snow lying on up faces with a crisp edge, near detail roughness
// The binding is always valid: without the file (or without the format) a 4 x 4 stand-in.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/rock_bake.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vibe {

void VulkanEngine::init_rock_bake() {
    if (!rock_bake_) rock_bake_ = new RockBakeState();
    RockBakeState& B = *rock_bake_;
    B.switches = (std::getenv("VIBE_NO_ROCK_TOPS") ? 0 : 1) | (std::getenv("VIBE_NO_ROCK_BAKE") ? 0 : 2) |
                 (std::getenv("VIBE_NO_ROCK_BREAKER") ? 0 : 4) | (std::getenv("VIBE_NO_ROCK_ACCUM") ? 0 : 8);
    const VkFormat fmt = VK_FORMAT_BC5_UNORM_BLOCK;
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(physical_device_, fmt, &fp);
    const bool fmt_ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

    std::vector<uint8_t> data;
    uint32_t hdr[4] = { 4, 1, 1, 0 };                    // size, layers, mips, pieces
    bool ok = false;
    if (fmt_ok && !std::getenv("VIBE_NO_ROCK_BAKE_FILE")) {
        if (FILE* fp2 = std::fopen("assets/rocks/kit_bake.bin", "rb")) {
            char magic[4];
            ok = std::fread(magic, 1, 4, fp2) == 4 && std::memcmp(magic, "RKB1", 4) == 0 && std::fread(hdr, 4, 4, fp2) == 4 &&
                 hdr[0] >= 64 && hdr[0] <= 4096 && (hdr[0] & (hdr[0] - 1)) == 0 && hdr[1] >= 1 && hdr[1] <= 256 && hdr[2] >= 1 && hdr[2] <= 8 &&
                 hdr[3] * 3 + 2 == hdr[1] && (hdr[0] >> (hdr[2] - 1)) >= 4;
            if (ok) {
                B.range.resize(hdr[3]);
                ok = std::fread(B.range.data(), 4, hdr[3], fp2) == hdr[3];
                size_t per = 0;
                for (uint32_t m = 0; m < hdr[2]; ++m) { const size_t s = hdr[0] >> m; per += (s / 4) * (s / 4) * 16; }
                data.resize(per * hdr[1]);
                ok = ok && std::fread(data.data(), 1, data.size(), fp2) == data.size();
            }
            std::fclose(fp2);
        }
    }
    // THE SHORE KIT's bake (tools/shore/make_shore_kit.py -> assets/shore/shore_bake.bin): its
    // pieces follow the rock kit's in the kit (rock_formations_gpu.cpp loads assets/shore/shore.rockf
    // as a second kit, appended), so their layers go between the rock pieces' and the breaker's.
    // Its layers are half the size (512): each mip is taken one level down, the top level made
    // by doubling the 512 one (nearest, block by block - BC5 is two BC4 blocks: the endpoints
    // stay, every texel's 3-bit index is the source texel's it covers).
    // THE CLIFF WALLS' bake (make_formations.py --set cliffs -> assets/cliffs/cliffs_bake.bin, 512)
    // the same way after the shore kit's: the third kit (rock_cliffs.cpp) follows the shore kit -
    // when the shore kit is wanted but its bake is not there, the cliffs' is not taken either
    // (its layers would stand where the shore pieces' belong).
    bool prev_ok = true;
    for (int kk = 0; kk < 2; ++kk) {
    const char* bake_path = kk ? "assets/cliffs/cliffs_bake.bin" : "assets/shore/shore_bake.bin";
    const bool wanted = !std::getenv(kk ? "VIBE_NO_CLIFF_WALLS" : "VIBE_NO_SHORE_KIT");
    bool merged = false;
    if (ok && wanted && prev_ok) {
        if (FILE* fs = std::fopen(bake_path, "rb")) {
            char magic[4]; uint32_t hs[4] = {};
            bool sok = std::fread(magic, 1, 4, fs) == 4 && std::memcmp(magic, "RKB1", 4) == 0 && std::fread(hs, 4, 4, fs) == 4 &&
                       hs[3] >= 1 && hs[3] <= 64 && hs[1] == hs[3] * 3 + 2 && hs[2] >= 1 && hs[2] <= 8 &&
                       (hs[0] == hdr[0] || (hs[0] * 2 == hdr[0] && hs[2] + 1 >= hdr[2]));
            std::vector<float> srange;
            std::vector<uint8_t> sdata;
            size_t sper = 0;
            if (sok) {
                srange.resize(hs[3]);
                sok = std::fread(srange.data(), 4, hs[3], fs) == hs[3];
                for (uint32_t m = 0; m < hs[2]; ++m) { const size_t s = hs[0] >> m; sper += (s / 4) * (s / 4) * 16; }
                sdata.resize(sper * size_t(hs[3]) * 3);         // (its pieces' layers; its breaker is not read)
                sok = sok && std::fread(sdata.data(), 1, sdata.size(), fs) == sdata.size();
            }
            std::fclose(fs);
            if (sok) {
                size_t per = 0;
                for (uint32_t m = 0; m < hdr[2]; ++m) { const size_t s = hdr[0] >> m; per += (s / 4) * (s / 4) * 16; }
                const bool half = hs[0] != hdr[0];
                std::vector<uint8_t> out;
                out.reserve(per * (size_t(hdr[1]) + size_t(hs[3]) * 3));
                out.insert(out.end(), data.begin(), data.begin() + per * size_t(hdr[3]) * 3);   // the rock pieces
                auto bc4_double = [](const uint8_t* src, int qx, int qy, uint8_t* dst) {
                    uint64_t si = 0, di = 0;
                    for (int b = 0; b < 6; ++b) si |= uint64_t(src[2 + b]) << (8 * b);
                    for (int y = 0; y < 4; ++y)
                        for (int x = 0; x < 4; ++x) {
                            const int s = (qy * 2 + y / 2) * 4 + (qx * 2 + x / 2);
                            di |= ((si >> (3 * s)) & 7u) << (3 * (y * 4 + x));
                        }
                    dst[0] = src[0]; dst[1] = src[1];
                    for (int b = 0; b < 6; ++b) dst[2 + b] = uint8_t(di >> (8 * b));
                };
                for (uint32_t l = 0; l < hs[3] * 3; ++l) {
                    const uint8_t* L = sdata.data() + sper * l;
                    // the source's mips' offsets
                    std::vector<size_t> so(hs[2], 0);
                    for (uint32_t m = 1; m < hs[2]; ++m) { const size_t s = hs[0] >> (m - 1); so[m] = so[m - 1] + (s / 4) * (s / 4) * 16; }
                    for (uint32_t m = 0; m < hdr[2]; ++m) {
                        const uint32_t s = hdr[0] >> m, nb = s / 4;
                        if (!half) { out.insert(out.end(), L + so[m], L + so[m] + size_t(nb) * nb * 16); continue; }
                        if (m > 0) { out.insert(out.end(), L + so[m - 1], L + so[m - 1] + size_t(nb) * nb * 16); continue; }
                        const uint32_t snb = nb / 2;
                        const size_t at = out.size();
                        out.resize(at + size_t(nb) * nb * 16);
                        for (uint32_t by = 0; by < nb; ++by)
                            for (uint32_t bx = 0; bx < nb; ++bx) {
                                const uint8_t* sb = L + (size_t(by / 2) * snb + bx / 2) * 16;
                                uint8_t* db = out.data() + at + (size_t(by) * nb + bx) * 16;
                                bc4_double(sb, int(bx & 1), int(by & 1), db);
                                bc4_double(sb + 8, int(bx & 1), int(by & 1), db + 8);
                            }
                    }
                }
                out.insert(out.end(), data.begin() + per * size_t(hdr[3]) * 3, data.end());       // the breaker
                data.swap(out);
                B.range.insert(B.range.end(), srange.begin(), srange.end());
                hdr[3] += hs[3];
                hdr[1] += hs[3] * 3;
                log::infof("[rocks] the %s kit's bake: %u pieces%s", kk ? "cliff" : "shore", hs[3], half ? " (512, doubled to the rock kit's size)" : "");
                merged = true;
            } else {
                log::warnf("[rocks] %s missing or not understood - those pieces have no bake", bake_path);
            }
        }
    }
    if (wanted) prev_ok = merged;
    }
    if (!ok) {
        hdr[0] = 4; hdr[1] = 1; hdr[2] = 1; hdr[3] = 0;
        data.assign(16, 0);
        data[0] = data[1] = data[8] = data[9] = 128;
        B.range.clear();
        log::warnf("[rocks] assets/rocks/kit_bake.bin %s - the rock kit has no bake and steep rock no breaker pattern (tools/rocks/make_formations.py makes it)",
                   fmt_ok ? "missing or not understood" : "cannot be used: no two-channel block format on this GPU");
    }
    B.size = int(hdr[0]); B.layers = int(hdr[1]); B.mips = int(hdr[2]); B.pieces = int(hdr[3]);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt_ok ? fmt : VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = { hdr[0], hdr[0], 1 };
    ici.mipLevels = hdr[2];
    ici.arrayLayers = hdr[1];
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &B.image, &B.alloc, nullptr), "rock bake image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = B.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, hdr[2], 0, hdr[1] };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &B.view), "rock bake view");
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy = 4.0f;
    si.maxLod = float(hdr[2] - 1);
    vk_check(vkCreateSampler(device_, &si, nullptr, &B.sampler), "rock bake sampler");

    // The upload: one staging buffer, a copy per layer and mip (the file's order).
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation sa = nullptr;
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = fmt_ok ? data.size() : 64;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo bac{};
        bac.usage = VMA_MEMORY_USAGE_AUTO;
        bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &bac, &stage, &sa, &info), "rock bake staging");
        if (fmt_ok) std::memcpy(info.pMappedData, data.data(), data.size());
        else std::memset(info.pMappedData, 128, 64);
        vmaFlushAllocation(allocator_, sa, 0, VK_WHOLE_SIZE);
    }
    std::vector<VkBufferImageCopy> regs;
    VkDeviceSize off = 0;
    for (uint32_t l = 0; l < hdr[1]; ++l)
        for (uint32_t m = 0; m < hdr[2]; ++m) {
            const uint32_t s = hdr[0] >> m;
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, l, 1 };
            r.imageExtent = { s, s, 1 };
            regs.push_back(r);
            off += VkDeviceSize(s / 4) * (s / 4) * 16;
        }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = B.image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, hdr[2], 0, hdr[1] };
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        vkCmdCopyBufferToImage(cb, stage, B.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regs.size()), regs.data());
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    });
    vmaDestroyBuffer(allocator_, stage, sa);

    // The tops picture: empty until rockf_tops_poll fills it.
    {
        VkImageCreateInfo ti = ici;
        ti.format = VK_FORMAT_R32G32_SFLOAT;
        ti.extent = { uint32_t(kRockTopsN), uint32_t(kRockTopsN), 1 };
        ti.mipLevels = 1; ti.arrayLayers = 1;
        vk_check(vmaCreateImage(allocator_, &ti, &aci, &B.tops_image, &B.tops_alloc, nullptr), "rock tops image");
        VkImageViewCreateInfo tv{};
        tv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        tv.image = B.tops_image; tv.viewType = VK_IMAGE_VIEW_TYPE_2D; tv.format = ti.format;
        tv.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &tv, nullptr, &B.tops_view), "rock tops view");
        VkSamplerCreateInfo ts{};
        ts.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ts.magFilter = ts.minFilter = VK_FILTER_NEAREST;
        ts.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ts.addressModeU = ts.addressModeV = ts.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &ts, nullptr, &B.tops_sampler), "rock tops sampler");
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = B.tops_image; b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            VkClearColorValue z{};
            VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, B.tops_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &r);
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        });
    }
    VkDescriptorImageInfo ii{ B.sampler, B.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_; w.dstBinding = 63; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    if (ok) log::infof("[rocks] bake: %d pieces + the breaker pattern, %d layers of %d x %d with %d mips (%.1f MB), switches %d",
                       B.pieces, B.layers, B.size, B.size, B.mips, double(data.size()) / (1024.0 * 1024.0), B.switches);
}

// (x pieces with a bake - 0: none, nothing of it may be read -, y the switches, z the breaker's first layer, w the bake's size)
glm::vec4 VulkanEngine::rock_bake_block() const {
    if (!rock_bake_) return glm::vec4(0.0f);
    return glm::vec4(float(rock_bake_->pieces), float(rock_bake_->switches), float(rock_bake_->pieces * 3), float(rock_bake_->size));
}

VkImageView VulkanEngine::rock_tops_view() const { return rock_bake_ ? rock_bake_->tops_view : VK_NULL_HANDLE; }
VkSampler   VulkanEngine::rock_tops_sampler() const { return rock_bake_ ? rock_bake_->tops_sampler : VK_NULL_HANDLE; }

// The tops picture sent to the GPU (a blocking upload of 1.2 MB: only when the window moves on
// or the formations in it change - rockf_tops_poll).
void VulkanEngine::rock_tops_upload(const float* rg) {
    if (!rock_bake_ || !rock_bake_->tops_image) return;
    RockBakeState& B = *rock_bake_;
    const VkDeviceSize bytes = VkDeviceSize(kRockTopsN) * kRockTopsN * 8;
    VkBuffer stage = VK_NULL_HANDLE; VmaAllocation sa = nullptr;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    vk_check(vmaCreateBuffer(allocator_, &bci, &bac, &stage, &sa, &info), "rock tops staging");
    std::memcpy(info.pMappedData, rg, size_t(bytes));
    vmaFlushAllocation(allocator_, sa, 0, VK_WHOLE_SIZE);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = B.tops_image; b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy r{};
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.imageExtent = { uint32_t(kRockTopsN), uint32_t(kRockTopsN), 1 };
        vkCmdCopyBufferToImage(cb, stage, B.tops_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    });
    vmaDestroyBuffer(allocator_, stage, sa);
}

void VulkanEngine::destroy_rock_bake() {
    if (!rock_bake_) return;
    RockBakeState& B = *rock_bake_;
    if (B.tops_sampler) vkDestroySampler(device_, B.tops_sampler, nullptr);
    if (B.tops_view) vkDestroyImageView(device_, B.tops_view, nullptr);
    if (B.tops_image) vmaDestroyImage(allocator_, B.tops_image, B.tops_alloc);
    if (B.sampler) vkDestroySampler(device_, B.sampler, nullptr);
    if (B.view) vkDestroyImageView(device_, B.view, nullptr);
    if (B.image) vmaDestroyImage(allocator_, B.image, B.alloc);
    delete rock_bake_;
    rock_bake_ = nullptr;
}

}  // namespace vibe
