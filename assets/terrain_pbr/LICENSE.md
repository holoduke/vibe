# Licences of the scanned material sets in this folder

Every set here comes from ambientCG (https://ambientcg.com), published under the
Creative Commons CC0 1.0 Universal licence (public domain dedication, no attribution
required: https://docs.ambientcg.com/license/). The folder name is the asset id on the site
(https://ambientcg.com/view?id=<id>). They are loaded by `load_terrain_pbr_overrides`
(`src/engine/vk_engine/world_textures.cpp`).

| Set | What | Used for |
|---|---|---|
| Rock035 | rock | the landscape's rock (slot 7) |
| Grass004 | grass | slot 8 |
| Ground024 | forest ground | dirt (slot 9) |
| Ground027 | sand | slot 10 |
| Snow006 | snow | slot 11 |
| RoofingTiles012A | roof tiles | slot 13 |
| PavingStones142 (PavingStones131 kept) | cobbles | the castle road (slot 14) |
| Bark012 | bark | slot 16 |
| Asphalt031 | asphalt | the highway (slot 32) |
| Gravel040 | gravel | the highway's verge (slot 33) |
| Rocks024S (1K) | flat grey river-bed stones | the bed under running water (slot 43) |
| Ground083 (1K) | wet river sand / silt | the bed under slow and deep water (slot 44) |
| Rock046S (1K) | grey rock with moss | the bed where it is steep (slot 45) |
| Moss001 (1K) | ground moss | growth on submerged up-facing ground (slot 46) |

The four bed sets were downloaded 2026-11-26 as the site's `<id>_1K-JPG.zip` archives; only the
colour, GL normal, displacement, roughness and ambient-occlusion maps are kept.
