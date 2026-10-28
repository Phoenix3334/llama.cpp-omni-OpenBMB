#pragma once

#include "ggml.h"

#include <cstddef>
#include <vector>

// Internal helpers shared by the vision runtime and its weight-free regression.
inline bool vision_patch_mm_supported(int nx, int ny, int patch_size, int batch_size,
                                      const ggml_tensor * weight) {
    return batch_size == 1 && patch_size > 0 && nx > 0 && ny > 0 &&
           nx % patch_size == 0 && ny % patch_size == 0 && weight &&
           weight->ne[0] == patch_size && weight->ne[1] == patch_size &&
           weight->ne[2] == 3 && ggml_is_contiguous(weight) &&
           (weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_F32);
}

inline std::vector<float> vision_pack_patches(const float * rgb, int nx, int ny, int patch_size) {
    GGML_ASSERT(rgb && patch_size > 0 && nx > 0 && ny > 0);
    GGML_ASSERT(nx % patch_size == 0 && ny % patch_size == 0);

    const int patches_x = nx / patch_size;
    const int patches_y = ny / patch_size;
    const size_t k_dim = (size_t) patch_size * patch_size * 3;
    std::vector<float> patches(k_dim * patches_x * patches_y);

    // Match the flattened conv-weight order: kw, kh, channel, output channel.
    // This replaces the existing HWC -> CHW pass, rather than adding a pass.
    for (int py = 0; py < patches_y; ++py) {
        for (int px = 0; px < patches_x; ++px) {
            const size_t patch = (size_t) py * patches_x + px;
            for (int c = 0; c < 3; ++c) {
                for (int kh = 0; kh < patch_size; ++kh) {
                    const float * src = rgb + 3 * ((size_t) (py * patch_size + kh) * nx + px * patch_size) + c;
                    float * dst = patches.data() + k_dim * patch + ((size_t) c * patch_size + kh) * patch_size;
                    for (int kw = 0; kw < patch_size; ++kw) {
                        dst[kw] = src[3 * kw];
                    }
                }
            }
        }
    }
    return patches;
}

inline ggml_tensor * vision_build_patch_mm(ggml_context * ctx, ggml_tensor * weight,
                                          ggml_tensor * bias, int nx, int ny, int patch_size) {
    GGML_ASSERT(vision_patch_mm_supported(nx, ny, patch_size, 1, weight));
    const int64_t k_dim = (int64_t) patch_size * patch_size * 3;
    const int64_t n_patches = (int64_t) (nx / patch_size) * (ny / patch_size);
    ggml_tensor * patches = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_dim, n_patches);
    ggml_set_name(patches, "inp_raw");
    ggml_set_input(patches);
    ggml_tensor * weights = ggml_reshape_2d(ctx, weight, k_dim, weight->ne[3]);
    ggml_tensor * result = ggml_mul_mat(ctx, weights, patches);
    if (bias) {
        result = ggml_add(ctx, result, bias);
    }
    return result;
}
