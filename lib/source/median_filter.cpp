#include "median_filter.hpp"
#include "median_sorting_networks.hpp"
#include <riscv_vector.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

// Reusable padded row loader
template <PaddingType Padding, int HALF_K>
inline void LoadPaddedRow(uint8_t* dst, const Image<uint8_t, ImageType::GRAY>& input,
                          int y, int width, int height, uint8_t constant_value)
{
    const uint8_t* src_row = input.GetRowPtr<Padding>(y);

    if constexpr (Padding == PaddingType::CONSTANT)
    {
        if (src_row == nullptr)
        {
            size_t total = width + 2 * HALF_K;
            size_t vl = __riscv_vsetvl_e8m8(total);
            vuint8m8_t v_const = __riscv_vmv_v_x_u8m8(constant_value, vl);
            for (size_t i = 0; i < total; i += vl)
            {
                vl = __riscv_vsetvl_e8m8(total - i);
                __riscv_vse8_v_u8m8(dst + i, v_const, vl);
            }
            return;
        }

        for (int i = 0; i < HALF_K; ++i) dst[i] = constant_value;
        for (int i = 0; i < HALF_K; ++i) dst[HALF_K + width + i] = constant_value;
    }
    else if constexpr (Padding == PaddingType::REPLICA)
    {
        for (int i = 0; i < HALF_K; ++i) dst[i] = src_row[0];
        for (int i = 0; i < HALF_K; ++i) dst[HALF_K + width + i] = src_row[width - 1];
    }
    else if constexpr (Padding == PaddingType::REFLECT)
    {
        for (int i = 0; i < HALF_K; ++i) dst[i] = src_row[HALF_K - i];
        for (int i = 0; i < HALF_K; ++i) dst[HALF_K + width + i] = src_row[width - 2 - i];
    }

    int c = 0;
    while (c < width)
    {
        size_t vl = __riscv_vsetvl_e8m8(width - c);
        vuint8m8_t v = __riscv_vle8_v_u8m8(src_row + c, vl);
        __riscv_vse8_v_u8m8(dst + HALF_K + c, v, vl);
        c += vl;
    }
}

// =============================================================================
// Unified Vector Median Filter Implementation (Reused for all kernels)
// =============================================================================
template <int KERNEL_SIZE, PaddingType Padding>
void MedianFilter_impl(const Image<uint8_t, ImageType::GRAY>& input,
                       Image<uint8_t, ImageType::GRAY>& output,
                       uint8_t constant_value)
{
    constexpr int HALF_K = KERNEL_SIZE / 2;
    const int width  = static_cast<int>(input.Width());
    const int height = static_cast<int>(input.Height());
    assert(width == static_cast<int>(output.Width()) && height == static_cast<int>(output.Height()));
    assert(width <= MAX_WIDTH);

    const int padded_width = width + 2 * HALF_K;

    // Static allocation in .bss to prevent stack overflow on baremetal GEM5
    alignas(32) static uint8_t row_buffers[KERNEL_SIZE][MAX_WIDTH + 64];
    alignas(32) static uint8_t vert_rows[KERNEL_SIZE][MAX_WIDTH + 64];
    alignas(32) static uint8_t chunk_buf[KERNEL_SIZE][KERNEL_SIZE][32];

    int ring_head = 0;
    for (int y = -HALF_K; y < HALF_K; ++y)
    {
        LoadPaddedRow<Padding, HALF_K>(row_buffers[ring_head], input, y, width, height, constant_value);
        ring_head = (ring_head + 1) % KERNEL_SIZE;
    }

    for (int y = 0; y < height; ++y)
    {
        LoadPaddedRow<Padding, HALF_K>(row_buffers[ring_head], input, y + HALF_K, width, height, constant_value);
        ring_head = (ring_head + 1) % KERNEL_SIZE;

        const uint8_t* active_rows[KERNEL_SIZE];
        for (int k = 0; k < KERNEL_SIZE; ++k)
        {
            active_rows[k] = row_buffers[(ring_head + k) % KERNEL_SIZE];
        }

        // Phase 1: Pre-sort columns vertically across entire row width
        SortColumnsVertically<KERNEL_SIZE>(active_rows, vert_rows, padded_width);

        uint8_t* out_row = output.GetPtr(0, y);
        int x = 0;
        while (x < width)
        {
            size_t vl = __riscv_vsetvl_e8m2(width - x);

            // Phase 2: Sort rows horizontally
            SortRowsHorizontally<KERNEL_SIZE>(vert_rows, chunk_buf, x, vl);

            // Phase 3+: Diagonal reductions to extract true mathematical median
            vuint8m2_t v_med = ExtractMedian<KERNEL_SIZE>(chunk_buf, vl);

            __riscv_vse8_v_u8m2(out_row + x, v_med, vl);
            x += vl;
        }
    }
}

template <PaddingType Padding>
inline void MedianFilter_dispatch(const Image<uint8_t, ImageType::GRAY>& input,
                                  Image<uint8_t, ImageType::GRAY>& output,
                                  int kernel_size, uint8_t constant_value)
{
    switch (kernel_size)
    {
    case 3:  MedianFilter_impl<3, Padding>(input, output, constant_value); break;
    case 5:  MedianFilter_impl<5, Padding>(input, output, constant_value); break;
    case 7:  MedianFilter_impl<7, Padding>(input, output, constant_value); break;
    case 9:  MedianFilter_impl<9, Padding>(input, output, constant_value); break;
    case 11: MedianFilter_impl<11, Padding>(input, output, constant_value); break;
    default: assert(false && "Unsupported kernel size for median filter"); break;
    }
}

} // namespace

void vec::MedianFilter(const Image<uint8_t, ImageType::GRAY>& input,
                       Image<uint8_t, ImageType::GRAY>& output,
                       int kernel_size,
                       PaddingType padding_type,
                       uint8_t constant_value)
{
    switch (padding_type)
    {
    case PaddingType::CONSTANT: MedianFilter_dispatch<PaddingType::CONSTANT>(input, output, kernel_size, constant_value); break;
    case PaddingType::REPLICA:  MedianFilter_dispatch<PaddingType::REPLICA>(input, output, kernel_size, constant_value); break;
    case PaddingType::REFLECT:  MedianFilter_dispatch<PaddingType::REFLECT>(input, output, kernel_size, constant_value); break;
    }
}
