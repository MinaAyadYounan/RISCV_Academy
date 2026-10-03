#include "box_filter.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace {

// Pads the left and right margins of the 1D column-sum buffer for horizontal convolution.
template <PaddingType Padding>
inline void ApplyHorizontalPadding(uint16_t* col_sums, int width, int padding, int kernel_size, uint8_t constant_value)
{
    for (int p = 1; p <= padding; ++p)
    {
        if constexpr (Padding == PaddingType::CONSTANT)
        {
            col_sums[-p]              = static_cast<uint16_t>(kernel_size * constant_value);
            col_sums[width - 1 + p]   = static_cast<uint16_t>(kernel_size * constant_value);
        }
        else if constexpr (Padding == PaddingType::REPLICA)
        {
            col_sums[-p]              = col_sums[0];
            col_sums[width - 1 + p]   = col_sums[width - 1];
        }
        else if constexpr (Padding == PaddingType::REFLECT)
        {
            const bool width_gt_1 = (width > 1);
            col_sums[-p]              = width_gt_1 ? col_sums[p] : col_sums[0];
            col_sums[width - 1 + p]   = width_gt_1 ? col_sums[width - 1 - p] : col_sums[width - 1];
        }
    }
}

// 1D horizontal sliding-window convolution + Q16 fixed-point normalization.
inline void ComputeHorizontalPass(const uint16_t* col_sums, uint8_t* out_row, int width, int padding, uint32_t norm_mul)
{
    // 1. Initial horizontal window sum for x = 0
    uint32_t sum_h = 0;
    for (int m = -padding; m <= padding; ++m)
    {
        sum_h += col_sums[m];
    }
    out_row[0] = static_cast<uint8_t>((sum_h * norm_mul) >> 16);

    // 2. Slide across x = 1 to width - 1 (1 add, 1 sub per pixel)
    for (int x = 1; x < width; ++x)
    {
        sum_h += col_sums[x + padding] - col_sums[x - padding - 1];
        out_row[x] = static_cast<uint8_t>((sum_h * norm_mul) >> 16);
    }
}

template <PaddingType Padding>
void BoxFilter_impl(const Image<uint8_t, ImageType::GRAY>& input,
                    Image<uint8_t, ImageType::GRAY>& output,
                    int kernel_size,
                    uint8_t constant_value)
{
    const int width  = static_cast<int>(input.Width());
    const int height = static_cast<int>(input.Height());
    assert(width == static_cast<int>(output.Width()) && height == static_cast<int>(output.Height()));
    assert(kernel_size % 2 == 1 && "Kernel size must be odd");

    const int padding = kernel_size / 2;
    constexpr int frac_bits = 16;
    const uint32_t norm_mul = static_cast<uint32_t>((1u << frac_bits) * (1.0f / (kernel_size * kernel_size)) + 0.5f);

    constexpr int MAX_PAD = 16;
    constexpr int MAX_WIDTH = 2048;
    assert(width <= MAX_WIDTH);

    // 1D stack buffer holding vertical column sums: [left_padding | width | right_padding]
    uint16_t buffer[MAX_WIDTH + 2 * MAX_PAD];
    uint16_t* col_sums = buffer + MAX_PAD;

    // -------------------------------------------------------------------------
    // Row 0: Warm-up vertical sum for all columns
    // -------------------------------------------------------------------------
    for (int x = 0; x < width; ++x)
    {
        col_sums[x] = 0;
    }
    for (int k = -padding; k <= padding; ++k)
    {
        const uint8_t* row = input.GetRowPtr<Padding>(k);
        if (row)
        {
            for (int x = 0; x < width; ++x)
            {
                col_sums[x] += row[x];
            }
        }
        else
        {
            for (int x = 0; x < width; ++x)
            {
                col_sums[x] += constant_value;
            }
        }
    }

    ApplyHorizontalPadding<Padding>(col_sums, width, padding, kernel_size, constant_value);
    ComputeHorizontalPass(col_sums, output.GetPtr(0, 0), width, padding, norm_mul);

    // -------------------------------------------------------------------------
    // Rows 1 to height - 1: Vertical sliding window (1 exit row, 1 enter row)
    // -------------------------------------------------------------------------
    for (int y = 1; y < height; ++y)
    {
        const uint8_t* row_exit  = input.GetRowPtr<Padding>(y - padding - 1);
        const uint8_t* row_enter = input.GetRowPtr<Padding>(y + padding);

        for (int x = 0; x < width; ++x)
        {
            const uint16_t v_exit  = row_exit  ? row_exit[x]  : constant_value;
            const uint16_t v_enter = row_enter ? row_enter[x] : constant_value;
            col_sums[x] += v_enter - v_exit;
        }

        ApplyHorizontalPadding<Padding>(col_sums, width, padding, kernel_size, constant_value);
        ComputeHorizontalPass(col_sums, output.GetPtr(0, y), width, padding, norm_mul);
    }
}

} // namespace

void ref::BoxFilter(const Image<uint8_t, ImageType::GRAY>& input,
                    Image<uint8_t, ImageType::GRAY>& output,
                    int kernel_size,
                    PaddingType padding_type,
                    uint8_t constant_value)
{
    switch (padding_type)
    {
    case PaddingType::CONSTANT:
        BoxFilter_impl<PaddingType::CONSTANT>(input, output, kernel_size, constant_value);
        break;
    case PaddingType::REPLICA:
        BoxFilter_impl<PaddingType::REPLICA>(input, output, kernel_size, constant_value);
        break;
    case PaddingType::REFLECT:
        BoxFilter_impl<PaddingType::REFLECT>(input, output, kernel_size, constant_value);
        break;
    }
}
