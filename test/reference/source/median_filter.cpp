#include "median_filter.hpp"
#include "median_sorting_networks.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

template <PaddingType Padding, int HALF_K>
inline void LoadPaddedRow(uint8_t* dst, const Image<uint8_t, ImageType::GRAY>& input,
                          int y, int width, int height, uint8_t constant_value)
{
    const uint8_t* src_row = input.GetRowPtr<Padding>(y);

    if constexpr (Padding == PaddingType::CONSTANT)
    {
        if (src_row == nullptr)
        {
            std::fill(dst, dst + width + 2 * HALF_K, constant_value);
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

    std::copy(src_row, src_row + width, dst + HALF_K);
}

// Unified scalar runner (Reused for all kernels)
template <int KERNEL_SIZE, PaddingType Padding>
void MedianFilter_scalar_impl(const Image<uint8_t, ImageType::GRAY>& input,
                             Image<uint8_t, ImageType::GRAY>& output,
                             uint8_t constant_value)
{
    constexpr int HALF_K = KERNEL_SIZE / 2;
    const int width  = static_cast<int>(input.Width());
    const int height = static_cast<int>(input.Height());
    assert(width == static_cast<int>(output.Width()) && height == static_cast<int>(output.Height()));
    assert(width <= MAX_WIDTH);

    const int padded_width = width + 2 * HALF_K;
    alignas(32) static uint8_t row_buffers[KERNEL_SIZE][MAX_WIDTH + 64];
    alignas(32) static uint8_t vert_rows[KERNEL_SIZE][MAX_WIDTH + 64];

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

        SortColumnsVerticallyScalar<KERNEL_SIZE>(active_rows, vert_rows, padded_width);

        uint8_t* out_row = output.GetPtr(0, y);
        for (int x = 0; x < width; ++x)
        {
            uint8_t A[KERNEL_SIZE][KERNEL_SIZE];
            SortRowsHorizontallyScalar<KERNEL_SIZE>(vert_rows, A, x);
            out_row[x] = ExtractMedianScalar<KERNEL_SIZE>(A);
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
    case 3:  MedianFilter_scalar_impl<3, Padding>(input, output, constant_value); break;
    case 5:  MedianFilter_scalar_impl<5, Padding>(input, output, constant_value); break;
    case 7:  MedianFilter_scalar_impl<7, Padding>(input, output, constant_value); break;
    case 9:  MedianFilter_scalar_impl<9, Padding>(input, output, constant_value); break;
    case 11: MedianFilter_scalar_impl<11, Padding>(input, output, constant_value); break;
    default: assert(false && "Unsupported kernel size for median filter"); break;
    }
}

} // namespace

void ref::MedianFilter(const Image<uint8_t, ImageType::GRAY>& input,
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
