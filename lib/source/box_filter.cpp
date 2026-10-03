#include "box_filter.hpp"
#include <riscv_vector.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

// Normalization factor helper in Q16 fixed point:
constexpr int frac_bits = 16;
constexpr uint32_t mult_factor = 1u << frac_bits;
constexpr uint16_t NORM_MUL_9 = static_cast<uint16_t>(mult_factor * (1.0f / 9.0f) + 0.5f);

// =============================================================================
// Helper: Load a row safely with padding (Zero runtime branch with constexpr)
// =============================================================================
template <PaddingType Padding>
static inline vuint8m2_t LoadRowVector(const Image<uint8_t, ImageType::GRAY>& input,
                                       int y, size_t j, size_t vl, uint8_t constant_value = 0)
{
    if constexpr (Padding == PaddingType::CONSTANT) {
        if (y < 0 || y >= static_cast<int>(input.Height())) {
            return __riscv_vmv_v_x_u8m2(constant_value, vl);
        }
        return __riscv_vle8_v_u8m2(input.GetRowPtr(y) + j, vl);
    } else {
        const uint8_t* ptr = input.GetRowPtr<Padding>(y);
        return __riscv_vle8_v_u8m2(ptr + j, vl);
    }
}

// =============================================================================
// 1. Vertical Computation Initial Load (Warm-up for Row 0 across all Kernel Sizes)
// =============================================================================
template <PaddingType Padding, size_t KernelSize = 3>
static inline void vertical_computation_initial_load(uint16_t* res,
                                                     const Image<uint8_t, ImageType::GRAY>& input,
                                                     size_t width, size_t vlmax,
                                                     uint8_t constant_value = 0)
{
    constexpr int padding = static_cast<int>(KernelSize / 2);
    size_t j = 0;
    for (; j + vlmax <= width; j += vlmax) {
        // First row of window: y = -padding
        const vuint8m2_t v_init = LoadRowVector<Padding>(input, -padding, j, vlmax, constant_value);
        vuint16m4_t sum = __riscv_vwaddu_vx_u16m4(v_init, 0, vlmax); // widen into 16-bit

        // Accumulate remaining rows in window: y = -padding + 1 .. +padding
        for (int m = -padding + 1; m <= padding; ++m) {
            const vuint8m2_t v_row = LoadRowVector<Padding>(input, m, j, vlmax, constant_value);
            sum = __riscv_vwaddu_wv_u16m4(sum, v_row, vlmax);
        }
        __riscv_vse16_v_u16m4(res + j, sum, vlmax);
    }

    // Tail handling
    if (j < width) {
        const size_t vl = __riscv_vsetvl_e8m2(width - j);
        const vuint8m2_t v_init = LoadRowVector<Padding>(input, -padding, j, vl, constant_value);
        vuint16m4_t sum_tail = __riscv_vwaddu_vx_u16m4(v_init, 0, vl);

        for (int m = -padding + 1; m <= padding; ++m) {
            const vuint8m2_t v_row = LoadRowVector<Padding>(input, m, j, vl, constant_value);
            sum_tail = __riscv_vwaddu_wv_u16m4(sum_tail, v_row, vl);
        }
        __riscv_vse16_v_u16m4(res + j, sum_tail, vl);
    }
}

// =============================================================================
// 2. Vertical Computation (Sliding update: res[j] = res[j] - v_exit + v_enter)
// =============================================================================
template <PaddingType Padding>
static inline void vertical_computation(uint16_t* res,
                                        const Image<uint8_t, ImageType::GRAY>& input,
                                        int y_exit, int y_enter,
                                        size_t width, size_t vlmax,
                                        uint8_t constant_value = 0)
{
    const uint8_t* ptr_exit  = nullptr;
    const uint8_t* ptr_enter = nullptr;

    if constexpr (Padding == PaddingType::CONSTANT) {
        const int h = static_cast<int>(input.Height());
        if (y_exit >= 0 && y_exit < h)   ptr_exit  = input.GetRowPtr(y_exit);
        if (y_enter >= 0 && y_enter < h) ptr_enter = input.GetRowPtr(y_enter);
    } else {
        ptr_exit  = input.GetRowPtr<Padding>(y_exit);
        ptr_enter = input.GetRowPtr<Padding>(y_enter);
    }

    size_t j = 0;
    const size_t vl2 = vlmax * 2;
    for (; j + vl2 <= width; j += vl2) {
        // unrolling the loop by two to hide latency of dependency and avoid stall 
        const vuint16m4_t sum0     = __riscv_vle16_v_u16m4(res + j, vlmax);
        const vuint16m4_t sum1     = __riscv_vle16_v_u16m4(res + j + vlmax, vlmax);

        const vuint8m2_t  v_exit0  = ptr_exit ? __riscv_vle8_v_u8m2(ptr_exit + j, vlmax) : __riscv_vmv_v_x_u8m2(constant_value, vlmax);
        const vuint8m2_t  v_exit1  = ptr_exit ? __riscv_vle8_v_u8m2(ptr_exit + j + vlmax, vlmax) : __riscv_vmv_v_x_u8m2(constant_value, vlmax);

        const vuint16m4_t sub0     = __riscv_vwsubu_wv_u16m4(sum0, v_exit0, vlmax);
        const vuint16m4_t sub1     = __riscv_vwsubu_wv_u16m4(sum1, v_exit1, vlmax);

        const vuint8m2_t  v_enter0 = ptr_enter ? __riscv_vle8_v_u8m2(ptr_enter + j, vlmax) : __riscv_vmv_v_x_u8m2(constant_value, vlmax);
        const vuint8m2_t  v_enter1 = ptr_enter ? __riscv_vle8_v_u8m2(ptr_enter + j + vlmax, vlmax) : __riscv_vmv_v_x_u8m2(constant_value, vlmax);

        const vuint16m4_t sum01_0  = __riscv_vwaddu_wv_u16m4(sub0, v_enter0, vlmax);
        const vuint16m4_t sum01_1  = __riscv_vwaddu_wv_u16m4(sub1, v_enter1, vlmax);

        __riscv_vse16_v_u16m4(res + j, sum01_0, vlmax);
        __riscv_vse16_v_u16m4(res + j + vlmax, sum01_1, vlmax);
    }

    for (; j + vlmax <= width; j += vlmax) {
        const vuint16m4_t sum     = __riscv_vle16_v_u16m4(res + j, vlmax);
        const vuint8m2_t  v_exit  = ptr_exit ? __riscv_vle8_v_u8m2(ptr_exit + j, vlmax) : __riscv_vmv_v_x_u8m2(constant_value, vlmax);
        const vuint16m4_t sub     = __riscv_vwsubu_wv_u16m4(sum, v_exit, vlmax);

        const vuint8m2_t  v_enter = ptr_enter ? __riscv_vle8_v_u8m2(ptr_enter + j, vlmax) : __riscv_vmv_v_x_u8m2(constant_value, vlmax);
        const vuint16m4_t sum01   = __riscv_vwaddu_wv_u16m4(sub, v_enter, vlmax);
        __riscv_vse16_v_u16m4(res + j, sum01, vlmax);
    }
    // Tail handling
    if (j < width) {
        const size_t vl = __riscv_vsetvl_e8m2(width - j);
        const vuint16m4_t sum     = __riscv_vle16_v_u16m4(res + j, vl);
        const vuint8m2_t  v_exit  = ptr_exit ? __riscv_vle8_v_u8m2(ptr_exit + j, vl) : __riscv_vmv_v_x_u8m2(constant_value, vl);
        const vuint16m4_t sub     = __riscv_vwsubu_wv_u16m4(sum, v_exit, vl);

        const vuint8m2_t  v_enter = ptr_enter ? __riscv_vle8_v_u8m2(ptr_enter + j, vl) : __riscv_vmv_v_x_u8m2(constant_value, vl);
        const vuint16m4_t sum01   = __riscv_vwaddu_wv_u16m4(sub, v_enter, vl);
        __riscv_vse16_v_u16m4(res + j, sum01, vl);
    }
}

// =============================================================================
// 3. Horizontal Computation
// =============================================================================
template <size_t KernelSize>
static inline vuint16m4_t horizontal_computation(const uint16_t* buff, size_t j, size_t vl)
{
    constexpr int R = static_cast<int>(KernelSize / 2);
    vuint16m4_t sum = __riscv_vle16_v_u16m4(buff + j - R, vl);
    for (int k = -R + 1; k <= R; ++k) {
        const vuint16m4_t val = __riscv_vle16_v_u16m4(buff + j + k, vl);
        sum = __riscv_vadd_vv_u16m4(sum, val, vl);
    }
    return sum;
}

template <uint16_t NormMul>
static inline void horizontal_normalize_and_store(uint8_t* out_row, vuint16m4_t sum, size_t j, size_t vl)
{
    const vuint16m4_t norm_16 = __riscv_vmulhu_vx_u16m4(sum, NormMul, vl);
    const vuint8m2_t  norm_8  = __riscv_vncvt_x_x_w_u8m2(norm_16, vl);
    __riscv_vse8_v_u8m2(out_row + j, norm_8, vl);
}

// =============================================================================
// 4. Box Filter Loop 
// =============================================================================

template <PaddingType Padding, size_t KernelSize = 3>
static void BoxFilter_loop(const Image<uint8_t, ImageType::GRAY>& input,
                           Image<uint8_t, ImageType::GRAY>& output, 
                           uint8_t constant_value)
{
    const size_t width  = input.Width();
    const size_t height = input.Height();

    const size_t vlmax = __riscv_vsetvlmax_e8m2();

    // Support padding up to 9x9 (Radius = 4)
    constexpr size_t MAX_RADIUS = 4;
    constexpr size_t MAX_WIDTH = 2048;
    uint16_t buff_pad[MAX_WIDTH + 2 * MAX_RADIUS];
    uint16_t* buff = buff_pad + MAX_RADIUS; // buff[0] is at buff_pad[MAX_RADIUS]

    constexpr int padding = static_cast<int>(KernelSize / 2); 

    auto set_horizontal_padding = [&]() {
        if constexpr (Padding == PaddingType::CONSTANT) {
            for (int i = 0; i < padding; ++i) {
                buff[-1 - i]    = KernelSize * constant_value;
                buff[width + i] = KernelSize * constant_value;
            }
        } else if constexpr (Padding == PaddingType::REPLICA) {
            for (int i = 0; i < padding; ++i) {
                buff[-1 - i]    = buff[0];
                buff[width + i] = buff[width - 1];
            }
        } else if constexpr (Padding == PaddingType::REFLECT) {
            const bool width_gt_1 = (width > 1);
            for (int i = 0; i < padding; ++i) {
                buff[-1 - i]    = width_gt_1 ? buff[i + 1] : buff[0];
                buff[width + i] = width_gt_1 ? buff[width - 2 - i] : buff[width - 1];
            }
        }
    };

    constexpr uint16_t NORM_MUL = static_cast<uint16_t>(mult_factor * (1.0f / (KernelSize * KernelSize)) + 0.5f);

    auto compute_horizontal_row = [&](uint8_t* out_row) {
        // unrolling horizontal computation loop by two . 
        size_t j = 0;
        const size_t vl2 = vlmax * 2;
        for (; j + vl2 <= width; j += vl2) {
            vuint16m4_t hsum0 = horizontal_computation<KernelSize>(buff, j, vlmax);
            vuint16m4_t hsum1 = horizontal_computation<KernelSize>(buff, j + vlmax, vlmax);
            horizontal_normalize_and_store<NORM_MUL>(out_row, hsum0, j, vlmax);
            horizontal_normalize_and_store<NORM_MUL>(out_row, hsum1, j + vlmax, vlmax);
        }
        for (; j + vlmax <= width; j += vlmax) {
            vuint16m4_t hsum = horizontal_computation<KernelSize>(buff, j, vlmax);
            horizontal_normalize_and_store<NORM_MUL>(out_row, hsum, j, vlmax);
        }
        if (j < width) {
            const size_t vl = __riscv_vsetvl_e8m2(width - j);
            vuint16m4_t hsum = horizontal_computation<KernelSize>(buff, j, vl);
            horizontal_normalize_and_store<NORM_MUL>(out_row, hsum, j, vl);
        }
    };

    // 1. Initial warm-up load for Row 0
    vertical_computation_initial_load<Padding, KernelSize>(buff, input, width, vlmax, constant_value);
    set_horizontal_padding();

    // 2. Horizontal computation for Row 0
    compute_horizontal_row(output.GetPtr(0, 0));

    // 3. Sliding window across subsequent rows: y = 1 to height - 1
    for (size_t y = 1; y < height; ++y) {
        const int y_exit  = static_cast<int>(y) - padding - 1;
        const int y_enter = static_cast<int>(y) + padding;

        vertical_computation<Padding>(buff, input, y_exit, y_enter, width, vlmax, constant_value);
        if constexpr (Padding != PaddingType::CONSTANT) {
            set_horizontal_padding();
        }

        compute_horizontal_row(output.GetPtr(0, y));
    }
}

// =============================================================================
// 5. Box Filter Entry Point
// =============================================================================

void vec::BoxFilter(const Image<uint8_t, ImageType::GRAY>& input,
                    Image<uint8_t, ImageType::GRAY>& output,
                    int kernel_size,
                    PaddingType padding_type,
                    uint8_t constant_value)
{
    assert(input.Width() == output.Width() && input.Height() == output.Height());
    switch (kernel_size)
    {
    case 3:
        switch (padding_type)
        {
        case PaddingType::CONSTANT: BoxFilter_loop<PaddingType::CONSTANT, 3>(input, output, constant_value); break;
        case PaddingType::REPLICA:  BoxFilter_loop<PaddingType::REPLICA, 3>(input, output, constant_value); break;
        case PaddingType::REFLECT:  BoxFilter_loop<PaddingType::REFLECT, 3>(input, output, constant_value); break;
        }
        break;
    case 5:
        switch (padding_type)
        {
        case PaddingType::CONSTANT: BoxFilter_loop<PaddingType::CONSTANT, 5>(input, output, constant_value); break;
        case PaddingType::REPLICA:  BoxFilter_loop<PaddingType::REPLICA, 5>(input, output, constant_value); break;
        case PaddingType::REFLECT:  BoxFilter_loop<PaddingType::REFLECT, 5>(input, output, constant_value); break;
        }
        break;
    case 7:
        switch (padding_type)
        {
        case PaddingType::CONSTANT: BoxFilter_loop<PaddingType::CONSTANT, 7>(input, output, constant_value); break;
        case PaddingType::REPLICA:  BoxFilter_loop<PaddingType::REPLICA, 7>(input, output, constant_value); break;
        case PaddingType::REFLECT:  BoxFilter_loop<PaddingType::REFLECT, 7>(input, output, constant_value); break;
        }
        break;
    case 9:
        switch (padding_type)
        {
        case PaddingType::CONSTANT: BoxFilter_loop<PaddingType::CONSTANT, 9>(input, output, constant_value); break;
        case PaddingType::REPLICA:  BoxFilter_loop<PaddingType::REPLICA, 9>(input, output, constant_value); break;
        case PaddingType::REFLECT:  BoxFilter_loop<PaddingType::REFLECT, 9>(input, output, constant_value); break;
        }
        break;
    default:
        assert(false && "Only kernel sizes 3, 5, 7, 9 are supported");
        break;
    }
}