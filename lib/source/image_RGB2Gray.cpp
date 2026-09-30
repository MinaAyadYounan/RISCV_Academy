#include "image_RGB2Gray.hpp"
#include <riscv_vector.h>

void vec::RGB2Gray(const Image<uint8_t, ImageType::RGB>& input, 
                   Image<uint8_t, ImageType::GRAY>& output)
{
    // Standard RGB to Grayscale coefficients (ITU-R BT.601)
    constexpr float r_coeff = 0.299f;
    constexpr float g_coeff = 0.587f;
    constexpr float b_coeff = 0.114f;

    // Fixed-point Q0.8 coefficients (8 fractional bits, mult_factor = 256)
    constexpr int frac_bits = 8;
    constexpr int mult_factor = 1 << frac_bits;
    constexpr uint8_t r_coeff_q8 = static_cast<uint8_t>(mult_factor * r_coeff + 0.5f); // 77
    constexpr uint8_t g_coeff_q8 = static_cast<uint8_t>(mult_factor * g_coeff + 0.5f); // 150
    constexpr uint8_t b_coeff_q8 = static_cast<uint8_t>(mult_factor * b_coeff + 0.5f); // 29

    const uint8_t* __restrict__ in_ptr  = input.GetPtr(0, 0);
    uint8_t*       __restrict__ out_ptr = output.GetPtr(0, 0);

    const size_t total_pixels = input.Width() * input.Height();
    const uint8_t* const out_end = out_ptr + total_pixels;
    const size_t vlmax = __riscv_vsetvlmax_e8m2();

    // 2x Unrolled Main Loop (LMUL=2): processes 2 * vlmax (64 pixels) per iteration
    for (; out_ptr + 2 * vlmax <= out_end; )
    {
        // 1. Load two consecutive vector chunks (64 pixels = 192 bytes)
        const vuint8m2x3_t RGB_A = __riscv_vlseg3e8_v_u8m2x3(in_ptr, vlmax);
        const vuint8m2x3_t RGB_B = __riscv_vlseg3e8_v_u8m2x3(in_ptr + vlmax * 3, vlmax);

        const vuint8m2_t R_A = __riscv_vget_v_u8m2x3_u8m2(RGB_A, 0);
        const vuint8m2_t G_A = __riscv_vget_v_u8m2x3_u8m2(RGB_A, 1);
        const vuint8m2_t B_A = __riscv_vget_v_u8m2x3_u8m2(RGB_A, 2);

        const vuint8m2_t R_B = __riscv_vget_v_u8m2x3_u8m2(RGB_B, 0);
        const vuint8m2_t G_B = __riscv_vget_v_u8m2x3_u8m2(RGB_B, 1);
        const vuint8m2_t B_B = __riscv_vget_v_u8m2x3_u8m2(RGB_B, 2);

        // 2. Interleaved Multiply-Accumulate across independent ALU pipelines
        vuint16m4_t acc_A = __riscv_vwmulu_vx_u16m4(R_A, r_coeff_q8, vlmax);
        vuint16m4_t acc_B = __riscv_vwmulu_vx_u16m4(R_B, r_coeff_q8, vlmax);

        acc_A = __riscv_vwmaccu_vx_u16m4(acc_A, g_coeff_q8, G_A, vlmax);
        acc_B = __riscv_vwmaccu_vx_u16m4(acc_B, g_coeff_q8, G_B, vlmax);

        acc_A = __riscv_vwmaccu_vx_u16m4(acc_A, b_coeff_q8, B_A, vlmax);
        acc_B = __riscv_vwmaccu_vx_u16m4(acc_B, b_coeff_q8, B_B, vlmax);

        // 3. Narrowing shift right (fixed-point division by 256) and store
        __riscv_vse8_v_u8m2(out_ptr,         __riscv_vnsrl_wx_u8m2(acc_A, frac_bits, vlmax), vlmax);
        __riscv_vse8_v_u8m2(out_ptr + vlmax, __riscv_vnsrl_wx_u8m2(acc_B, frac_bits, vlmax), vlmax);

        in_ptr  += vlmax * 6;
        out_ptr += vlmax * 2;
    }

    // Tail Loop: handles any remaining pixels (< 2 * vlmax)
    size_t vl = 0;
    for (; out_ptr < out_end; out_ptr += vl, in_ptr += vl * 3)
    {
        vl = __riscv_vsetvl_e8m2(out_end - out_ptr);

        const vuint8m2x3_t RGB = __riscv_vlseg3e8_v_u8m2x3(in_ptr, vl);
        const vuint8m2_t R = __riscv_vget_v_u8m2x3_u8m2(RGB, 0);
        const vuint8m2_t G = __riscv_vget_v_u8m2x3_u8m2(RGB, 1);
        const vuint8m2_t B = __riscv_vget_v_u8m2x3_u8m2(RGB, 2);

        vuint16m4_t acc = __riscv_vwmulu_vx_u16m4(R, r_coeff_q8, vl);
        acc = __riscv_vwmaccu_vx_u16m4(acc, g_coeff_q8, G, vl);
        acc = __riscv_vwmaccu_vx_u16m4(acc, b_coeff_q8, B, vl);

        const vuint8m2_t gray = __riscv_vnsrl_wx_u8m2(acc, frac_bits, vl);
        __riscv_vse8_v_u8m2(out_ptr, gray, vl);
    }
}
