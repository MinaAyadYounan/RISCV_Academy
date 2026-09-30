#include "image_RGB2Grey.hpp"
#include <riscv_vector.h>

static void RGB2Grey_loop(const uint8_t* in0, 
                          uint8_t* out,
                          const size_t width, 
                          const size_t height)
{
    constexpr uint8_t R_val = 77;
    constexpr uint8_t G_val = 150;
    constexpr uint8_t B_val = 29;

    const size_t vlmax = __riscv_vsetvlmax_e8m2();

    for (size_t y = 0; y < height; ++y)
    {   
        const uint8_t* row0 = &in0[y * width * 3];
        uint8_t* row_out = &out[y * width];
        size_t x = 0;

        // 2x Unrolled loop
        for (; x + 2 * vlmax <= width; x += 2 * vlmax)
        {
            vuint8m2x3_t rgb0 = __riscv_vlseg3e8_v_u8m2x3(&row0[3 * x], vlmax);
            vuint8m2x3_t rgb1 = __riscv_vlseg3e8_v_u8m2x3(&row0[3 * (x + vlmax)], vlmax);

            vuint8m2_t r0 = __riscv_vget_v_u8m2x3_u8m2(rgb0, 0);
            vuint8m2_t g0 = __riscv_vget_v_u8m2x3_u8m2(rgb0, 1);
            vuint8m2_t b0 = __riscv_vget_v_u8m2x3_u8m2(rgb0, 2);

            vuint8m2_t r1 = __riscv_vget_v_u8m2x3_u8m2(rgb1, 0);
            vuint8m2_t g1 = __riscv_vget_v_u8m2x3_u8m2(rgb1, 1);
            vuint8m2_t b1 = __riscv_vget_v_u8m2x3_u8m2(rgb1, 2);

            vuint16m4_t sum0 = __riscv_vwmulu_vx_u16m4(r0, R_val, vlmax);
            vuint16m4_t sum1 = __riscv_vwmulu_vx_u16m4(r1, R_val, vlmax);

            sum0 = __riscv_vwmaccu_vx_u16m4(sum0, G_val, g0, vlmax);
            sum1 = __riscv_vwmaccu_vx_u16m4(sum1, G_val, g1, vlmax);

            sum0 = __riscv_vwmaccu_vx_u16m4(sum0, B_val, b0, vlmax);
            sum1 = __riscv_vwmaccu_vx_u16m4(sum1, B_val, b1, vlmax);

            vuint8m2_t out0 = __riscv_vnsrl_wx_u8m2(sum0, 8, vlmax);
            vuint8m2_t out1 = __riscv_vnsrl_wx_u8m2(sum1, 8, vlmax);

            __riscv_vse8_v_u8m2(&row_out[x], out0, vlmax);
            __riscv_vse8_v_u8m2(&row_out[x + vlmax], out1, vlmax);
        }

        // Tail loop
        size_t vl = 0;
        for (; x < width; x += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - x);
            vuint8m2x3_t rgb = __riscv_vlseg3e8_v_u8m2x3(&row0[3 * x], vl);
            vuint8m2_t r = __riscv_vget_v_u8m2x3_u8m2(rgb, 0);
            vuint8m2_t g = __riscv_vget_v_u8m2x3_u8m2(rgb, 1);
            vuint8m2_t b = __riscv_vget_v_u8m2x3_u8m2(rgb, 2);

            vuint16m4_t sum = __riscv_vwmulu_vx_u16m4(r, R_val, vl);
            sum = __riscv_vwmaccu_vx_u16m4(sum, G_val, g, vl);
            sum = __riscv_vwmaccu_vx_u16m4(sum, B_val, b, vl);

            vuint8m2_t out_val = __riscv_vnsrl_wx_u8m2(sum, 8, vl);
            __riscv_vse8_v_u8m2(&row_out[x], out_val, vl);
        }
    }
}

void vec::RGB2Grey(const Image<uint8_t, ImageType::RGB>& input, 
                   Image<uint8_t, ImageType::GRAY>& output)
{
    const uint8_t* in0 = input.GetPtr(0, 0);
    uint8_t* out = output.GetPtr(0, 0);
    const size_t width = input.Width();
    const size_t height = input.Height();
    RGB2Grey_loop(in0, out, width, height);
}
