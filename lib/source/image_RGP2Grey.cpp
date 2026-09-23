
#include "image_RGP2Grey.hpp"
#include <riscv_vector.h>

static void RGP2Grey_loop(const uint8_t* in0, 
                           uint8_t* out,
                     const size_t width, 
                     const size_t height)
{
    size_t vl = 0;
    uint8_t  R_val = 77 ;
    uint8_t G_val = 150 ;
    uint8_t B_val = 29 ;
    for (size_t y = 0; y < height; ++y)
    {   
        const uint8_t* row0 = &in0[y * width*3];
        uint8_t* row_out = &out[y * width];

        for (size_t x = 0; x < width; x += vl)
        {
            vl = __riscv_vsetvl_e8m4(width - x);
            const vuint8m4_t R = __riscv_vlse8_v_u8m4(&row0[3 * x],     3, vl);
            const vuint8m4_t G = __riscv_vlse8_v_u8m4(&row0[3 * x + 1], 3, vl);
            const vuint8m4_t B = __riscv_vlse8_v_u8m4(&row0[3 * x + 2], 3, vl);
                                     
            const vuint16m8_t R_weighted  = __riscv_vwmulu_vx_u16m8(R, R_val, vl);
            const vuint16m8_t G_weighted  = __riscv_vwmulu_vx_u16m8(G, G_val, vl);
            const vuint16m8_t B_weighted  = __riscv_vwmulu_vx_u16m8(B, B_val, vl);
                       
            const vuint16m8_t RG_sum      = __riscv_vadd_vv_u16m8(R_weighted, G_weighted, vl);
            const vuint16m8_t RGP_avg     = __riscv_vadd_vv_u16m8(RG_sum, B_weighted, vl);
            const  vuint8m4_t Grey_result =	__riscv_vnsrl_wx_u8m4(RGP_avg, 8, vl);
            __riscv_vse8_v_u8m4(&row_out[x], Grey_result, vl); 
                    
        }
    }
}


void vec::RGP2Grey(const Image<uint8_t, ImageType::RGB>& input, 
                Image<uint8_t, ImageType::GRAY>& output)
{
    const uint8_t* in0 = input.GetPtr(0, 0);
    uint8_t* out = output.GetPtr(0, 0);
    const size_t width = input.Width();
    const size_t height = input.Height();
    RGP2Grey_loop(in0, out, width, height);
   
}

