#include "image_RGB2Grey.hpp"

void ref::RGB2Grey(const Image<uint8_t, ImageType::RGB> &input1, 
                         Image<uint8_t, ImageType::GRAY> &output)
{

    constexpr float r_coeff = 0.299f;
    constexpr float g_coeff = 0.587f;
    constexpr float b_coeff = 0.114f;
    
    // Fixed-point Q0.8 coefficients (8 fractional bits).
    constexpr int frac_bits = 8;
    constexpr int mult_factor = 1 << frac_bits;
    constexpr uint16_t r_coeff_q8 = static_cast<uint16_t>(mult_factor * r_coeff);
    constexpr uint16_t g_coeff_q8 = static_cast<uint16_t>(mult_factor * g_coeff);
    constexpr uint16_t b_coeff_q8 = static_cast<uint16_t>(mult_factor * b_coeff);
    for (int y = 0; y < input1.Height(); y++)
    {
        for (int x = 0; x < input1.Width(); x++)
        {

            const uint16_t R = static_cast<uint16_t>(input1.GetPixel(x, y, 0));
            const uint16_t G = static_cast<uint16_t>(input1.GetPixel(x, y, 1));
            const uint16_t B = static_cast<uint16_t>(input1.GetPixel(x, y, 2));
            uint16_t value = r_coeff_q8 * R + g_coeff_q8 * G + b_coeff_q8 * B; // (0.299R + 0.587G + 0.114B)*2^8 using: Q8.8
            value = value >> 8;

            output.SetPixel(x, y, static_cast<uint8_t>(value));
        }
    }
}
