#include "image_RGB2Gray.hpp"

void ref::RGB2Gray(const Image<uint8_t, ImageType::RGB>& input, 
                   Image<uint8_t, ImageType::GRAY>& output)
{
    // Standard RGB to Grayscale coefficients (ITU-R BT.601)
    constexpr float r_coeff = 0.299f;
    constexpr float g_coeff = 0.587f;
    constexpr float b_coeff = 0.114f;

    // Fixed-point Q0.8 coefficients (8 fractional bits)
    constexpr int frac_bits = 8;
    constexpr int mult_factor = 1 << frac_bits;
    constexpr uint16_t r_coeff_q8 = static_cast<uint16_t>(mult_factor * r_coeff + 0.5f);
    constexpr uint16_t g_coeff_q8 = static_cast<uint16_t>(mult_factor * g_coeff + 0.5f);
    constexpr uint16_t b_coeff_q8 = static_cast<uint16_t>(mult_factor * b_coeff + 0.5f);

    for (int y = 0; y < input.Height(); y++)
    {
        for (int x = 0; x < input.Width(); x++)
        {
            const uint16_t R = static_cast<uint16_t>(input.GetPixel(x, y, 0));
            const uint16_t G = static_cast<uint16_t>(input.GetPixel(x, y, 1));
            const uint16_t B = static_cast<uint16_t>(input.GetPixel(x, y, 2));

            // Fixed-point multiply-accumulate and right-shift
            uint16_t value = r_coeff_q8 * R + g_coeff_q8 * G + b_coeff_q8 * B;
            value = value >> frac_bits;

            output.SetPixel(x, y, static_cast<uint8_t>(value));
        }
    }
}
