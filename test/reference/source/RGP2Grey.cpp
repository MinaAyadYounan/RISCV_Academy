#include "RGP2Grey.hpp"

void ref::RGP2Grey(const Image<uint8_t, ImageType::RGB> &input1, 
                          Image<uint8_t,ImageType::GRAY> &output)
{


    for (int y = 0; y < input1.Height(); y++)
    {
     
        for (int x = 0; x < input1.Width(); x++)
        {
            const uint16_t R = static_cast<uint16_t>(input1.GetPixel(x, y,0));
            const uint16_t G = static_cast<uint16_t>(input1.GetPixel(x, y,1));
            const uint16_t B = static_cast<uint16_t>(input1.GetPixel(x, y,2));
            
            uint16_t value = 77*R + 150*G + 29*B; // (0.299R + 0.587G + 0.114B)*2^8 using: Q8.8
            value = value >> 8;

            output.SetPixel(x, y, static_cast<uint8_t>(value));
        }
    }
}
