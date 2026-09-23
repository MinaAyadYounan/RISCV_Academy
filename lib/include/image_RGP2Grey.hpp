#pragma once
#include "types.hpp"

namespace vec 
{
  void RGP2Grey(const Image<uint8_t, ImageType::RGB>& input, 
                Image<uint8_t, ImageType::GRAY>& output);
                
}
            