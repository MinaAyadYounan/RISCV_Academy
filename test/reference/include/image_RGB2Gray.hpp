#pragma once
#include "types.hpp"

namespace ref 
{
    void RGB2Gray(const Image<uint8_t, ImageType::RGB>& input, 
                  Image<uint8_t, ImageType::GRAY>& output);
}
