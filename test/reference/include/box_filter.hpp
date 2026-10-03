#pragma once

#include "types.hpp"

namespace ref
{
    void BoxFilter(const Image<uint8_t, ImageType::GRAY>& input,
                         Image<uint8_t, ImageType::GRAY>& output,
                         int kernel_size = 3,
                         PaddingType padding_type = PaddingType::CONSTANT,
                         uint8_t constant_value = 0);

    template <PaddingType Padding>
    void BoxFilter(const Image<uint8_t, ImageType::GRAY>& input,
                         Image<uint8_t, ImageType::GRAY>& output,
                         int kernel_size = 3,
                         uint8_t constant_value = 0)
    {
        BoxFilter(input, output, kernel_size, Padding, constant_value);
    }
}
