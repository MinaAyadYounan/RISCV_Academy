#include "gtest_lite.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include <tuple>

using BoxFilterParam = std::tuple<int, int, int, PaddingType>;

class BoxFilterTest : public ::testing::TestWithParam<BoxFilterParam>
{
};

TEST_P(BoxFilterTest, VectorizedMatchesReference)
{
    const auto& param = GetParam();
    const int width = std::get<0>(param);
    const int height = std::get<1>(param);
    const int kernel_size = std::get<2>(param);
    const PaddingType padding = std::get<3>(param);

    printf("[ width: %17d ]\n", width);
    printf("[ height: %16d ]\n", height);
    printf("[ kernel: %16d ]\n", kernel_size);
    printf("[ pad: %19s ]\n",
           padding == PaddingType::CONSTANT ? "CONSTANT" :
           padding == PaddingType::REPLICA  ? "REPLICA"  : "REFLECT");

    Image<uint8_t, ImageType::GRAY> input(width, height);
    Image<uint8_t, ImageType::GRAY> output_ref(width, height);
    Image<uint8_t, ImageType::GRAY> output_vec(width, height);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input);

    const uint8_t constant_val = 0;

    ref::BoxFilter(input, output_ref, kernel_size, padding, constant_val);
    vec::BoxFilter(input, output_vec, kernel_size, padding, constant_val);

    for (int y = 0; y < height; ++y)
    {
        const uint8_t* ref_row = output_ref.GetPtr(0, y);
        const uint8_t* vec_row = output_vec.GetPtr(0, y);
        for (int x = 0; x < width; ++x)
        {
            ASSERT_EQ(ref_row[x], vec_row[x]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ImageSizesAndKernels, BoxFilterTest,
    ::testing::Combine(
        ::testing::Values(
            1,          // edge case: width = 1
            15,         // less than 1 vector (< 16)
            16,         // exact 1 vector
            31,         // 1 vector + tail
            64          // multiple exact vectors
        ),
        ::testing::Values(
            13,         // odd height
            22          // even height
        ),
        ::testing::Values(3, 5, 7, 9), // 3x3, 5x5, 7x7, 9x9
        ::testing::Values(
            PaddingType::CONSTANT,
            PaddingType::REPLICA,
            PaddingType::REFLECT
        )
    )
);
