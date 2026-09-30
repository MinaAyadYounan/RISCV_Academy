#include "gtest_lite.hpp"

#include <tuple>

#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"

class RGB2GrayTest : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(RGB2GrayTest, VectorizedMatchesReference)
{
    const auto [width, height] = GetParam();

    // print the test parameters
    printf("[ %-7s %16d ]\n", "width:", width);
    printf("[ %-7s %16d ]\n", "height:", height);

    Image<uint8_t, ImageType::RGB> input0(width, height);

    Image<uint8_t, ImageType::GRAY> output_reference(width, height);
    Image<uint8_t, ImageType::GRAY> output_vectorized(width, height);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);

    ref::RGB2Gray(input0, output_reference);
    vec::RGB2Gray(input0, output_vectorized);

    ExpectImagesEqual(output_reference, output_vectorized);
}

// In RGB2Gray: LMUL=2 (vlmax = 32 elements). Unrolled 2x = 64 elements.
// Testing edge cases: sub-vector (<32), around vlmax (31, 32, 33), 
// unroll boundary (63, 64, 65), and multiple vectors + tail (127, 128).
INSTANTIATE_TEST_SUITE_P(ImageSizes, RGB2GrayTest,
    ::testing::Combine(
        ::testing::Values(
            1, 7,       // width < vlmax (pure tail loop)
            16,         // half vector
            31, 32, 33, // around 1 vector chunk (LMUL=2)
            63, 64, 65, // around 2x unroll boundary (2 * vlmax)
            127, 128),  // multiple unrolled iterations + tail
        ::testing::Values(
            1,          // single row
            13,         // odd height
            22)         // even height
        )
    );
