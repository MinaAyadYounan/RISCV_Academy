#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

template <int kernel_size, PaddingType padding_type>
int box_filter_benchmark()
{
    const int width = 128;
    const int height = 128;
    const int loop_count = 1;

    printf("Box Filter benchmark: %dx%d (K=%d), %d iterations\n", width, height, kernel_size, loop_count);

    Image<uint8_t, ImageType::GRAY> input0(width, height);
    Image<uint8_t, ImageType::GRAY> output_reference(width, height);
    Image<uint8_t, ImageType::GRAY> output_vectorized(width, height);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);
    bool all_correct = true;

    uint64_t reference_cycles = 0, vectorized_cycles = 0;
    uint64_t reference_instrs = 0, vectorized_instrs = 0;

    for (int i = 0; i < loop_count; ++i)
    {
        Timer timer_scalar, timer_vectorized;

        timer_scalar.Start();
        ref::BoxFilter(input0, output_reference, kernel_size, padding_type);
        timer_scalar.Stop();

        timer_vectorized.Start();
        vec::BoxFilter(input0, output_vectorized, kernel_size, padding_type);
        timer_vectorized.Stop();

        reference_cycles += timer_scalar.ElapsedCycles();
        reference_instrs += timer_scalar.ElapsedInstructions();

        vectorized_cycles += timer_vectorized.ElapsedCycles();
        vectorized_instrs += timer_vectorized.ElapsedInstructions();

        if (!CheckCorrectness(output_reference, output_vectorized))
        {
            all_correct = false;
            break;
        }
    }

    char name[32];
    snprintf(name, sizeof(name), "BoxFilter %dx%d", kernel_size, kernel_size);
    PrintTime(name, width * height,
              vectorized_cycles / loop_count, vectorized_instrs / loop_count,
              reference_cycles / loop_count, reference_instrs / loop_count);

    if (all_correct)
    {
        printf("Output is correct.\n\n");
        return 0;
    }
    else
    {
        printf("Output is wrong!\n\n");
        return 1;
    }
}

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    int ret = 0;
    ret |= box_filter_benchmark<3, PaddingType::REFLECT>();
    ret |= box_filter_benchmark<5, PaddingType::REPLICA>();
    ret |= box_filter_benchmark<7, PaddingType::CONSTANT>();
    ret |= box_filter_benchmark<9, PaddingType::REFLECT>();
    return ret;
}