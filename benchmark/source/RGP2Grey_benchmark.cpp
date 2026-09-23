#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

int main()
{
#ifdef RISCV_BAREMETAL
    // Enable the vector unit before any auto-vectorized loop runs (nexus-am's
    // _start only enables the FPU).
    riscv_enable_vector();
#endif

    const int width = 128;
    const int height = 100;
    const int loop_count = 3;

    printf("RGP to Grey benchmark: %dx%d, %d iterations\n", width, height, loop_count);

    Image<uint8_t, ImageType::RGB>input0(width, height);

    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    // Deterministic fill so runs are reproducible.
    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);
    bool all_correct = true;

        uint64_t reference_cycles = 0, vectorized_cycles = 0;
        uint64_t reference_instrs = 0, vectorized_instrs = 0;

        for (int i = 0; i < loop_count; ++i)
        {
            Timer timer_scalar, timer_vectorized;

            timer_scalar.Start();
            ref::RGP2Grey(input0, output_reference);
            timer_scalar.Stop();

            timer_vectorized.Start();
            vec::RGP2Grey(input0,output_vectorized);
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
        PrintTime("RGP2Grey", width * height,
                  vectorized_cycles / loop_count, vectorized_instrs / loop_count,
                  reference_cycles / loop_count, reference_instrs / loop_count);
    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    else
    {
        printf("Output is wrong!\n");
        return 1;
    }
}