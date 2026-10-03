#include "riscv_cv.hpp"

#ifndef RISCV_QEMU
#include "image0.hpp"
#endif

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    Image<uint8_t, ImageType::GRAY> image0;

#ifdef RISCV_QEMU
    image0.Read("/home/mina/RISCV_Academy/images/sample_640×426.pgm");
#else
    image0.Read(image0_width, image0_height, image0_data);
#endif

    Image<uint8_t, ImageType::GRAY> out(image0.Width(), image0.Height());

    Timer timer;

    timer.Start();
    vec::BoxFilter(image0, out, 7, PaddingType::REFLECT);
    timer.Stop();

    printf("Vector BoxFilter 7x7: %llu cycles, %llu instructions\n",
           (unsigned long long)timer.ElapsedCycles(),
           (unsigned long long)timer.ElapsedInstructions());

#ifdef RISCV_QEMU
    out.Write("/home/mina/RISCV_Academy/images/output_boxfilter_vec.pgm");
    printf("Wrote output_boxfilter_vec.pgm (%dx%d)\n", out.Width(), out.Height());
#endif

    printf("Example passed.\n");
    return 0;
}
