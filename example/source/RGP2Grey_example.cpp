#include "riscv_cv.hpp"
#include "reference_cv.hpp"

#ifndef RISCV_QEMU
#include "image0.hpp"
#include "image1.hpp"
#endif

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    // Read two input images
    Image<uint8_t, ImageType::RGB> image0;

#ifdef RISCV_QEMU
    image0.Read("/home/mina/RISCV_Academy/images/sample_640×426.ppm");
#else
    image0.Read(image0_width, image0_height, image0_data);
#endif

    // Add with saturation
    Image<uint8_t> out(image0.Width(), image0.Height());

    Timer timer;

    timer.Start();
    ref::RGP2Grey(image0, out);
    timer.Stop();

    printf("Time: %llu cycles, %llu instructions\n",
           (unsigned long long)timer.ElapsedCycles(),
           (unsigned long long)timer.ElapsedInstructions());

    // Write output
#ifdef RISCV_QEMU
    out.Write("/home/mina/RISCV_Academy/images/output_grey.pgm");
    printf("Wrote output.pgm (%dx%d)\n", out.Width(), out.Height());
#endif

    printf("Example passed.\n");
    return 0;
}
