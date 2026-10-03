# RISC-V Vector RGB to Gray Optimization

**Target Architecture:** XiangShan Out-of-Order Core (KMHV3 Model in gem5)  
**Algorithm:** Fixed-Point Q0.8 ITU-R BT.601 ($Y = (77 \cdot R + 150 \cdot G + 29 \cdot B) \gg 8$)

---

## 1. Benchmark Results (128x100 Image, 3 Iterations)

| Implementation | Memory Access | Vector Math | Cycles (gem5) | Instructions | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scalar Reference** | `LBU` / `SB` | 32-bit Integer Ops | 67,248 | 410,212 | **1.00×** |
| **Initial Vector** | Strided Loads (`vlse8`, stride 3) | `vwmulu` + `vadd` | 68,939 | 5,325 | **0.99×** |
| **Segment Load + MAC (No Unroll)** | Segment Load (`vlseg3e8`, LMUL=2) | Fused Widening MAC (`vwmaccu`) | 46,508 | 4,419 | **1.45×** |
| **Final Optimized (2× Unroll)** | 2× Unroll (`vlseg3e8`, LMUL=2) | Interleaved Dual-Stream MAC | **35,374** | **3,821** | **1.90×** |

---

## 2. Key Optimizations Applied

1. **Hardware Segment Loads (`vlseg3e8`, LMUL=2):**

   - De-interleaves R, G, and B color channels directly into three vector register groups in a single instruction.

2. **Fused Widening Multiply-Accumulate (`vwmaccu`):**
   - Computes $\text{acc} + (X \times C)$ in 16-bit precision without intermediate truncation or separate additions.

3. **2× Loop Unrolling & Latency Hiding:**
   - Interleaves two independent 32-pixel streams (Streams A and B) per iteration (processing 64 pixels = 192 bytes total).
   - Allows out-of-order execution pipelines to overlap memory latency and ALU computation across independent vector execution units.

4. **Constant-VL Main Loop & Tail Loop Separation:**
   - Uses static `vlmax` (`__riscv_vsetvlmax_e8m2()`) in the unrolled main loop, eliminating per-iteration `vsetvl` instruction overhead.
   - Remaining pixels ($< 2 \times vlmax$) are handled cleanly in the sub-vector tail loop.


