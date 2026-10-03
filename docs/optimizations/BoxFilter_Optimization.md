# RISC-V Vector Box Filter Optimization

**Target Architecture:** XiangShan Out-of-Order Core (KMHV3 Model in gem5)  
**Algorithm:** 2D Separable Sliding-Window Box Filter with Q16 Fixed-Point Normalization

---

## 1. Benchmark Results (128x128 Image, 1 Iteration)

| Kernel Size | Padding Type | Scalar Cycles | Vector Cycles | Scalar Instrs | Vector Instrs | Scalar px/cycle | Vector px/cycle | **Speedup** | Correctness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$3\times 3$** | `REFLECT`  | 111,260 | **35,036** | 217,849 | 21,114 | 0.147 | **0.468** | **$3.18\times$** | **PASSED** |
| **$5\times 5$** | `REPLICA`  | 112,213 | **36,846** | 216,593 | 24,075 | 0.146 | **0.445** | **$3.05\times$** | **PASSED** |
| **$7\times 7$** | `CONSTANT` | 73,193  | **38,983** | 351,989 | 25,942 | 0.224 | **0.420** | **$1.88\times$** | **PASSED** |
| **$9\times 9$** | `REFLECT`  | 116,055 | **42,090** | 224,516 | 33,301 | 0.141 | **0.389** | **$2.76\times$** | **PASSED** |

> **Note on Padding:** Padding mode noticeably affects speedup. Modes like `REFLECT` favor vector due to higher scalar coordinate-calculation overhead, whereas simpler modes like `CONSTANT` are easier on scalar execution.

---

## 2. Scalar Optimization Stages

1. **Naive 2D Convolution:**  
   Direct 2D nested loops with hardware integer division (`/ K^2`).  
   - **Speedup compared to optimized vector:** **$\sim 26\times$** (scalar suffered from repeated 2D passes and 30-cycle hardware division stalls).

2. **1D Separable Convolution:**  
   Separated into 1D horizontal and 1D vertical passes with Q16 fixed-point math.  
   - **Speedup compared to optimized vector:** **$\sim 22\times$** for $3\times 3$, but exploded to **$\sim 150\times$** for $5\times 5$ and **$\sim 256\times$** for $9\times 9$ because vertical passes re-summed all rows repeatedly.

3. **2D Sliding Window:**  
   Applied sliding window both horizontally and vertically so scalar runs in true $O(1)$ time per pixel.  
   - **Speedup compared to optimized vector:** Normalized for fair comparision.

---

## 3. Vector Optimization Stages

1. **1D Separable Optimization:**  
   Implemented 1D separable convolution using RVV vector registers and fixed-point normalization (`vmulhu` + `vncvt`).

2. **Vertical Sliding Window:**  
   Vector slides row by row (subtracting exiting row, adding entering row). This kept vector cycle counts virtually flat across all kernel sizes.

3. **Vector Loop Unrolling:**  
   Unrolled vector loops by $2\times$ and hoisted row pointers out of chunk loops. This showed a noticeable improvement, reducing instructions by ~28% and achieving **$\sim 1.88\times - 3.18\times$** speedup over the optimized scalar sliding window.

---

## 4. Verification


- **Verification:** 100% bit-exact across all 120 unit tests (kernels 3, 5, 7, 9; image width 1, 15, 16, 31, 64; image height 13, 22; `CONSTANT`, `REPLICA`, `REFLECT` padding).
