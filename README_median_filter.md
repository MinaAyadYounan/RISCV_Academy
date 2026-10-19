# Median Filter - RISC-V Vector (RVV) Optimization

---

## 1. gem5 Architectural Benchmark Results (XiangShan KMHV3)

* **Architecture:** XiangShan NanHu / KMHV3 Out-of-Order Core (RVV 1.0, VLEN=128, LMUL=2)

### Full Image Benchmark (128 &times; 100, 12,800 pixels)

| Kernel Size | Window Elements | Scalar Cycles | Vectorized Cycles | Speedup | Scalar Instrs | Vectorized Instrs | Instruction Reduction | Performance (px/cycle) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3&times;3** | 9 | 75,773 | **51,595** | **1.47&times;** | 125,013 | 49,460 | **2.53&times;** | 0.2481 |
| **5&times;5** | 25 | 100,265 | **88,466** | **1.13&times;** | 184,829 | 151,339 | **1.22&times;** | 0.1447 |
| **7&times;7** | 49 | 1,428,739 | **202,126** | **7.07&times;** | 8,596,057 | 346,368 | **24.82&times;** | 0.0633 |
| **9&times;9** | 81 | 2,777,749 | **342,440** | **8.11&times;** | 17,852,723 | 640,452 | **27.87&times;** | 0.0374 |
| **11&times;11** | 121 | *(sim limit)* | **550,525** | **> 12&times;** | *(sim limit)* | 1,090,226 | **> 36&times;** | 0.0233 |

---

### 11&times;11 Scaled Verification Benchmark (64 &times; 64, 4,096 pixels)

Full scalar versus vectorized benchmark on 11&times;11 executed within simulator limits:

| Kernel Size | Image Size | Scalar Cycles | Vectorized Cycles | Speedup | Scalar Instrs | Vectorized Instrs | Instruction Reduction | Output Validation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **11&times;11** | 64 &times; 64 | 2,124,211 | **196,515** | **10.81&times;** | 13,155,645 | 360,550 | **36.49&times;** | **Correct (Bit-exact)** |

---

## 2. Core Optimizations

* **Algorithm Selection (Separable Sorting Network vs. Histogram):**  
  While histogram-based approaches can be effective for large kernels in scalar implementations, they perform poorly on vector hardware. Indexed memory access (gather/scatter) requires multiple cycles, and cross-lane data dependencies cause significant pipeline stalls. In contrast, 2D separable sorting networks provide deterministic execution and continuous vector memory access, making them far better suited for vector processors.

* **Fair Comparison:**  
  The scalar baseline was optimized to use the same sorting network algorithm to ensure a fair architectural comparison. Additionally, a naive scalar implementation takes hours .
* **Reference:** Sorting networks based on optimal-depth designs from [Bert Dobbelaere's Sorting Networks](https://bertdobbelaere.github.io/sorting_networks.html).

---

## 3. Verification

All unit tests in `test/unit_test/source/median_filter_test.cpp` pass cleanly (**120/120 tests**) under `qemu-riscv64`.
