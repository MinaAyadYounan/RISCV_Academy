#pragma once

#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

// =============================================================================
// Configuration Constants
// =============================================================================
constexpr int MAX_WIDTH = 1024;
constexpr size_t MAX_VL = 32;


// =============================================================================
// Vector Compare-and-Swap Primitives (LMUL = 2)
// =============================================================================
static inline void VecCasAsc(vuint8m2_t &a, vuint8m2_t &b, size_t vl)
{
    vuint8m2_t mn = __riscv_vminu_vv_u8m2(a, b, vl);
    vuint8m2_t mx = __riscv_vmaxu_vv_u8m2(a, b, vl);
    a = mn; b = mx;
}

static inline void VecCasDesc(vuint8m2_t &a, vuint8m2_t &b, size_t vl)
{
    vuint8m2_t mn = __riscv_vminu_vv_u8m2(a, b, vl);
    vuint8m2_t mx = __riscv_vmaxu_vv_u8m2(a, b, vl);
    a = mx; b = mn;
}

// =============================================================================
// =============================================================================
// Optimal Vectorized Sorting Networks for Lines (Lengths 3 to 11)
// =============================================================================
// =============================================================================
static inline void SortLine3(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, size_t vl)
{
    VecCasAsc(x0, x1, vl); VecCasAsc(x1, x2, vl); VecCasAsc(x0, x1, vl);
}

static inline void SortLine4(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, size_t vl)
{
    VecCasAsc(x0, x1, vl); VecCasAsc(x2, x3, vl);
    VecCasAsc(x0, x2, vl); VecCasAsc(x1, x3, vl);
    VecCasAsc(x1, x2, vl);
}

static inline void SortLine5(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, vuint8m2_t& x4, size_t vl)
{
    VecCasAsc(x0, x1, vl); VecCasAsc(x3, x4, vl);
    VecCasAsc(x2, x4, vl); VecCasAsc(x2, x3, vl);
    VecCasAsc(x0, x3, vl); VecCasAsc(x0, x2, vl);
    VecCasAsc(x1, x4, vl); VecCasAsc(x1, x3, vl);
    VecCasAsc(x1, x2, vl);
}

static inline void SortLine6(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, vuint8m2_t& x4, vuint8m2_t& x5, size_t vl)
{
    VecCasAsc(x1, x2, vl); VecCasAsc(x0, x2, vl); VecCasAsc(x0, x1, vl);
    VecCasAsc(x4, x5, vl); VecCasAsc(x3, x5, vl); VecCasAsc(x3, x4, vl);
    VecCasAsc(x0, x3, vl); VecCasAsc(x1, x4, vl); VecCasAsc(x2, x5, vl);
    VecCasAsc(x2, x4, vl); VecCasAsc(x1, x3, vl);
    VecCasAsc(x2, x3, vl);
}

static inline void SortLine7(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3,
                             vuint8m2_t& x4, vuint8m2_t& x5, vuint8m2_t& x6, size_t vl)
{
    VecCasAsc(x0, x6, vl); VecCasAsc(x2, x3, vl); VecCasAsc(x4, x5, vl);
    VecCasAsc(x0, x2, vl); VecCasAsc(x1, x4, vl); VecCasAsc(x3, x6, vl);
    VecCasAsc(x0, x1, vl); VecCasAsc(x2, x5, vl); VecCasAsc(x3, x4, vl);
    VecCasAsc(x1, x2, vl); VecCasAsc(x4, x6, vl);
    VecCasAsc(x2, x3, vl); VecCasAsc(x4, x5, vl);
    VecCasAsc(x1, x2, vl); VecCasAsc(x3, x4, vl); VecCasAsc(x5, x6, vl);
}

// =============================================================================
static inline void SortLine8(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, vuint8m2_t& x4, vuint8m2_t& x5, vuint8m2_t& x6, vuint8m2_t& x7, size_t vl)
{
    VecCasAsc(x0, x2, vl);
    VecCasAsc(x1, x3, vl);
    VecCasAsc(x4, x6, vl);
    VecCasAsc(x5, x7, vl);
    VecCasAsc(x0, x4, vl);
    VecCasAsc(x1, x5, vl);
    VecCasAsc(x2, x6, vl);
    VecCasAsc(x3, x7, vl);
    VecCasAsc(x0, x1, vl);
    VecCasAsc(x2, x3, vl);
    VecCasAsc(x4, x5, vl);
    VecCasAsc(x6, x7, vl);
    VecCasAsc(x2, x4, vl);
    VecCasAsc(x3, x5, vl);
    VecCasAsc(x1, x4, vl);
    VecCasAsc(x3, x6, vl);
    VecCasAsc(x1, x2, vl);
    VecCasAsc(x3, x4, vl);
    VecCasAsc(x5, x6, vl);
}
static inline void SortLine9(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, vuint8m2_t& x4, vuint8m2_t& x5, vuint8m2_t& x6, vuint8m2_t& x7, vuint8m2_t& x8, size_t vl)
{
    VecCasAsc(x0, x3, vl);
    VecCasAsc(x1, x7, vl);
    VecCasAsc(x2, x5, vl);
    VecCasAsc(x4, x8, vl);
    VecCasAsc(x0, x7, vl);
    VecCasAsc(x2, x4, vl);
    VecCasAsc(x3, x8, vl);
    VecCasAsc(x5, x6, vl);
    VecCasAsc(x0, x2, vl);
    VecCasAsc(x1, x3, vl);
    VecCasAsc(x4, x5, vl);
    VecCasAsc(x7, x8, vl);
    VecCasAsc(x1, x4, vl);
    VecCasAsc(x3, x6, vl);
    VecCasAsc(x5, x7, vl);
    VecCasAsc(x0, x1, vl);
    VecCasAsc(x2, x4, vl);
    VecCasAsc(x3, x5, vl);
    VecCasAsc(x6, x8, vl);
    VecCasAsc(x2, x3, vl);
    VecCasAsc(x4, x5, vl);
    VecCasAsc(x6, x7, vl);
    VecCasAsc(x1, x2, vl);
    VecCasAsc(x3, x4, vl);
    VecCasAsc(x5, x6, vl);
}
static inline void SortLine10(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, vuint8m2_t& x4, vuint8m2_t& x5, vuint8m2_t& x6, vuint8m2_t& x7, vuint8m2_t& x8, vuint8m2_t& x9, size_t vl)
{
    VecCasAsc(x0, x8, vl);
    VecCasAsc(x1, x9, vl);
    VecCasAsc(x2, x7, vl);
    VecCasAsc(x3, x5, vl);
    VecCasAsc(x4, x6, vl);
    VecCasAsc(x0, x2, vl);
    VecCasAsc(x1, x4, vl);
    VecCasAsc(x5, x8, vl);
    VecCasAsc(x7, x9, vl);
    VecCasAsc(x0, x3, vl);
    VecCasAsc(x2, x4, vl);
    VecCasAsc(x5, x7, vl);
    VecCasAsc(x6, x9, vl);
    VecCasAsc(x0, x1, vl);
    VecCasAsc(x3, x6, vl);
    VecCasAsc(x8, x9, vl);
    VecCasAsc(x1, x5, vl);
    VecCasAsc(x2, x3, vl);
    VecCasAsc(x4, x8, vl);
    VecCasAsc(x6, x7, vl);
    VecCasAsc(x1, x2, vl);
    VecCasAsc(x3, x5, vl);
    VecCasAsc(x4, x6, vl);
    VecCasAsc(x7, x8, vl);
    VecCasAsc(x2, x3, vl);
    VecCasAsc(x4, x5, vl);
    VecCasAsc(x6, x7, vl);
    VecCasAsc(x3, x4, vl);
    VecCasAsc(x5, x6, vl);
}
static inline void SortLine11(vuint8m2_t& x0, vuint8m2_t& x1, vuint8m2_t& x2, vuint8m2_t& x3, vuint8m2_t& x4, vuint8m2_t& x5, vuint8m2_t& x6, vuint8m2_t& x7, vuint8m2_t& x8, vuint8m2_t& x9, vuint8m2_t& x10, size_t vl)
{
    VecCasAsc(x0, x1, vl);
    VecCasAsc(x2, x3, vl);
    VecCasAsc(x4, x5, vl);
    VecCasAsc(x6, x7, vl);
    VecCasAsc(x8, x9, vl);
    VecCasAsc(x0, x2, vl);
    VecCasAsc(x1, x3, vl);
    VecCasAsc(x4, x6, vl);
    VecCasAsc(x5, x7, vl);
    VecCasAsc(x8, x10, vl);
    VecCasAsc(x1, x2, vl);
    VecCasAsc(x5, x6, vl);
    VecCasAsc(x0, x4, vl);
    VecCasAsc(x1, x5, vl);
    VecCasAsc(x2, x6, vl);
    VecCasAsc(x3, x7, vl);
    VecCasAsc(x2, x4, vl);
    VecCasAsc(x3, x5, vl);
    VecCasAsc(x1, x2, vl);
    VecCasAsc(x3, x4, vl);
    VecCasAsc(x5, x6, vl);
    VecCasAsc(x9, x10, vl);
    VecCasAsc(x0, x8, vl);
    VecCasAsc(x1, x9, vl);
    VecCasAsc(x2, x10, vl);
    VecCasAsc(x4, x8, vl);
    VecCasAsc(x5, x9, vl);
    VecCasAsc(x6, x10, vl);
    VecCasAsc(x2, x4, vl);
    VecCasAsc(x3, x5, vl);
    VecCasAsc(x6, x8, vl);
    VecCasAsc(x7, x9, vl);
    VecCasAsc(x1, x2, vl);
    VecCasAsc(x3, x4, vl);
    VecCasAsc(x5, x6, vl);
    VecCasAsc(x7, x8, vl);
    VecCasAsc(x9, x10, vl);
}

// =============================================================================
// Phase 1: Vertical Column Sorting Across Padded Width
// =============================================================================
template <int K>
inline void SortColumnsVertically(const uint8_t* active_rows[K],
                                  uint8_t vert_rows[K][MAX_WIDTH + 64],
                                  int padded_width);

template <>
inline void SortColumnsVertically<3>(const uint8_t* active_rows[3], uint8_t vert_rows[3][MAX_WIDTH + 64], int padded_width)
{
    int c = 0;
    while (c < padded_width)
    {
        size_t vl = __riscv_vsetvl_e8m2(padded_width - c);
        vuint8m2_t r0 = __riscv_vle8_v_u8m2(active_rows[0] + c, vl);
        vuint8m2_t r1 = __riscv_vle8_v_u8m2(active_rows[1] + c, vl);
        vuint8m2_t r2 = __riscv_vle8_v_u8m2(active_rows[2] + c, vl);
        SortLine3(r0, r1, r2, vl);
        __riscv_vse8_v_u8m2(vert_rows[0] + c, r0, vl);
        __riscv_vse8_v_u8m2(vert_rows[1] + c, r1, vl);
        __riscv_vse8_v_u8m2(vert_rows[2] + c, r2, vl);
        c += vl;
    }
}

template <>
inline void SortColumnsVertically<5>(const uint8_t* active_rows[5], uint8_t vert_rows[5][MAX_WIDTH + 64], int padded_width)
{
    int c = 0;
    while (c < padded_width)
    {
        size_t vl = __riscv_vsetvl_e8m2(padded_width - c);
        vuint8m2_t r0 = __riscv_vle8_v_u8m2(active_rows[0] + c, vl);
        vuint8m2_t r1 = __riscv_vle8_v_u8m2(active_rows[1] + c, vl);
        vuint8m2_t r2 = __riscv_vle8_v_u8m2(active_rows[2] + c, vl);
        vuint8m2_t r3 = __riscv_vle8_v_u8m2(active_rows[3] + c, vl);
        vuint8m2_t r4 = __riscv_vle8_v_u8m2(active_rows[4] + c, vl);
        SortLine5(r0, r1, r2, r3, r4, vl);
        __riscv_vse8_v_u8m2(vert_rows[0] + c, r0, vl);
        __riscv_vse8_v_u8m2(vert_rows[1] + c, r1, vl);
        __riscv_vse8_v_u8m2(vert_rows[2] + c, r2, vl);
        __riscv_vse8_v_u8m2(vert_rows[3] + c, r3, vl);
        __riscv_vse8_v_u8m2(vert_rows[4] + c, r4, vl);
        c += vl;
    }
}

template <>
inline void SortColumnsVertically<7>(const uint8_t* active_rows[7], uint8_t vert_rows[7][MAX_WIDTH + 64], int padded_width)
{
    int c = 0;
    while (c < padded_width)
    {
        size_t vl = __riscv_vsetvl_e8m2(padded_width - c);
        vuint8m2_t r0 = __riscv_vle8_v_u8m2(active_rows[0] + c, vl);
        vuint8m2_t r1 = __riscv_vle8_v_u8m2(active_rows[1] + c, vl);
        vuint8m2_t r2 = __riscv_vle8_v_u8m2(active_rows[2] + c, vl);
        vuint8m2_t r3 = __riscv_vle8_v_u8m2(active_rows[3] + c, vl);
        vuint8m2_t r4 = __riscv_vle8_v_u8m2(active_rows[4] + c, vl);
        vuint8m2_t r5 = __riscv_vle8_v_u8m2(active_rows[5] + c, vl);
        vuint8m2_t r6 = __riscv_vle8_v_u8m2(active_rows[6] + c, vl);
        SortLine7(r0, r1, r2, r3, r4, r5, r6, vl);
        __riscv_vse8_v_u8m2(vert_rows[0] + c, r0, vl);
        __riscv_vse8_v_u8m2(vert_rows[1] + c, r1, vl);
        __riscv_vse8_v_u8m2(vert_rows[2] + c, r2, vl);
        __riscv_vse8_v_u8m2(vert_rows[3] + c, r3, vl);
        __riscv_vse8_v_u8m2(vert_rows[4] + c, r4, vl);
        __riscv_vse8_v_u8m2(vert_rows[5] + c, r5, vl);
        __riscv_vse8_v_u8m2(vert_rows[6] + c, r6, vl);
        c += vl;
    }
}

template <>
inline void SortColumnsVertically<9>(const uint8_t* active_rows[9], uint8_t vert_rows[9][MAX_WIDTH + 64], int padded_width)
{
    int c = 0;
    while (c < padded_width)
    {
        size_t vl = __riscv_vsetvl_e8m2(padded_width - c);
        vuint8m2_t r0 = __riscv_vle8_v_u8m2(active_rows[0] + c, vl);
        vuint8m2_t r1 = __riscv_vle8_v_u8m2(active_rows[1] + c, vl);
        vuint8m2_t r2 = __riscv_vle8_v_u8m2(active_rows[2] + c, vl);
        vuint8m2_t r3 = __riscv_vle8_v_u8m2(active_rows[3] + c, vl);
        vuint8m2_t r4 = __riscv_vle8_v_u8m2(active_rows[4] + c, vl);
        vuint8m2_t r5 = __riscv_vle8_v_u8m2(active_rows[5] + c, vl);
        vuint8m2_t r6 = __riscv_vle8_v_u8m2(active_rows[6] + c, vl);
        vuint8m2_t r7 = __riscv_vle8_v_u8m2(active_rows[7] + c, vl);
        vuint8m2_t r8 = __riscv_vle8_v_u8m2(active_rows[8] + c, vl);
        SortLine9(r0, r1, r2, r3, r4, r5, r6, r7, r8, vl);
        __riscv_vse8_v_u8m2(vert_rows[0] + c, r0, vl);
        __riscv_vse8_v_u8m2(vert_rows[1] + c, r1, vl);
        __riscv_vse8_v_u8m2(vert_rows[2] + c, r2, vl);
        __riscv_vse8_v_u8m2(vert_rows[3] + c, r3, vl);
        __riscv_vse8_v_u8m2(vert_rows[4] + c, r4, vl);
        __riscv_vse8_v_u8m2(vert_rows[5] + c, r5, vl);
        __riscv_vse8_v_u8m2(vert_rows[6] + c, r6, vl);
        __riscv_vse8_v_u8m2(vert_rows[7] + c, r7, vl);
        __riscv_vse8_v_u8m2(vert_rows[8] + c, r8, vl);
        c += vl;
    }
}

template <>
inline void SortColumnsVertically<11>(const uint8_t* active_rows[11], uint8_t vert_rows[11][MAX_WIDTH + 64], int padded_width)
{
    int c = 0;
    while (c < padded_width)
    {
        size_t vl = __riscv_vsetvl_e8m2(padded_width - c);
        vuint8m2_t r0  = __riscv_vle8_v_u8m2(active_rows[0] + c, vl);
        vuint8m2_t r1  = __riscv_vle8_v_u8m2(active_rows[1] + c, vl);
        vuint8m2_t r2  = __riscv_vle8_v_u8m2(active_rows[2] + c, vl);
        vuint8m2_t r3  = __riscv_vle8_v_u8m2(active_rows[3] + c, vl);
        vuint8m2_t r4  = __riscv_vle8_v_u8m2(active_rows[4] + c, vl);
        vuint8m2_t r5  = __riscv_vle8_v_u8m2(active_rows[5] + c, vl);
        vuint8m2_t r6  = __riscv_vle8_v_u8m2(active_rows[6] + c, vl);
        vuint8m2_t r7  = __riscv_vle8_v_u8m2(active_rows[7] + c, vl);
        vuint8m2_t r8  = __riscv_vle8_v_u8m2(active_rows[8] + c, vl);
        vuint8m2_t r9  = __riscv_vle8_v_u8m2(active_rows[9] + c, vl);
        vuint8m2_t r10 = __riscv_vle8_v_u8m2(active_rows[10] + c, vl);
        SortLine11(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, vl);
        __riscv_vse8_v_u8m2(vert_rows[0] + c, r0, vl);
        __riscv_vse8_v_u8m2(vert_rows[1] + c, r1, vl);
        __riscv_vse8_v_u8m2(vert_rows[2] + c, r2, vl);
        __riscv_vse8_v_u8m2(vert_rows[3] + c, r3, vl);
        __riscv_vse8_v_u8m2(vert_rows[4] + c, r4, vl);
        __riscv_vse8_v_u8m2(vert_rows[5] + c, r5, vl);
        __riscv_vse8_v_u8m2(vert_rows[6] + c, r6, vl);
        __riscv_vse8_v_u8m2(vert_rows[7] + c, r7, vl);
        __riscv_vse8_v_u8m2(vert_rows[8] + c, r8, vl);
        __riscv_vse8_v_u8m2(vert_rows[9] + c, r9, vl);
        __riscv_vse8_v_u8m2(vert_rows[10] + c, r10, vl);
        c += vl;
    }
}

// =============================================================================
// Horizontal Row Sorts
// =============================================================================
template <int K>
inline void SortRowsHorizontally(const uint8_t vert_rows[K][MAX_WIDTH + 64],
                                 uint8_t chunk_buf[K][K][32],
                                 int x, size_t vl);

template <>
inline void SortRowsHorizontally<3>(const uint8_t vert_rows[3][MAX_WIDTH + 64], uint8_t chunk_buf[3][3][32], int x, size_t vl)
{
    for (int r = 0; r < 3; ++r)
    {
        vuint8m2_t v0 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 0, vl);
        vuint8m2_t v1 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 1, vl);
        vuint8m2_t v2 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 2, vl);
        SortLine3(v0, v1, v2, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][0], v0, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][1], v1, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][2], v2, vl);
    }
}

template <>
inline void SortRowsHorizontally<5>(const uint8_t vert_rows[5][MAX_WIDTH + 64], uint8_t chunk_buf[5][5][32], int x, size_t vl)
{
    for (int r = 0; r < 5; ++r)
    {
        vuint8m2_t v0 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 0, vl);
        vuint8m2_t v1 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 1, vl);
        vuint8m2_t v2 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 2, vl);
        vuint8m2_t v3 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 3, vl);
        vuint8m2_t v4 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 4, vl);
        SortLine5(v0, v1, v2, v3, v4, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][0], v0, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][1], v1, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][2], v2, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][3], v3, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][4], v4, vl);
    }
}

template <>
inline void SortRowsHorizontally<7>(const uint8_t vert_rows[7][MAX_WIDTH + 64], uint8_t chunk_buf[7][7][32], int x, size_t vl)
{
    for (int r = 0; r < 7; ++r)
    {
        vuint8m2_t v0 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 0, vl);
        vuint8m2_t v1 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 1, vl);
        vuint8m2_t v2 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 2, vl);
        vuint8m2_t v3 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 3, vl);
        vuint8m2_t v4 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 4, vl);
        vuint8m2_t v5 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 5, vl);
        vuint8m2_t v6 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 6, vl);
        SortLine7(v0, v1, v2, v3, v4, v5, v6, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][0], v0, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][1], v1, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][2], v2, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][3], v3, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][4], v4, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][5], v5, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][6], v6, vl);
    }
}

template <>
inline void SortRowsHorizontally<9>(const uint8_t vert_rows[9][MAX_WIDTH + 64], uint8_t chunk_buf[9][9][32], int x, size_t vl)
{
    for (int r = 0; r < 9; ++r)
    {
        vuint8m2_t v0 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 0, vl);
        vuint8m2_t v1 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 1, vl);
        vuint8m2_t v2 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 2, vl);
        vuint8m2_t v3 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 3, vl);
        vuint8m2_t v4 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 4, vl);
        vuint8m2_t v5 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 5, vl);
        vuint8m2_t v6 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 6, vl);
        vuint8m2_t v7 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 7, vl);
        vuint8m2_t v8 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 8, vl);
        SortLine9(v0, v1, v2, v3, v4, v5, v6, v7, v8, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][0], v0, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][1], v1, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][2], v2, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][3], v3, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][4], v4, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][5], v5, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][6], v6, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][7], v7, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][8], v8, vl);
    }
}

template <>
inline void SortRowsHorizontally<11>(const uint8_t vert_rows[11][MAX_WIDTH + 64], uint8_t chunk_buf[11][11][32], int x, size_t vl)
{
    for (int r = 0; r < 11; ++r)
    {
        vuint8m2_t v0  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 0, vl);
        vuint8m2_t v1  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 1, vl);
        vuint8m2_t v2  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 2, vl);
        vuint8m2_t v3  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 3, vl);
        vuint8m2_t v4  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 4, vl);
        vuint8m2_t v5  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 5, vl);
        vuint8m2_t v6  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 6, vl);
        vuint8m2_t v7  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 7, vl);
        vuint8m2_t v8  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 8, vl);
        vuint8m2_t v9  = __riscv_vle8_v_u8m2(vert_rows[r] + x + 9, vl);
        vuint8m2_t v10 = __riscv_vle8_v_u8m2(vert_rows[r] + x + 10, vl);
        SortLine11(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][0], v0, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][1], v1, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][2], v2, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][3], v3, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][4], v4, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][5], v5, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][6], v6, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][7], v7, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][8], v8, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][9], v9, vl);
        __riscv_vse8_v_u8m2(chunk_buf[r][10], v10, vl);
    }
}

// =============================================================================
// Diagonal Reductions (ExtractMedian)
// =============================================================================
template <int K>
inline vuint8m2_t ExtractMedian(uint8_t chunk_buf[K][K][32], size_t vl);

template <>
inline vuint8m2_t ExtractMedian<3>(uint8_t chunk_buf[3][3][32], size_t vl)
{
    vuint8m2_t d0 = __riscv_vle8_v_u8m2(chunk_buf[0][2], vl);
    vuint8m2_t d1 = __riscv_vle8_v_u8m2(chunk_buf[1][1], vl);
    vuint8m2_t d2 = __riscv_vle8_v_u8m2(chunk_buf[2][0], vl);
    SortLine3(d0, d1, d2, vl);
    return d1;
}

template <>
inline vuint8m2_t ExtractMedian<5>(uint8_t chunk_buf[5][5][32], size_t vl)
{
    vuint8m2_t a30, a22, a14;
    {
        vuint8m2_t d0 = __riscv_vle8_v_u8m2(chunk_buf[0][3], vl);
        vuint8m2_t d1 = __riscv_vle8_v_u8m2(chunk_buf[1][2], vl);
        vuint8m2_t d2 = __riscv_vle8_v_u8m2(chunk_buf[2][1], vl);
        vuint8m2_t d3 = __riscv_vle8_v_u8m2(chunk_buf[3][0], vl);
        SortLine4(d0, d1, d2, d3, vl);
        a30 = d3;
    }
    {
        vuint8m2_t d0 = __riscv_vle8_v_u8m2(chunk_buf[0][4], vl);
        vuint8m2_t d1 = __riscv_vle8_v_u8m2(chunk_buf[1][3], vl);
        vuint8m2_t d2 = __riscv_vle8_v_u8m2(chunk_buf[2][2], vl);
        vuint8m2_t d3 = __riscv_vle8_v_u8m2(chunk_buf[3][1], vl);
        vuint8m2_t d4 = __riscv_vle8_v_u8m2(chunk_buf[4][0], vl);
        SortLine5(d0, d1, d2, d3, d4, vl);
        a22 = d2;
    }
    {
        vuint8m2_t d0 = __riscv_vle8_v_u8m2(chunk_buf[1][4], vl);
        vuint8m2_t d1 = __riscv_vle8_v_u8m2(chunk_buf[2][3], vl);
        vuint8m2_t d2 = __riscv_vle8_v_u8m2(chunk_buf[3][2], vl);
        vuint8m2_t d3 = __riscv_vle8_v_u8m2(chunk_buf[4][1], vl);
        SortLine4(d0, d1, d2, d3, vl);
        a14 = d0;
    }
    SortLine3(a14, a22, a30, vl);
    return a22;
}

template <>
inline vuint8m2_t ExtractMedian<7>(uint8_t chunk_buf[7][7][32], size_t vl)
{
    #define LOAD_V(r, c) __riscv_vle8_v_u8m2(chunk_buf[r][c], vl)
    #define STORE_V(r, c, v) __riscv_vse8_v_u8m2(chunk_buf[r][c], v, vl)

    #define SORT_D5(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4); \
        SortLine5(d0, d1, d2, d3, d4, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); \
    } while(0)

    #define SORT_D6(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5); \
        SortLine6(d0, d1, d2, d3, d4, d5, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); \
    } while(0)

    #define SORT_D7(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6); \
        SortLine7(d0, d1, d2, d3, d4, d5, d6, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); \
    } while(0)

    SORT_D5(0,4, 1,3, 2,2, 3,1, 4,0);
    SORT_D6(0,5, 1,4, 2,3, 3,2, 4,1, 5,0);
    SORT_D7(0,6, 1,5, 2,4, 3,3, 4,2, 5,1, 6,0);
    SORT_D6(1,6, 2,5, 3,4, 4,3, 5,2, 6,1);
    SORT_D5(2,6, 3,5, 4,4, 5,3, 6,2);

    #undef SORT_D5
    #undef SORT_D6
    #undef SORT_D7

    vuint8m2_t s3_p40, s3_p33, s3_p26;
    {
        vuint8m2_t d0 = LOAD_V(1, 6), d1 = LOAD_V(2, 4), d2 = LOAD_V(3, 2), d3 = LOAD_V(4, 0);
        SortLine4(d0, d1, d2, d3, vl);
        s3_p40 = d3;
    }
    {
        vuint8m2_t d0 = LOAD_V(2, 5), d1 = LOAD_V(3, 3), d2 = LOAD_V(4, 1);
        SortLine3(d0, d1, d2, vl);
        s3_p33 = d1;
    }
    {
        vuint8m2_t d0 = LOAD_V(2, 6), d1 = LOAD_V(3, 4), d2 = LOAD_V(4, 2), d3 = LOAD_V(5, 0);
        SortLine4(d0, d1, d2, d3, vl);
        s3_p26 = d0;
    }

    #undef LOAD_V
    #undef STORE_V

    SortLine3(s3_p26, s3_p33, s3_p40, vl);
    return s3_p33;
}

template <>
inline vuint8m2_t ExtractMedian<9>(uint8_t chunk_buf[9][9][32], size_t vl)
{
    #define LOAD_V(r, c) __riscv_vle8_v_u8m2(chunk_buf[r][c], vl)
    #define STORE_V(r, c, v) __riscv_vse8_v_u8m2(chunk_buf[r][c], v, vl)

    #define SORT_D4(r0,c0, r1,c1, r2,c2, r3,c3) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3); \
        SortLine4(d0, d1, d2, d3, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); \
    } while(0)

    #define SORT_D5(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4); \
        SortLine5(d0, d1, d2, d3, d4, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); \
    } while(0)

    #define SORT_D6(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5); \
        SortLine6(d0, d1, d2, d3, d4, d5, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); \
    } while(0)

    #define SORT_D7(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6); \
        SortLine7(d0, d1, d2, d3, d4, d5, d6, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); \
    } while(0)

    #define SORT_D8(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6, r7,c7) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6), d7 = LOAD_V(r7,c7); \
        SortLine8(d0, d1, d2, d3, d4, d5, d6, d7, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); STORE_V(r7,c7, d7); \
    } while(0)

    #define SORT_D9(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6, r7,c7, r8,c8) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6), d7 = LOAD_V(r7,c7), d8 = LOAD_V(r8,c8); \
        SortLine9(d0, d1, d2, d3, d4, d5, d6, d7, d8, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); STORE_V(r7,c7, d7); STORE_V(r8,c8, d8); \
    } while(0)

    SORT_D6(0,5, 1,4, 2,3, 3,2, 4,1, 5,0);
    SORT_D7(0,6, 1,5, 2,4, 3,3, 4,2, 5,1, 6,0);
    SORT_D8(0,7, 1,6, 2,5, 3,4, 4,3, 5,2, 6,1, 7,0);
    SORT_D9(0,8, 1,7, 2,6, 3,5, 4,4, 5,3, 6,2, 7,1, 8,0);
    SORT_D8(1,8, 2,7, 3,6, 4,5, 5,4, 6,3, 7,2, 8,1);
    SORT_D7(2,8, 3,7, 4,6, 5,5, 6,4, 7,3, 8,2);
    SORT_D6(3,8, 4,7, 5,6, 6,5, 7,4, 8,3);

    SORT_D5(1,8, 2,6, 3,4, 4,2, 5,0);
    SORT_D4(2,7, 3,5, 4,3, 5,1);
    SORT_D5(2,8, 3,6, 4,4, 5,2, 6,0);
    SORT_D4(3,7, 4,5, 5,3, 6,1);
    SORT_D5(3,8, 4,6, 5,4, 6,2, 7,0);

    #undef SORT_D4
    #undef SORT_D5
    #undef SORT_D6
    #undef SORT_D7
    #undef SORT_D8
    #undef SORT_D9

    vuint8m2_t s4_p50, s4_p44, s4_p38;
    {
        vuint8m2_t d0 = LOAD_V(3,6), d1 = LOAD_V(4,3), d2 = LOAD_V(5,0);
        SortLine3(d0, d1, d2, vl);
        s4_p50 = d2;
    }
    {
        vuint8m2_t d0 = LOAD_V(3,7), d1 = LOAD_V(4,4), d2 = LOAD_V(5,1);
        SortLine3(d0, d1, d2, vl);
        s4_p44 = d1;
    }
    {
        vuint8m2_t d0 = LOAD_V(3,8), d1 = LOAD_V(4,5), d2 = LOAD_V(5,2);
        SortLine3(d0, d1, d2, vl);
        s4_p38 = d0;
    }

    #undef LOAD_V
    #undef STORE_V

    SortLine3(s4_p38, s4_p44, s4_p50, vl);
    return s4_p44;
}

template <>
inline vuint8m2_t ExtractMedian<11>(uint8_t chunk_buf[11][11][32], size_t vl)
{
    #define LOAD_V(r, c) __riscv_vle8_v_u8m2(chunk_buf[r][c], vl)
    #define STORE_V(r, c, v) __riscv_vse8_v_u8m2(chunk_buf[r][c], v, vl)

    #define SORT_D3(r0,c0, r1,c1, r2,c2) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2); \
        SortLine3(d0, d1, d2, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); \
    } while(0)

    #define SORT_D4(r0,c0, r1,c1, r2,c2, r3,c3) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3); \
        SortLine4(d0, d1, d2, d3, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); \
    } while(0)

    #define SORT_D5(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4); \
        SortLine5(d0, d1, d2, d3, d4, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); \
    } while(0)

    #define SORT_D6(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5); \
        SortLine6(d0, d1, d2, d3, d4, d5, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); \
    } while(0)

    #define SORT_D7(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6); \
        SortLine7(d0, d1, d2, d3, d4, d5, d6, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); \
    } while(0)

    #define SORT_D8(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6, r7,c7) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6), d7 = LOAD_V(r7,c7); \
        SortLine8(d0, d1, d2, d3, d4, d5, d6, d7, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); STORE_V(r7,c7, d7); \
    } while(0)

    #define SORT_D9(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6, r7,c7, r8,c8) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6), d7 = LOAD_V(r7,c7), d8 = LOAD_V(r8,c8); \
        SortLine9(d0, d1, d2, d3, d4, d5, d6, d7, d8, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); STORE_V(r7,c7, d7); STORE_V(r8,c8, d8); \
    } while(0)

    #define SORT_D10(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6, r7,c7, r8,c8, r9,c9) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6), d7 = LOAD_V(r7,c7), d8 = LOAD_V(r8,c8), d9 = LOAD_V(r9,c9); \
        SortLine10(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); STORE_V(r7,c7, d7); STORE_V(r8,c8, d8); STORE_V(r9,c9, d9); \
    } while(0)

    #define SORT_D11(r0,c0, r1,c1, r2,c2, r3,c3, r4,c4, r5,c5, r6,c6, r7,c7, r8,c8, r9,c9, r10,c10) do { \
        vuint8m2_t d0 = LOAD_V(r0,c0), d1 = LOAD_V(r1,c1), d2 = LOAD_V(r2,c2), d3 = LOAD_V(r3,c3), d4 = LOAD_V(r4,c4), d5 = LOAD_V(r5,c5), d6 = LOAD_V(r6,c6), d7 = LOAD_V(r7,c7), d8 = LOAD_V(r8,c8), d9 = LOAD_V(r9,c9), d10 = LOAD_V(r10,c10); \
        SortLine11(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, vl); \
        STORE_V(r0,c0, d0); STORE_V(r1,c1, d1); STORE_V(r2,c2, d2); STORE_V(r3,c3, d3); STORE_V(r4,c4, d4); STORE_V(r5,c5, d5); STORE_V(r6,c6, d6); STORE_V(r7,c7, d7); STORE_V(r8,c8, d8); STORE_V(r9,c9, d9); STORE_V(r10,c10, d10); \
    } while(0)

    // Slope k = 1: 9 diagonals
    SORT_D7(0,6, 1,5, 2,4, 3,3, 4,2, 5,1, 6,0);
    SORT_D8(0,7, 1,6, 2,5, 3,4, 4,3, 5,2, 6,1, 7,0);
    SORT_D9(0,8, 1,7, 2,6, 3,5, 4,4, 5,3, 6,2, 7,1, 8,0);
    SORT_D10(0,9, 1,8, 2,7, 3,6, 4,5, 5,4, 6,3, 7,2, 8,1, 9,0);
    SORT_D11(0,10, 1,9, 2,8, 3,7, 4,6, 5,5, 6,4, 7,3, 8,2, 9,1, 10,0);
    SORT_D10(1,10, 2,9, 3,8, 4,7, 5,6, 6,5, 7,4, 8,3, 9,2, 10,1);
    SORT_D9(2,10, 3,9, 4,8, 5,7, 6,6, 7,5, 8,4, 9,3, 10,2);
    SORT_D8(3,10, 4,9, 5,8, 6,7, 7,6, 8,5, 9,4, 10,3);
    SORT_D7(4,10, 5,9, 6,8, 7,7, 8,6, 9,5, 10,4);

    // Slope k = 2: 7 diagonals
    SORT_D6(1,10, 2,8, 3,6, 4,4, 5,2, 6,0);
    SORT_D5(2,9, 3,7, 4,5, 5,3, 6,1);
    SORT_D6(2,10, 3,8, 4,6, 5,4, 6,2, 7,0);
    SORT_D5(3,9, 4,7, 5,5, 6,3, 7,1);
    SORT_D6(3,10, 4,8, 5,6, 6,4, 7,2, 8,0);
    SORT_D5(4,9, 5,7, 6,5, 7,3, 8,1);
    SORT_D6(4,10, 5,8, 6,6, 7,4, 8,2, 9,0);

    // Slope k = 3: 5 diagonals
    SORT_D4(3,9, 4,6, 5,3, 6,0);
    SORT_D4(3,10, 4,7, 5,4, 6,1);
    SORT_D3(4,8, 5,5, 6,2);
    SORT_D4(4,9, 5,6, 6,3, 7,0);
    SORT_D4(4,10, 5,7, 6,4, 7,1);

    #undef SORT_D4
    #undef SORT_D5
    #undef SORT_D6
    #undef SORT_D7
    #undef SORT_D8
    #undef SORT_D9
    #undef SORT_D10
    #undef SORT_D11

    // Slope k = 4: 3 diagonals
    vuint8m2_t s5_p60, s5_p55, s5_p410;
    {
        vuint8m2_t d0 = LOAD_V(4,8), d1 = LOAD_V(5,4), d2 = LOAD_V(6,0);
        SortLine3(d0, d1, d2, vl);
        s5_p60 = d2;
    }
    {
        vuint8m2_t d0 = LOAD_V(4,9), d1 = LOAD_V(5,5), d2 = LOAD_V(6,1);
        SortLine3(d0, d1, d2, vl);
        s5_p55 = d1;
    }
    {
        vuint8m2_t d0 = LOAD_V(4,10), d1 = LOAD_V(5,6), d2 = LOAD_V(6,2);
        SortLine3(d0, d1, d2, vl);
        s5_p410 = d0;
    }

    #undef LOAD_V
    #undef STORE_V
    #undef SORT_D3

    // Slope k = 5: in-register ALU
    SortLine3(s5_p410, s5_p55, s5_p60, vl);
    return s5_p55;
}
