#pragma once

#include <cstddef>
#include <cstdint>

// =============================================================================
// Scalar Compare-and-Swap Primitives
// =============================================================================
constexpr int MAX_WIDTH = 1024;
constexpr size_t MAX_VL = 32;
#define SCALAR_CAS(a, b) do { \
    uint8_t mn = (a < b) ? a : b; \
    uint8_t mx = (a < b) ? b : a; \
    a = mn; b = mx; \
} while(0)

#define SCALAR_CAS_DESC(a, b) do { \
    uint8_t mn = (a < b) ? a : b; \
    uint8_t mx = (a < b) ? b : a; \
    a = mx; b = mn; \
} while(0)

// =============================================================================
// =============================================================================
// Optimal Scalar Sorting Networks for Lines (Lengths 3 to 11)
// =============================================================================
// =============================================================================
static inline void sort_line_3_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2)
{
    SCALAR_CAS(x0, x1);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x0, x1);
}

static inline void sort_line_4_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3)
{
    SCALAR_CAS(x0, x1); SCALAR_CAS(x2, x3);
    SCALAR_CAS(x0, x2); SCALAR_CAS(x1, x3);
    SCALAR_CAS(x1, x2);
}

static inline void sort_line_5_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3, uint8_t& x4)
{
    SCALAR_CAS(x0, x1); SCALAR_CAS(x3, x4);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x2, x3);
    SCALAR_CAS(x0, x3);
    SCALAR_CAS(x0, x2);
    SCALAR_CAS(x1, x4);
    SCALAR_CAS(x1, x3);
    SCALAR_CAS(x1, x2);
}

static inline void sort_line_6_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3, uint8_t& x4, uint8_t& x5)
{
    SCALAR_CAS(x1, x2); SCALAR_CAS(x0, x2); SCALAR_CAS(x0, x1);
    SCALAR_CAS(x4, x5); SCALAR_CAS(x3, x5); SCALAR_CAS(x3, x4);
    SCALAR_CAS(x0, x3); SCALAR_CAS(x1, x4); SCALAR_CAS(x2, x5);
    SCALAR_CAS(x2, x4); SCALAR_CAS(x1, x3);
    SCALAR_CAS(x2, x3);
}

static inline void sort_line_7_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3,
                                      uint8_t& x4, uint8_t& x5, uint8_t& x6)
{
    SCALAR_CAS(x0, x6); SCALAR_CAS(x2, x3); SCALAR_CAS(x4, x5);
    SCALAR_CAS(x0, x2); SCALAR_CAS(x1, x4); SCALAR_CAS(x3, x6);
    SCALAR_CAS(x0, x1); SCALAR_CAS(x2, x5); SCALAR_CAS(x3, x4);
    SCALAR_CAS(x1, x2); SCALAR_CAS(x4, x6);
    SCALAR_CAS(x2, x3); SCALAR_CAS(x4, x5);
    SCALAR_CAS(x1, x2); SCALAR_CAS(x3, x4); SCALAR_CAS(x5, x6);
}

static inline void sort_line_8_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3, uint8_t& x4, uint8_t& x5, uint8_t& x6, uint8_t& x7)
{
    SCALAR_CAS(x0, x2);
    SCALAR_CAS(x1, x3);
    SCALAR_CAS(x4, x6);
    SCALAR_CAS(x5, x7);
    SCALAR_CAS(x0, x4);
    SCALAR_CAS(x1, x5);
    SCALAR_CAS(x2, x6);
    SCALAR_CAS(x3, x7);
    SCALAR_CAS(x0, x1);
    SCALAR_CAS(x2, x3);
    SCALAR_CAS(x4, x5);
    SCALAR_CAS(x6, x7);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x3, x5);
    SCALAR_CAS(x1, x4);
    SCALAR_CAS(x3, x6);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x3, x4);
    SCALAR_CAS(x5, x6);
}
static inline void sort_line_9_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3, uint8_t& x4, uint8_t& x5, uint8_t& x6, uint8_t& x7, uint8_t& x8)
{
    SCALAR_CAS(x0, x3);
    SCALAR_CAS(x1, x7);
    SCALAR_CAS(x2, x5);
    SCALAR_CAS(x4, x8);
    SCALAR_CAS(x0, x7);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x3, x8);
    SCALAR_CAS(x5, x6);
    SCALAR_CAS(x0, x2);
    SCALAR_CAS(x1, x3);
    SCALAR_CAS(x4, x5);
    SCALAR_CAS(x7, x8);
    SCALAR_CAS(x1, x4);
    SCALAR_CAS(x3, x6);
    SCALAR_CAS(x5, x7);
    SCALAR_CAS(x0, x1);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x3, x5);
    SCALAR_CAS(x6, x8);
    SCALAR_CAS(x2, x3);
    SCALAR_CAS(x4, x5);
    SCALAR_CAS(x6, x7);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x3, x4);
    SCALAR_CAS(x5, x6);
}
static inline void sort_line_10_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3, uint8_t& x4, uint8_t& x5, uint8_t& x6, uint8_t& x7, uint8_t& x8, uint8_t& x9)
{
    SCALAR_CAS(x0, x8);
    SCALAR_CAS(x1, x9);
    SCALAR_CAS(x2, x7);
    SCALAR_CAS(x3, x5);
    SCALAR_CAS(x4, x6);
    SCALAR_CAS(x0, x2);
    SCALAR_CAS(x1, x4);
    SCALAR_CAS(x5, x8);
    SCALAR_CAS(x7, x9);
    SCALAR_CAS(x0, x3);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x5, x7);
    SCALAR_CAS(x6, x9);
    SCALAR_CAS(x0, x1);
    SCALAR_CAS(x3, x6);
    SCALAR_CAS(x8, x9);
    SCALAR_CAS(x1, x5);
    SCALAR_CAS(x2, x3);
    SCALAR_CAS(x4, x8);
    SCALAR_CAS(x6, x7);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x3, x5);
    SCALAR_CAS(x4, x6);
    SCALAR_CAS(x7, x8);
    SCALAR_CAS(x2, x3);
    SCALAR_CAS(x4, x5);
    SCALAR_CAS(x6, x7);
    SCALAR_CAS(x3, x4);
    SCALAR_CAS(x5, x6);
}
static inline void sort_line_11_scalar(uint8_t& x0, uint8_t& x1, uint8_t& x2, uint8_t& x3, uint8_t& x4, uint8_t& x5, uint8_t& x6, uint8_t& x7, uint8_t& x8, uint8_t& x9, uint8_t& x10)
{
    SCALAR_CAS(x0, x1);
    SCALAR_CAS(x2, x3);
    SCALAR_CAS(x4, x5);
    SCALAR_CAS(x6, x7);
    SCALAR_CAS(x8, x9);
    SCALAR_CAS(x0, x2);
    SCALAR_CAS(x1, x3);
    SCALAR_CAS(x4, x6);
    SCALAR_CAS(x5, x7);
    SCALAR_CAS(x8, x10);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x5, x6);
    SCALAR_CAS(x0, x4);
    SCALAR_CAS(x1, x5);
    SCALAR_CAS(x2, x6);
    SCALAR_CAS(x3, x7);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x3, x5);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x3, x4);
    SCALAR_CAS(x5, x6);
    SCALAR_CAS(x9, x10);
    SCALAR_CAS(x0, x8);
    SCALAR_CAS(x1, x9);
    SCALAR_CAS(x2, x10);
    SCALAR_CAS(x4, x8);
    SCALAR_CAS(x5, x9);
    SCALAR_CAS(x6, x10);
    SCALAR_CAS(x2, x4);
    SCALAR_CAS(x3, x5);
    SCALAR_CAS(x6, x8);
    SCALAR_CAS(x7, x9);
    SCALAR_CAS(x1, x2);
    SCALAR_CAS(x3, x4);
    SCALAR_CAS(x5, x6);
    SCALAR_CAS(x7, x8);
    SCALAR_CAS(x9, x10);
}


// =============================================================================
// Generalized Scalar SortLine Dispatcher (Resolved at compile-time via if constexpr)
// =============================================================================
template <int K>
inline void sort_line_scalar(uint8_t v[K])
{
    if constexpr (K == 3)
    {
        sort_line_3_scalar(v[0], v[1], v[2]);
    }
    else if constexpr (K == 5)
    {
        sort_line_5_scalar(v[0], v[1], v[2], v[3], v[4]);
    }
    else if constexpr (K == 7)
    {
        sort_line_7_scalar(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }
    else if constexpr (K == 9)
    {
        sort_line_9_scalar(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    }
    else if constexpr (K == 11)
    {
        sort_line_11_scalar(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
    }
}

// =============================================================================
// Generalized Scalar Vertical Column Sort
// =============================================================================
template <int K>
inline void SortColumnsVerticallyScalar(const uint8_t* active_rows[K],
                                        uint8_t vert_rows[K][MAX_WIDTH + 64],
                                        int padded_width)
{
    for (int c = 0; c < padded_width; ++c)
    {
        uint8_t col[K];
        for (int i = 0; i < K; ++i)
        {
            col[i] = active_rows[i][c];
        }

        sort_line_scalar<K>(col);

        for (int i = 0; i < K; ++i)
        {
            vert_rows[i][c] = col[i];
        }
    }
}

// =============================================================================
// Generalized Scalar Horizontal Row Sort
// =============================================================================
template <int K>
inline void SortRowsHorizontallyScalar(const uint8_t vert_rows[K][MAX_WIDTH + 64],
                                       uint8_t A[K][K], int x)
{
    for (int r = 0; r < K; ++r)
    {
        uint8_t row[K];
        for (int c = 0; c < K; ++c)
        {
            row[c] = vert_rows[r][x + c];
        }

        sort_line_scalar<K>(row);

        for (int c = 0; c < K; ++c)
        {
            A[r][c] = row[c];
        }
    }
}

// =============================================================================
// Scalar Diagonal Reductions (ExtractMedianScalar)
// =============================================================================
template <int K>
inline uint8_t ExtractMedianScalar(uint8_t A[K][K]);

template <>
inline uint8_t ExtractMedianScalar<3>(uint8_t A[3][3])
{
    sort_line_3_scalar(A[0][2], A[1][1], A[2][0]);
    return A[1][1];
}

template <>
inline uint8_t ExtractMedianScalar<5>(uint8_t A[5][5])
{
    sort_line_4_scalar(A[0][3], A[1][2], A[2][1], A[3][0]);
    sort_line_5_scalar(A[0][4], A[1][3], A[2][2], A[3][1], A[4][0]);
    sort_line_4_scalar(A[1][4], A[2][3], A[3][2], A[4][1]);
    sort_line_3_scalar(A[1][4], A[2][2], A[3][0]);
    return A[2][2];
}

template <>
inline uint8_t ExtractMedianScalar<7>(uint8_t A[7][7])
{
    sort_line_5_scalar(A[0][4], A[1][3], A[2][2], A[3][1], A[4][0]);
    sort_line_6_scalar(A[0][5], A[1][4], A[2][3], A[3][2], A[4][1], A[5][0]);
    sort_line_7_scalar(A[0][6], A[1][5], A[2][4], A[3][3], A[4][2], A[5][1], A[6][0]);
    sort_line_6_scalar(A[1][6], A[2][5], A[3][4], A[4][3], A[5][2], A[6][1]);
    sort_line_5_scalar(A[2][6], A[3][5], A[4][4], A[5][3], A[6][2]);

    sort_line_4_scalar(A[1][6], A[2][4], A[3][2], A[4][0]);
    sort_line_3_scalar(A[2][5], A[3][3], A[4][1]);
    sort_line_4_scalar(A[2][6], A[3][4], A[4][2], A[5][0]);

    sort_line_3_scalar(A[2][6], A[3][3], A[4][0]);
    return A[3][3];
}

template <>
inline uint8_t ExtractMedianScalar<9>(uint8_t A[9][9])
{
    sort_line_6_scalar(A[0][5], A[1][4], A[2][3], A[3][2], A[4][1], A[5][0]);
    sort_line_7_scalar(A[0][6], A[1][5], A[2][4], A[3][3], A[4][2], A[5][1], A[6][0]);
    sort_line_8_scalar(A[0][7], A[1][6], A[2][5], A[3][4], A[4][3], A[5][2], A[6][1], A[7][0]);
    sort_line_9_scalar(A[0][8], A[1][7], A[2][6], A[3][5], A[4][4], A[5][3], A[6][2], A[7][1], A[8][0]);
    sort_line_8_scalar(A[1][8], A[2][7], A[3][6], A[4][5], A[5][4], A[6][3], A[7][2], A[8][1]);
    sort_line_7_scalar(A[2][8], A[3][7], A[4][6], A[5][5], A[6][4], A[7][3], A[8][2]);
    sort_line_6_scalar(A[3][8], A[4][7], A[5][6], A[6][5], A[7][4], A[8][3]);

    sort_line_5_scalar(A[1][8], A[2][6], A[3][4], A[4][2], A[5][0]);
    sort_line_4_scalar(A[2][7], A[3][5], A[4][3], A[5][1]);
    sort_line_5_scalar(A[2][8], A[3][6], A[4][4], A[5][2], A[6][0]);
    sort_line_4_scalar(A[3][7], A[4][5], A[5][3], A[6][1]);
    sort_line_5_scalar(A[3][8], A[4][6], A[5][4], A[6][2], A[7][0]);

    sort_line_3_scalar(A[3][6], A[4][3], A[5][0]);
    sort_line_3_scalar(A[3][7], A[4][4], A[5][1]);
    sort_line_3_scalar(A[3][8], A[4][5], A[5][2]);

    sort_line_3_scalar(A[3][8], A[4][4], A[5][0]);
    return A[4][4];
}

template <>
inline uint8_t ExtractMedianScalar<11>(uint8_t A[11][11])
{
    sort_line_7_scalar(A[0][6], A[1][5], A[2][4], A[3][3], A[4][2], A[5][1], A[6][0]);
    sort_line_8_scalar(A[0][7], A[1][6], A[2][5], A[3][4], A[4][3], A[5][2], A[6][1], A[7][0]);
    sort_line_9_scalar(A[0][8], A[1][7], A[2][6], A[3][5], A[4][4], A[5][3], A[6][2], A[7][1], A[8][0]);
    sort_line_10_scalar(A[0][9], A[1][8], A[2][7], A[3][6], A[4][5], A[5][4], A[6][3], A[7][2], A[8][1], A[9][0]);
    sort_line_11_scalar(A[0][10], A[1][9], A[2][8], A[3][7], A[4][6], A[5][5], A[6][4], A[7][3], A[8][2], A[9][1], A[10][0]);
    sort_line_10_scalar(A[1][10], A[2][9], A[3][8], A[4][7], A[5][6], A[6][5], A[7][4], A[8][3], A[9][2], A[10][1]);
    sort_line_9_scalar(A[2][10], A[3][9], A[4][8], A[5][7], A[6][6], A[7][5], A[8][4], A[9][3], A[10][2]);
    sort_line_8_scalar(A[3][10], A[4][9], A[5][8], A[6][7], A[7][6], A[8][5], A[9][4], A[10][3]);
    sort_line_7_scalar(A[4][10], A[5][9], A[6][8], A[7][7], A[8][6], A[9][5], A[10][4]);

    sort_line_6_scalar(A[1][10], A[2][8], A[3][6], A[4][4], A[5][2], A[6][0]);
    sort_line_5_scalar(A[2][9], A[3][7], A[4][5], A[5][3], A[6][1]);
    sort_line_6_scalar(A[2][10], A[3][8], A[4][6], A[5][4], A[6][2], A[7][0]);
    sort_line_5_scalar(A[3][9], A[4][7], A[5][5], A[6][3], A[7][1]);
    sort_line_6_scalar(A[3][10], A[4][8], A[5][6], A[6][4], A[7][2], A[8][0]);
    sort_line_5_scalar(A[4][9], A[5][7], A[6][5], A[7][3], A[8][1]);
    sort_line_6_scalar(A[4][10], A[5][8], A[6][6], A[7][4], A[8][2], A[9][0]);

    sort_line_4_scalar(A[3][9], A[4][6], A[5][3], A[6][0]);
    sort_line_4_scalar(A[3][10], A[4][7], A[5][4], A[6][1]);
    sort_line_3_scalar(A[4][8], A[5][5], A[6][2]);
    sort_line_4_scalar(A[4][9], A[5][6], A[6][3], A[7][0]);
    sort_line_4_scalar(A[4][10], A[5][7], A[6][4], A[7][1]);

    sort_line_3_scalar(A[4][8], A[5][4], A[6][0]);
    sort_line_3_scalar(A[4][9], A[5][5], A[6][1]);
    sort_line_3_scalar(A[4][10], A[5][6], A[6][2]);

    sort_line_3_scalar(A[4][10], A[5][5], A[6][0]);
    return A[5][5];
}
