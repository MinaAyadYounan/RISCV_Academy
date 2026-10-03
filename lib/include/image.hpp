#pragma once

#include "riscv_port.hpp"
#include "types.hpp"
#include <cstddef>
#include <cstdint>
#include <algorithm>

enum class ImageType
{
    GRAY,
    RGB,
};

enum class PaddingType
{
    CONSTANT,
    REPLICA,
    REFLECT,
};

template <typename T, ImageType Type = ImageType::GRAY>
class Image
{
private:
    int width;
    int height;
    int stride;

    T* data;

    static constexpr int Channels()
    {
        return Type == ImageType::GRAY ? 1 : 3;
    }

    int index(const int x, const int y, const int c = 0) const
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        assert(c >= 0 && c < Channels());
        return y * stride + x * Channels() + c;
    }

    void Allocate(int _width, int _height)
    {
        assert(_width > 0 && _height > 0);
        width = _width;
        height = _height;
        stride = _width * Channels();
        data = new T[stride * _height];
    }

public:
    Image() : width(0), height(0), stride(0), data(nullptr) {}
    Image(int width, int height)
    {
        Allocate(width, height);
    }

    ~Image()
    {
        delete[] data;
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& other) noexcept
        : width(other.width), height(other.height),
          stride(other.stride), data(other.data)
    {
        other.width = 0;
        other.height = 0;
        other.stride = 0;
        other.data = nullptr;
    }

    Image& operator=(Image&& other) noexcept
    {
        if (this != &other)
        {
            delete[] data;
            width = other.width;
            height = other.height;
            stride = other.stride;
            data = other.data;
            other.width = 0;
            other.height = 0;
            other.stride = 0;
            other.data = nullptr;
        }
        return *this;
    }

    T GetPixel(const int x, const int y, const int c = 0) const
    {
        return data[index(x, y, c)];
    }

    void SetPixel(const int x, const int y, const T& value, const int c = 0)
    {
        data[index(x, y, c)] = value;
    }

    T* GetPtr(const int x, const int y)
    {
        return data + index(x, y);
    }

    const T* GetPtr(const int x, const int y) const
    {
        return data + index(x, y);
    }
    template <PaddingType Padding = PaddingType::REPLICA>
    const T* GetRowPtr(int y) const
    {
        if constexpr (Padding == PaddingType::REPLICA)
        {
            int cy = std::clamp(y, 0, height - 1);
            return data + cy * stride;
        }
        else if constexpr (Padding == PaddingType::REFLECT)
        {
            int ry = (y < 0) ? -y : (y >= height) ? 2 * height - 2 - y : y;
            ry = std::clamp(ry, 0, height - 1);
            return data + ry * stride;
        }
        else if constexpr (Padding == PaddingType::CONSTANT)
        {
            if (y >= 0 && y < height)
            {
                return data + y * stride;
            }
            return nullptr;
        }
    }

    const T* GetRowPtr(int y, PaddingType padding) const
    {
        switch (padding)
        {
        case PaddingType::REPLICA:  return GetRowPtr<PaddingType::REPLICA>(y);
        case PaddingType::REFLECT:  return GetRowPtr<PaddingType::REFLECT>(y);
        case PaddingType::CONSTANT: return GetRowPtr<PaddingType::CONSTANT>(y);
        }
        return GetRowPtr<PaddingType::REPLICA>(y);
    }

#ifdef RISCV_QEMU
    // PGM/PPM file I/O. Only available on qemu-user (hosted libc).
    // Read accepts ASCII (P2/P3) and binary (P5/P6). Write always binary.
    void Read(const char* path);
    void Write(const char* path) const;
#else
    void Read(int new_width, int new_height, const uint8_t* buffer);
#endif

    int Width() const noexcept { return width; }
    int Height() const noexcept { return height; }
};