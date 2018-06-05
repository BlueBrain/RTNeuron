//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2016 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

typedef std::unique_ptr<uint32_t[]> uint32_tsPtr;
typedef std::unique_ptr<char[]> charsPtr;

typedef void(*Deleter)(void*);
typedef std::unique_ptr<uint32_t[], Deleter> uint32_tsDevPtr;
typedef std::unique_ptr<char, Deleter> charsDevPtr;


class FragmentData
{
public:
    uint32_t width;
    uint32_t height;
    uint32_t items;
    uint32_tsPtr counts;
    uint32_tsPtr heads;
    charsPtr rawFragments;
    uint32_tsPtr offsets;
    charsPtr fragments;

    FragmentData() {}

    FragmentData(const std::string& filename)
    {
        std::ifstream input(filename, std::ios::binary);
        if (!input)
        {
            std::cerr << "Could not open data file: " << filename << std::endl;
            exit(-1);
        }

        std::streampos totalSize = input.tellg();
        input.seekg(0, std::ios::end);
        totalSize = input.tellg() - totalSize;
        input.seekg(0, std::ios::beg);

        input.read((char*)&width, sizeof(uint32_t));
        input.read((char*)&height, sizeof(uint32_t));
        input.read((char*)&items, sizeof(uint32_t));

        size_t size = width * height;
        counts.reset(new uint32_t[size]);
        input.read((char*)counts.get(), size * sizeof(uint32_t));

        heads.reset(new uint32_t[size]);
        input.read((char*)heads.get(), size * sizeof(uint32_t));

        size = items * sizeof(uint32_t) * 3;
        rawFragments.reset(new char[size]);
        input.read((char*)rawFragments.get(), size);

        if (input.fail())
        {
            std::cerr << "Error reading data file: " << filename << std::endl;
            exit(-1);
        }
    }
};

struct DeviceData
{
    DeviceData(DeviceData&& other)
        : width(other.width)
        , height(other.height)
        , items(other.items)
        , counts(other.counts)
        , heads(other.heads)
        , rawFragments(std::move(other.rawFragments))
        , offsets(std::move(other.offsets))
        , fragments(std::move(other.fragments))
    {
        other.counts = 0;
        other.heads = 0;
    }

    explicit DeviceData(const FragmentData& data)
        : width(data.width)
        , height(data.height)
        , items(data.items)
        , counts(0)
        , heads(0)
        , rawFragments(nullptr, (Deleter)cudaFree)
        , offsets(nullptr, (Deleter)cudaFree)
        , fragments(nullptr, (Deleter)cudaFree)
    {
        size_t size = width * height * sizeof(uint32_t);
        cudaChannelFormatDesc desc{32, 0, 0, 0, cudaChannelFormatKindUnsigned};
        cudaError_t error = cudaMallocArray(&counts, &desc, width, height);
        if (error != cudaSuccess)
        {
            std::cerr << "Error allocating counts array: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
        cudaMemcpyToArray(
            counts, 0, 0, data.counts.get(), size, cudaMemcpyHostToDevice);

        error = cudaMallocArray(&heads, &desc, width, height);
        if (error != cudaSuccess)
        {
            std::cerr << "Error allocating heads array: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
        cudaMemcpyToArray(
            heads, 0, 0, data.heads.get(), size, cudaMemcpyHostToDevice);

        size = data.items * sizeof(uint32_t) * 3;
        char* fragmentsPtr;
        error = cudaMalloc(&fragmentsPtr, size);
        if (error != cudaSuccess)
        {
            std::cerr << "Error allocating fragments buffer: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
        rawFragments = charsDevPtr(fragmentsPtr, (Deleter)cudaFree);

        error = cudaMemcpy(fragmentsPtr, data.rawFragments.get(), size,
                           cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            std::cerr << "Error copying fragments to GPU: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }

    ~DeviceData()
    {
        cudaFreeArray(counts);
        cudaFreeArray(heads);
    }

    uint32_t width;
    uint32_t height;
    uint32_t items;
    cudaArray_t counts;
    cudaArray_t heads;
    charsDevPtr rawFragments;
    uint32_tsDevPtr offsets;
    charsDevPtr fragments;
};

