//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2016 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#include "FragmentData.h"

#include "cuda/fragments.h"
#include "cuda/timer.h"

#include <osg/Image>
#include <osgDB/WriteFile>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <string.h>

using namespace bbp::rtneuron::core::cuda;

namespace cpu
{

void countsToOffsets(FragmentData& data)
{
    Timer timer;
    size_t size = data.width * data.height + 1;
    data.offsets.reset(new uint32_t[size + 1]);
    uint32_t* offsets = data.offsets.get();
    offsets[0] = 0;
    for (size_t i = 0; i != size; ++i)
        offsets[i + 1] = offsets[i] + data.counts[i];

}

void compact(FragmentData& data)
{
    Timer timer;
    data.fragments.reset(new char[data.items * sizeof(uint32_t) * 2]);
    uint32_t* outDepths = (uint32_t*)data.fragments.get();
    uint32_t* outColors = (uint32_t*)data.fragments.get() + data.items;

    const uint32_t* rawFragments = (const uint32_t*)data.rawFragments.get();

    const size_t pixels = data.width * data.height;

    #pragma omp parallel for
    for (size_t i = 0; i < pixels; ++i)
    {
        size_t offset = data.offsets[i];
        size_t counts = data.offsets[i + 1] - offset;
        if (counts && data.heads[i] == 0xFFFFFFFF)
        {
            std::cerr << "Inconsistency between offsets and head "
                         "pointers found" << std::endl;
            exit(-1);
        }
        for (uint32_t next = data.heads[i]; next != 0xFFFFFFFF; ++offset)
        {
            next *= 3;
            outDepths[offset] = rawFragments[next + 1];
            outColors[offset] = rawFragments[next + 2];
            next = rawFragments[next];
        }
    }
}

void sort(FragmentData& data, float alphaThreshold)
{
    Timer timer;

    const size_t pixels = data.width * data.height;

    const uint32_t* offsets = data.offsets.get();
    char* compacted = data.fragments.get();

    const float inv255 = 1 / 255.0;

    bool countsUpdated = false;

    #pragma omp parallel for
    for (size_t i = 0; i < pixels; ++i)
    {
        size_t offset = offsets[i];
        size_t count = offsets[i + 1] - offset;
        if (count < 2)
            continue;

        float* depths = (float*)compacted + offset;
        uint32_t* colors = (uint32_t*)compacted + offset + data.items;

        std::vector<uint32_t> indices(count, 0);
        for (size_t k = 0; k != count; ++k)
            indices[k] = k;
        std::sort(indices.begin(), indices.end(),
                  [depths](uint32_t a, uint32_t b)
                      { return depths[a] < depths[b]; });

        /* Copying all the pixel fragment data to temporary arrays before
           changing modifying the counts. */
        std::vector<float> d(depths, depths + count);
        std::vector<uint32_t> c(colors, colors + count);

        /* Checking if we can remove some fragments from the list due to
           alpha accumulation. */
        if (alphaThreshold != 0 && count <= 64)
        {
            float transparency = 1;
            for (size_t k = 0; k < count; ++k )
            {
                transparency *= 1 - (colors[indices[k]] >> 24) * inv255;
                if (transparency <= 1 - alphaThreshold)
                {
                    data.counts[i] = k + 1;
                    count = k + 1;
                    countsUpdated = true;
                    break;
                }
            }
        }
        for (size_t k = 0; k != count; ++k)
        {
            const uint32_t index = indices[k];
            depths[k] = d[index];
            colors[k] = c[index];
        }
    }

    if (countsUpdated)
    {
        /* Computing the new offsets and compacting the fragment buffers */
        const auto oldOffsets = std::move(data.offsets);
        std::cout << "  Recomputing offsets:";
        countsToOffsets(data);
        offsets = data.offsets.get();
        const size_t oldItemCount = data.items;
        data.items = data.offsets[pixels];

        const auto oldFragments = std::move(data.fragments);
        const size_t size = data.items * sizeof(uint32_t) * 2;
        data.fragments.reset(new char[size]);
        compacted = data.fragments.get();

        #pragma omp parallel for
        for (size_t i = 0; i < pixels; ++i)
        {
            const size_t oldOffset = oldOffsets[i];
            const size_t offset = offsets[i];
            const size_t count = offsets[i + 1] - offset;

            float* depths = (float*)compacted + offset;
            uint32_t* colors = (uint32_t*)compacted + offset + data.items;
            const float* oldDepths =
                (const float*)oldFragments.get() + oldOffset;
            const uint32_t* oldColors =
                (const uint32_t*)oldFragments.get() + oldOffset + oldItemCount;

            for (size_t k = 0; k < count; ++k)
            {
                depths[k] = oldDepths[k];
                colors[k] = oldColors[k];
            }
        }

        const float oldSize = oldItemCount * sizeof(uint32_t) * 2 / 1024./1024.;
        float newSize = size / 1024./1024.;
        std::cout << "  Fragment buffer compacted from " << oldSize << " MB to "
                  << newSize << " MB (" << newSize/oldSize << "), "
                  << oldItemCount << " " << data.items
                  << std::endl;
    }
}

charsPtr mergeAndBlend(const FragmentData& data)
{
    Timer timer;

    const size_t pixels = data.width * data.height;
    charsPtr out(new char[pixels * 4]);

    const uint32_t* colors = (uint32_t*)data.fragments.get() + data.items;

    #pragma omp parallel for
    for (size_t i = 0; i < pixels; ++i)
    {
        char* pixel = out.get() + i * 4;
        const size_t offset = data.offsets[i];
        const size_t count = data.offsets[i + 1] - offset;
        osg::Vec4 color(0, 0, 0, 0);
        for (size_t k = 0; k != count; ++k)
        {
            const uint32_t raw = colors[offset + k];
            const float alpha = (raw >> 24) / 255.0;
            const osg::Vec3 rgb(raw & 0xff, (raw >> 8) & 0xff,
                                (raw >> 16) & 0xff);
            color += osg::Vec4(rgb / 255.0 * alpha, alpha) * (1 - color[3]);
        }
        color *= 255.0;
        const uint32_t alpha = round(color[3]);
        if (alpha == 0)
        {
            for (size_t c = 0; c != 4; ++c)
                pixel[c] = 0;
        }
        else
        {
            pixel[3] = round(alpha);
            for (size_t c = 0; c != 3; ++c)
                pixel[c] = round(color[c]);
        }
    }
    return out;
}

charsPtr mergeAndBlend(const std::vector<FragmentData>& data)
{
    if (data.empty())
        return charsPtr();

    if (data.size() == 1)
        return mergeAndBlend(data[0]);

    Timer timer;

    const size_t pixels = data[0].width * data[0].height;
    charsPtr out(new char[pixels * 4]);

#ifndef NDEBUG
    for (const auto& i : data)
        assert(i.width * i.height == pixels);
#endif

    #pragma omp parallel for
    for (size_t i = 0; i < pixels; ++i)
    {
        std::vector<float*> depths;
        std::vector<uint32_t*> colors;
        std::vector<size_t> counts;
        depths.reserve(data.size());
        colors.reserve(data.size());
        counts.reserve(data.size());
        size_t total = 0;
        for (size_t c = 0; c != data.size(); ++c)
        {
            const auto& piece = data[c];
            const size_t offset = piece.offsets[i];
            const size_t count = piece.offsets[i + 1] - offset;
            if (count == 0)
                continue;
            counts.push_back(count);
            total += count;
            uint32_t* fragment = (uint32_t*)piece.fragments.get() + offset;
            depths.push_back((float*)fragment);
            colors.push_back(fragment + piece.items);
        }

        char* pixel = out.get() + i * 4;
        osg::Vec4 color(0, 0, 0, 0);
        /* Front-to-back data.size()-way merge of the fragments */
        for (size_t j = 0; j != total; ++j)
        {
            /* Picking the closes fragment first */
            float closest = 1.0;
            size_t index = 0;
            for (size_t c = 0; c != counts.size(); ++c)
            {
                const float depth = *depths[c];
                if (depth < closest)
                {
                    closest = depth;
                    index = c;
                }
            }
            uint32_t raw = *colors[index];

            /* Advancing to the next fragment in the list from where it was
               picked. */
            if (--counts[index] == 0)
            {
                depths.erase(depths.begin() + index);
                colors.erase(colors.begin() + index);
                counts.erase(counts.begin() + index);
            }
            else
            {
                ++depths[index];
                ++colors[index];
            }

            /* Blending */
            const float alpha = (raw >> 24) / 255.0;
            const osg::Vec3 rgb(raw & 0xff, (raw >> 8) & 0xff,
                                (raw >> 16) & 0xff);
            color += osg::Vec4(rgb / 255.0 * alpha, alpha) * (1 - color[3]);
        }
        assert(counts.empty());
        color *= 255.0;
        const uint32_t alpha = round(color[3]);
        if (alpha == 0)
        {
            for (size_t c = 0; c != 4; ++c)
                pixel[c] = 0;
        }
        else
        {
            pixel[3] = round(alpha);
            for (size_t c = 0; c != 3; ++c)
                pixel[c] = round(color[c]);
        }
    }
    return out;
}

}

namespace gpu
{

void countsToOffsets(DeviceData& data)
{
    void* tmp = 0;
    size_t tmpSize = 0;
    {
        Timer timer(0, true);
        uint32_t* output = 0;
        const cudaError_t error = fragmentCountsToOffsets(
            data.width, data.height, 0, 0, data.counts, output, tmp, tmpSize);
        if (error != cudaSuccess)
        {
            std::cerr << "Error in fragmentCountsToOffsets: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
        data.offsets = uint32_tsDevPtr(output, (Deleter)cudaFree);
     }
    cudaFree(tmp);
}

void compact(DeviceData& data)
{
    Timer timer(0, true);
    void* compacted = 0;
    size_t size;
    cudaError_t error;
    error = compactFragmentLists(
        data.width, data.height, 0, 0, data.offsets.get(), data.heads,
        (uint32_t*)data.rawFragments.get(), compacted, size);

    if (error != cudaSuccess)
    {
        std::cerr << "Error in compactFragmentLists: "
                  << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    if (size != data.items * sizeof(uint32_t) * 2)
    {
        std::cerr << "Invalid GPU output buffer size" << std::endl;
        exit(-1);
    }
    data.fragments = charsDevPtr((char*)compacted, (Deleter)cudaFree);
}

void sort(DeviceData& data, float alphaThreshold)
{
    {
        Timer timer(0, true);
        cudaError_t error;

        if (alphaThreshold == 0)
        {
            error = sortFragmentLists(
                data.fragments.get(), data.items, data.width * data.height,
                data.offsets.get());
        }
        else
        {

            void* tmp = 0;
            error = sortFragmentLists(
                data.fragments.get(), data.items, data.width * data.height,
                data.offsets.get(), alphaThreshold, tmp);
            if (tmp != 0)
                cudaFree(tmp);
        }
        if (error != cudaSuccess)
        {
            std::cerr << "Error in sortFragmentLists: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }
    cudaStreamSynchronize(0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "Error in sortFragmentLists: "
                  << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    if (alphaThreshold != 0)
    {
        /* Updating total fragment count. */
        error = cudaMemcpy(
            &data.items, data.offsets.get() + data.width * data.height,
            sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            std::cerr << "Error updating item count: "
                      << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }
}

charsDevPtr mergeAndBlend(const std::vector<DeviceData>& data)
{
    if (data.empty())
        return charsDevPtr(0, (Deleter)cudaFree);

    const size_t pixels = data[0].width * data[0].height;
#ifndef NDEBUG
    for (const auto& i : data)
        assert(i.width * i.height == pixels);
#endif

    void* output = 0;
    {
        Timer timer(0, true);
        std::vector<const float*> depths;
        for (const auto& i : data)
            depths.push_back((const float*)i.fragments.get());
        std::vector<const uint32_t*> colors;
        for (const auto& i : data)
            colors.push_back((const uint32_t*)i.fragments.get() + i.items);
        std::vector<const uint32_t*> offsets;
        for (const auto& i : data)
            offsets.push_back(i.offsets.get());

        const cudaError_t error =
            mergeAndBlendFragments(depths, colors, offsets, 0, pixels, output);
        if (error != cudaSuccess)
        {
            std::cerr << "Error in mergeAndBlend: "
                          << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }
    cudaStreamSynchronize(0);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "Error in mergeAndBlend: "
                  << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    return charsDevPtr((char*)output, (Deleter)cudaFree);
}

}

void validate(const FragmentData& data)
{
    const size_t pixels = data.width * data.height;
    for (size_t i = 0; i < pixels; ++i)
    {
        const size_t offset = data.offsets[i];
        const size_t count = data.offsets[i + 1] - offset;
        if (count < 2)
            continue;
        float* depths = (float*)data.fragments.get() + offset;
        for (size_t k = 1; k != count; ++k)
        {
            if (depths[k - 1] > depths[k])
            {
                std::cout << "Out of order fragment: "
                          << i << " " << k << std::endl;
            }
        }
    }
}

template<typename T>
bool compare(const T* host, const T* device,  const size_t size)
{
    std::unique_ptr<T[]> copy(new T[size]);
    cudaError_t error = cudaMemcpy(copy.get(), device, size * sizeof(T),
                                   cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cerr << "Error copying data from device to host: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return memcmp(copy.get(), host, size * sizeof(T)) == 0;
}

bool compareCompacted(const FragmentData& data, const DeviceData& devData)
{
    const uint32_t* offsets = data.offsets.get();
    const char* host = data.fragments.get();
    const char* device = devData.fragments.get();
    const size_t pixels = data.width * data.height;
    const size_t size = data.items * 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> copy(new char[size]);
    cudaError_t error =
        cudaMemcpy(copy.get(), device, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cerr << "Error copying data from device to host: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    for (size_t i = 0; i != pixels; i++)
    {
        const size_t offset = offsets[i];
        const size_t count = offsets[i + 1] - offset;
        const float* d = (const float*)host + offset;
        const float* d2 = (const float*)copy.get() + offset;
        const uint32_t* c = (const uint32_t*)host + offset + data.items;
        const uint32_t* c2 = (const uint32_t*)copy.get() + offset + data.items;

        if (count == 0)
            continue;

        if (count == 1)
        {
            if (*d != *d2 || *c != *c2)
                return false;
            continue;
        }

        for (size_t j = 0; j != count; ++j)
        {
            bool found = false;
            for (size_t k = 0; k != count && !found; ++k)
            {
                if (d[j] == d2[k] && c[j] == c2[k])
                    found = true;
            }
            if (!found)
                return false;
        }
    }
    return true;
}

bool isBelowThreshold(const float transparency, const float alphaThreshold)
{
    /* The CPU and the GPU compute the current alpha accumulation doing the
       operations in different order. Since ensuring the CPU does exactly the
       same thing is tedious and not very robust, we add a small delta to
       this comparison. */
    const float minTransp = 1 - alphaThreshold;
    return transparency - minTransp < minTransp * 0.000003;
}

bool comparePixelLists(const float* d1, const uint32_t* c1, const size_t count1,
                       const float* d2, const uint32_t* c2, const size_t count2,
                       const float alphaThreshold)
{
    if (count1 == 0 && count2 == 0)
        return true;

    size_t count = count1;
    bool thresholdReached = false;
    if (alphaThreshold != 0)
    {
        float transp1 = 1;
        for (size_t i = 0; i < count1; ++i)
            transp1 *= 1 - (c1[i] >> 24) / 255.0;

        float transp2 = 1;
        size_t fragmentsPassThreshold = 0;
        for (size_t i = 0; i < count2; ++i)
        {
            transp2 *= 1 - (c2[i] >> 24) / 255.0;
            if (isBelowThreshold(transp2, alphaThreshold))
            {
                if (++fragmentsPassThreshold > 1 && i % 2 == 0)
                    /* cut-off didn't worked in the GPU. */
                    return false;
            }
        }

        thresholdReached = isBelowThreshold(transp1, alphaThreshold);
        if (thresholdReached != isBelowThreshold(transp2, alphaThreshold))
            return false;

        if (thresholdReached)
        {
            /* Comparing the end of the lists in this case is really hard due to
               the sorting unstabilities and the optimization in the GPU
               algorithm for lists between 33 and 64 fragments that only allows
               cutting off the fragments lists to even lengths.
               For that reason we just focus on comparins the lists up to the
               shortest of both. */
            count = std::min(count1, count2);
        }
        else
        {
            /* If threshold was not reached the lists must have the same
               length. */
            if (count1 != count2)
                return false;
        }
    }
    else
    {
        if (count1 != count2)
            return false;
    }

    size_t blockLength;
    for (size_t j = 0; j < count; j += blockLength)
    {
        blockLength = 1;
        /* Find out the the run length of fragments with the same depth. */
        for (size_t k = j; k < count - 1 && d1[k] == d1[k + 1];
             ++k, ++blockLength)
            ;

        const bool lastBlock = j + blockLength == count;
        if (lastBlock && thresholdReached)
        {
            /* The last block of fragment depths is special when
               alphaThreshold. It's possible to construct many examples in which
               the CPU and GPU lists are different and yet both are correct
               due to the algorithmic differences (and the undecidability of
               what's the right order for fragments with equal depths).
               Therefore we only check that the depths are the same until. */
            const float depth = d1[j];
            for (size_t k = j; k < count; ++k)
                if (d2[k] != depth)
                    return false;
        }
        else
        {
            const float depth = d1[j];
            for (size_t k = 0; k < blockLength; ++k)
                if (d2[j + k] != depth)
                    return false;

            /* The colors can be out of order due to the sorting
               unstabilities. */
            std::set<uint32_t> colors1, colors2;
            for (size_t k = 0; k < blockLength; ++k)
            {
                colors1.insert(c1[j + k]);
                colors2.insert(c2[j + k]);
            }
            if (colors1 != colors2)
                return false;
        }
    }
    return true;
}

/* This comparison is special because we are not requiring the sorting
   algorithm on the GPU to be a stable sort. */
bool compareFinal(const FragmentData& data, const DeviceData& devData,
                  const float alphaThreshold = 0, const size_t maxCount = -1)
{
    const uint32_t* offsets1 = data.offsets.get();
    const char* sorted = data.fragments.get();
    const size_t pixels = data.width * data.height;
    std::unique_ptr<uint32_t[]> offsets2(new uint32_t[pixels + 1]);
    cudaError_t error =
        cudaMemcpy(offsets2.get(), devData.offsets.get(),
                   sizeof(uint32_t) * (pixels + 1), cudaMemcpyDeviceToHost);
    const size_t size = devData.items * 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> copy(new char[size]);
    if (error == cudaSuccess)
        error = cudaMemcpy(copy.get(), devData.fragments.get(),
                           size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cerr << "Error copying data from device to host: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    bool good = true;
    #pragma omp parallel for shared(good)
    for (size_t i = 0; i < pixels; i++)
    {
        if (!good)
            continue;

        const size_t offset1 = offsets1[i];
        const size_t count1 = offsets1[i + 1] - offset1;
        const size_t offset2 = offsets2[i];
        const size_t count2 = offsets2[i + 1] - offset2;
        const float* d1 = (const float*)sorted + offset1;
        const float* d2 = (const float*)copy.get() + offset2;
        const uint32_t* c1 = (const uint32_t*)sorted + offset1 + data.items;
        const uint32_t* c2 =
            (const uint32_t*)copy.get() + offset2 + devData.items;

        if (count1 > maxCount && count2 > maxCount)
            continue;

        if (!comparePixelLists(d1, c1, count1, d2, c2, count2, alphaThreshold))
        {
            #pragma omp critical
            {
                std::cout << std::setprecision(
                    std::numeric_limits<float>::digits10 + 1);
                std::cout << i << " " << count1 << " " << count2 << std::endl;
                float t1 = 1, t2 = 1;
                for (size_t j = 0, k = 0; j < count1 || k < count2; ++j, ++k)
                {
                    if (j < count1)
                    {
                        t1 *= 1 - (c1[j] >> 24) / 255.0;
                        std::cout << d1[j] << " " << c1[j] << " " << t1 << " ";
                    }
                    else
                        std::cout << "                          ";
                    if (k < count2)
                    {
                        t2 *= 1 - (c2[j] >> 24) / 255.0;
                        std::cout << d2[k] << " " << c2[k] << " " << t2
                                  << std::endl;
                    }
                    else
                        std::cout << std::endl;
                }
            }
            good = false;
        }
    }

    return good;
}

void writeImage(const std::string& filename,
                const char* data, const size_t width, const size_t height)
{
    osg::ref_ptr<osg::Image> image(new osg::Image());

    /* Undoing alpha multiplication. */
    uint32_t* out = new uint32_t[width * height];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < width * height; ++i)
    {
        const uint32_t pixel = ((uint32_t*)data)[i];
        const uint32_t alpha = pixel >> 24;
        if (alpha != 0)
        {
            const uint32_t blue = (pixel >> 16) & 0xff;
            const uint32_t green = (pixel >> 8) & 0xff;
            const uint32_t red = pixel & 0xff;
            out[i] = (( alpha << 24 ) |
                     (((255 * blue) / alpha ) << 16 ) |
                     (((255 * green) / alpha )  << 8 ) |
                     ((255 * red) / alpha ));
        }
        else
            out[i] = 0;
    }

    image->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                    (unsigned char*)out, osg::Image::USE_NEW_DELETE);
    osgDB::writeImageFile(*image, filename);
}

void compactFragments(FragmentData& data, DeviceData& devData,
                      const float alphaThreshold)
{
    std::cout << "Computing offsets (CPU)" << std::endl;
    cpu::countsToOffsets(data);

    std::cout << "Computing offsets (GPU)" << std::endl;
    gpu::countsToOffsets(devData);
    std::cout << "Validating: ";
    if (!compare(data.offsets.get(), devData.offsets.get(),
                 data.width * data.height + 1))
    {
        std::cerr << "Bad!" << std::endl;
        exit(-1);
    }
    std::cout << "OK!" << std::endl;

    std::cout << "Compacting fragments (CPU)" << std::endl;
    cpu::compact(data);

    std::cout << "Compacting fragments (GPU)" << std::endl;
    gpu::compact(devData);
    std::cout << "Validating: ";
    if (!compareCompacted(data, devData))
    {
        std::cerr << "Bad!" << std::endl;
        exit(-1);
    }
    std::cout << "OK!" << std::endl;

    std::cout << "Sorting fragments (CPU)" << std::endl;
    cpu::sort(data, alphaThreshold);

    std::cout << "Sorting fragments (GPU)" << std::endl;
    gpu::sort(devData, alphaThreshold);

    std::cout << "Validating: ";
    if (!compareFinal(data, devData, alphaThreshold, 1000))
    {
        std::cerr << "Bad!" << std::endl;
        exit(-1);
    }
    std::cout << "OK!" << std::endl;
}

void mergeFragments(const std::vector<FragmentData>& data,
                    const std::vector<DeviceData>& devData,
                    const std::string& output)
{
    std::cout << "Merging and blending fragments (CPU)" << std::endl;
    charsPtr host = cpu::mergeAndBlend(data);
    std::cout << "Merging and blending fragments (GPU)" << std::endl;
    charsDevPtr device = gpu::mergeAndBlend(devData);

    if (!output.empty())
    {
        const size_t width = data[0].width;
        const size_t height = data[0].height;

        const size_t size = width * height;
        charsPtr copy(new char[size * 4]);
        cudaMemcpy(copy.get(), device.get(), size * 4, cudaMemcpyDeviceToHost);

        size_t pos = output.find_last_of(".");
        std::string ext = pos != std::string::npos ? output.substr(pos) : "";
        std::string outputGPU = output.substr(0, pos) + ".gpu" + ext;

        writeImage(output, host.get(), width, height);
        writeImage(outputGPU, copy.get(), width, height);
    }
}

int main(int argc, char* argv[])
{
    bool error = false;
    std::vector<std::string> inputFiles;
    std::string outputFile;
    bool alphaAware = false;
    for (int i = 1; i != argc; ++i)
    {
        if (!strcmp(argv[i], "--out"))
        {
            if (i == argc - 1)
                error = true;
            else
                outputFile = argv[++i];
        }
        else if (!strcmp(argv[i], "--alpha-aware"))
        {
            alphaAware = true;
        }
        else
            inputFiles.push_back(argv[i]);
    }
    error = error | inputFiles.empty();
    if (error)
    {
        std::cerr << "Usage: fragment_operations [--out image] datafile [...]"
                  << std::endl;
        return -1;
    }

    cudaSetDevice(0);

    std::vector<FragmentData> cpuFragments;
    std::vector<DeviceData> gpuFragments;
    for (auto file : inputFiles)
    {
        std::cout << "Processing file " << file << std::endl;
        cpuFragments.emplace_back(FragmentData(file));
        FragmentData& fragments = cpuFragments.back();
        uint32_t count = 0;
        for (size_t i = 0; i != fragments.width * fragments.height; ++i)
            count = std::max(count, fragments.counts[i]);
        std::cout << "Width " << fragments.width
                  << ", height " << fragments.height
                  << ", fragments " << fragments.items
                  << ", longest list " << count << std::endl;

        gpuFragments.emplace_back(DeviceData(fragments));
        compactFragments(cpuFragments.back(), gpuFragments.back(),
                         alphaAware ? 0.99 : 0);
    }

    mergeFragments(cpuFragments, gpuFragments, outputFile);
}
