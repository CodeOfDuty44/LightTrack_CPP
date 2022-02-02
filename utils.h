#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <vector>
#include "assert.h"


//Tries to read engine file from the disk, returns empty string if engine file doesn't exist
std::string readEngine(std::string const& path);

//Reads image in PGM format from disk and stores it into the buffer
void readPGMImage(const std::string& fileName,  uint8_t *buffer, int inH, int inW);

//Converts image in NHWC format into image in NCHW format
void HWC_to_CHW(uint8_t * src, uint8_t * dst, int h, int w, int chnls);

//Puts softmax layer's outputs in order
template <typename T>
std::vector<std::size_t> order_probs(const std::vector<T>& v)
{
	std::vector<std::size_t> indices(v.size());
	std::iota(indices.begin(), indices.end(), 0u);
	std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return v[lhs] < v[rhs];
    });
    std::vector<std::size_t> result(v.size());
    for (std::size_t i = 0; i != indices.size(); ++i) {
    	result[indices[i]] = i;
    }
    return result;
}

#endif
