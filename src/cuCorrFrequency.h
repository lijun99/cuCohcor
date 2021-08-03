/*
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain
 */

// code guard
#ifndef __CUCORRFREQUENCY_H
#define __CUCORRFREQUENCY_H

// dependencies
#include "cudaUtil.h"
#include "cuArrays.h"

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    cufftHandle fftPlan;
    // work data
    cuArrays<float2> *workFM;
    cuArrays<float2> *workFS;
    cuArrays<float2> *workT;
    // cuda stream
    cudaStream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(cuArrays<float2> *templates, cuArrays<float2> *images, cuArrays<float> *results);
};

#endif //__CUCORRFREQUENCY_H
// end of file