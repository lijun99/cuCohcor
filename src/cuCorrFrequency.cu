/*
 * @file  cuCorrFrequency.cu
 * @brief A class performs cross correlation in frequency domain
 */

#include "cuCorrFrequency.h"
#include "cuAmpcorUtil.h"

/*
 * cuFreqCorrelator Constructor
 * @param imageNX height of each image
 * @param imageNY width of each image
 * @param nImages number of images in the batch
 * @param stream CUDA stream
 */
cuFreqCorrelator::cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_)
{

    int imageSize = imageNX*imageNY;
    int n[NRANK] ={imageNX, imageNY};

    // set up fft plans
    cufft_Error(cufftPlanMany(&fftPlan, NRANK, n,
                              NULL, 1, imageSize,
                              NULL, 1, imageSize,
                              CUFFT_C2C, nImages));

    stream = stream_;
    cufftSetStream(fftPlan, stream);

    // set up work arrays
    workFM = new cuArrays<float2>(imageNX, imageNY, nImages);
    workFM->allocate();
    workFS = new cuArrays<float2>(imageNX, imageNY, nImages);
    workFS->allocate();
    workT = new cuArrays<float2> (imageNX, imageNY, nImages);
    workT->allocate();
}

/// destructor
cuFreqCorrelator::~cuFreqCorrelator()
{
    cufft_Error(cufftDestroy(fftPlan));
    workFM->deallocate();
    workFS->deallocate();
    workT->deallocate();
}


/**
 * Execute the cross correlation
 * @param[in] templates the reference windows
 * @param[in] images the search windows
 * @param[out] results the correlation surfaces
 */

void cuFreqCorrelator::execute(cuArrays<float2> *templates, cuArrays<float2> *images, cuArrays<float> *results)
{
    // pad the reference windows to the the size of search windows
    // note: cuArraysCopyPadded has both float and float2 implementations
    cuArraysCopyPadded(templates, workT, stream);
    // forward fft to frequency domain
    cufft_Error(cufftExecC2C(fftPlan, workT->devData, workFM->devData, CUFFT_FORWARD));
    cufft_Error(cufftExecC2C(fftPlan, images->devData, workFS->devData, CUFFT_FORWARD));
    // cufft doesn't normalize, so manually get the image size for normalization
    float coef = 1.0/(images->size);
    // multiply reference with secondary windows in frequency domain
    cuArraysElementMultiplyConjugate(workFM, workFS, coef, stream);
    // backward fft to get correlation surface in time domain
    cufft_Error(cufftExecC2C(fftPlan, workFM->devData, workT->devData, CUFFT_INVERSE));
    // extract to get proper size of correlation surface
    cuArraysCopyExtractC2A(workT, results, make_int2(0, 0), stream);
    // all done
}

// a = a^* * b
inline __device__ float2 cuMulConj(float2 a, float2 b)
{
    return make_float2(a.x*b.x + a.y*b.y, -a.y*b.x + a.x*b.y);
}

// cuda kernel for cuArraysElementMultiplyConjugate
__global__ void cudaKernel_elementMulConjugate(float2 *ainout, float2 *bin, int size, float coef)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        cuComplex prod;
        prod = cuMulConj(ainout[idx], bin[idx]);
        ainout [idx] = prod*coef;
    }
}

/**
 * Perform multiplication of coef*Conjugate[image1]*image2 for each element
 * @param[inout] image1, the first image
 * @param[in] image2, the secondary image
 * @param[in] coef, usually the normalization factor
 */
void cuArraysElementMultiplyConjugate(cuArrays<float2> *image1, cuArrays<float2> *image2, float coef, cudaStream_t stream)
{
    int size = image1->getSize();
    int threadsperblock = NTHREADS;
    int blockspergrid = IDIVUP (size, threadsperblock);
    cudaKernel_elementMulConjugate<<<blockspergrid, threadsperblock, 0, stream>>>(image1->devData, image2->devData, size, coef );
    getLastCudaError("cuArraysElementMultiply error\n");
}
//end of file
