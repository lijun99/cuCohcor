/*
 * @file cuWindowFilter.cu
 * @brief Window Function filter
 *
 */

#include "cuWindowFilter.h"
#include "cuAmpcorUtil.h"

// Hann Window Filter

// forward declaration of kernels
__global__ void _init_2d_hann_coef_kernel(float *coef, const int NX, const int NY);
__global__ void _hann_filtering_kernel(float2 *images, const float *coef, const int NX, const int NY);

// constructor
cuHannWindowFilter::cuHannWindowFilter(int NX, int NY)
{
    // allocate the coefficient
    hann_coef = new cuArrays<float>(NX, NY, 1);
    hann_coef ->allocate();
    // initialize the coefficients
    init_coef();
}

// destructor
cuHannWindowFilter::~cuHannWindowFilter()
{
    delete hann_coef;
}

// initialize Hanning window coefficients
void cuHannWindowFilter::init_coef()
{
    const int NX = hann_coef->height;
    const int NY = hann_coef->width;
    int nthreads = NTHREADS2D;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(NX,nthreads), IDIVUP(NY,nthreads), 1);
    // call the kernel
    _init_2d_hann_coef_kernel<<<gridSize, blockSize>>>(hann_coef->devData, NX, NY);
    getLastCudaError("cuHannWindowFilter_init_2d_hann_coef_kernel error");
}

void cuHannWindowFilter::filter(cuArrays<float2> * images, cudaStream_t stream)
{
    const int NX = images->height;
    const int NY = images->width;
    int nthreads = NTHREADS2D;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(NX,nthreads), IDIVUP(NY,nthreads), images->count);
    // call the kernel
    _hann_filtering_kernel<<<gridSize, blockSize, 0, stream>>>(images->devData, hann_coef->devData, NX, NY);
    getLastCudaError("cuHannWindowFilter__hann_filtering_kernel error");
}


__global__ void _init_2d_hann_coef_kernel(float *coef, const int NX, const int NY)
{
    // get the thread id as pixel in correlation surface
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    int ty = threadIdx.y + blockDim.y*blockIdx.y;
    // check the range
    if (tx < NX && ty < NY) {
        // Hann Window Function 1/2(1-cos(2Pin/(N-1)).
        coef[tx*NY + ty] = 0.5*(1.-cos(2.*PI*tx/(NX-1)))*0.5*(1.-cos(2.*PI*ty/(NY-1)));
    }
}

__global__ void _hann_filtering_kernel(float2 *images, const float *coef, const int NX, const int NY)
{
    // get pixel x-coordinate (along height)
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    // get pixel y-coordinate (along width)
    int ty = threadIdx.y + blockDim.y*blockIdx.y;
    if(tx < NX && ty < NY)
    {
        //pixel index in a 2d image
        int pixelIdx = tx*NY + ty;
        //pixel index in the batched 2d images
        int imageIdx = blockIdx.z*NX*NY + pixelIdx;
        // multiply by the window coefficients
        images[imageIdx] *= coef[pixelIdx];
    }
}

// end of file
