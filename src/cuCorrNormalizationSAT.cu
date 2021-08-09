/*
 * @file cuCorrNormalizationSAT.cu
 * @brief Utilities to normalize the 2D correlation surface with the sum area table
 *
 */

#include <cooperative_groups.h>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cooperative_groups/reduce.h>
#endif

// my declarations
#include "cuAmpcorUtil.h"
// for FLT_EPSILON
#include <float.h>

// alias for cuda cooperative groups
namespace cg = cooperative_groups;


/**
 * cuda kernel for sum value^2 (std)
 * compute the sum value square (std) of the reference image
 * @param[out] sum2 sum of value square
 * @param[in] images the reference images
 * @param[in] n total elements in one image nx*ny
 * @param[in] batch number of images
 * @note use one thread block for each image, blockIdx.x is image index
 **/


#if __CUDACC_VER_MAJOR__ >= 11
// use cg::reduce for NVCC 11 and above
__global__ void sum_square_kernel(float *sum2, const float2 *images, int n, int batch)
{
    // get block id for each image
    int imageid = blockIdx.x;
    const float2 *image = images + imageid*n;

    // get the thread block
    cg::thread_block cta = cg::this_thread_block();
    // get the shared memory
    extern float __shared__ sdata[];

    // get the current thread
    int tid = cta.thread_rank();

    // stride over grid and add the values to shared memory
    sdata[tid] = 0;

    for(int i = tid; i < n; i += cta.size() ) {
        auto value = image[i];
        sdata[tid] += complexSquare(value);
    }

    cg::sync(cta);

    // partition thread block into tiles in size 32 (warp)
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // reduce in each tile with warp
    sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<float>());
    cg::sync(cta);

    // reduce all tiles with thread 0
    if(tid == 0) {
        float sum = 0.0;
        for (int i = 0; i < cta.size(); i += tile32.size())
            sum += sdata[i];
        // assign the value to results
        sum2[imageid] = sum;
    }
}

#else
// use warp-shuffle reduction for NVCC 9 & 10
__global__ void sum_square_kernel(float *sum2, const float2 *images, int n, int batch)
{
    // get block id for each image
    int imageid = blockIdx.x;
    const float2 *image = images + imageid*n;

    // get the thread block
    cg::thread_block cta = cg::this_thread_block();
    // get the shared memory
    extern float __shared__ sdata[];

    // get the current thread
    unsigned int tid = cta.thread_rank();
    unsigned int blockSize = cta.size();

    // stride over grid and add the values to the shared memory
    float sum = 0;

    for(int i = tid; i < n; i += blockSize ) {
        auto value = image[i];
        sum += complexSquare(value);
    }
    sdata[tid] = sum;
    cg::sync(cta);

    // do reduction in shared memory in log2 steps
    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = sum = sum + sdata[tid + 256];
    }
    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = sum = sum + sdata[tid + 128];
    }
    cg::sync(cta);

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = sum = sum + sdata[tid + 64];
    }
    cg::sync(cta);

    // partition thread block into tiles in size 32 (warp)
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // reduce within warp
    if(tid < 32) {
        if(blockSize >=64) sum += sdata[tid + 32];
        for (int offset = tile32.size()/2; offset >0; offset /=2) {
            sum += tile32.shfl_down(sum, offset);
        }
    }

    // return results with thread 0
    if(tid == 0) {
        // assign the value to results
        sum2[imageid] = sum;
    }
}
#endif // __CUDACC_VER_MAJOR check


/**
 * cuda kernel for 2d sum area table
 * Compute the (inclusive) sum area table of the value and value^2 of a batch of 2d images.
 * @param[out] sat the sum area table
 * @param[out] sat2 the sum area table of value^2
 * @param[in] data search images
 * @param[in] nx image height (subleading dimension)
 * @param[in] ny image width (leading dimension)
 * @param[in] batch number of images
 **/

__global__ void sat2d_kernel(float2 *sat, float * sat2, const float2 *data, const int nx, const int ny, const int batch)
{
    // get block id for each image
    int imageid = blockIdx.x;

    // get the thread id for each row/column
    int tid =  threadIdx.x;

    // compute prefix-sum along row at first
    // the number of rows may be bigger than the number of threads, iterate
    for (int row = tid; row < nx; row += blockDim.x) {
        // running sum for value and value^2
        float2 sum = make_float2(0.0f, 0.0f);
        float sum2 = 0.0f;
        // starting position for this row
        int index = (imageid*nx+row)*ny;
        // iterative over column
        for (int i=0; i<ny; i++, index++) {
            float2 val = data[index];
            sum += val;
            sat[index] = sum;
            sum2 += val.x*val.x + val.y*val.y;
            sat2[index] = sum2;
        }
    }

    // wait till all rows are done
    __syncthreads();

    // compute prefix-sum along column
    for (int col = tid; col < ny; col += blockDim.x) {

        // start position of the current column
        int index = col + imageid*nx*ny;

        // assign sum with the first line value
        float2 sum = sat[index];
        float sum2 = sat2[index];
    	// iterative over rest lines
    	for (int i=1; i<nx; i++) {
            index += ny;
            sum += sat[index];
            sat[index] = sum;
            sum2 += sat2[index];
            sat2[index] = sum2;
        }
    }
    // all done
}


__global__ void cuCorrNormalizeSAT_kernel(float *correlation, const float *referenceSum2, const float2 *secondarySat,
    const float *secondarySat2, const int corNX, const int corNY, const int referenceNX, const int referenceNY,
    const int secondaryNX, const int secondaryNY)
{
    //get the image id from block z index
    int imageid = blockIdx.z;

    // get the thread id as pixel in correlation surface
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    int ty = threadIdx.y + blockDim.y*blockIdx.y;
    // check the range
    if (tx < corNX && ty < corNY) {
        // get the reference std
        float refSum2 = referenceSum2[imageid];

        // compute the sum and sum square of the search image from the sum area table
        // sum
        const float2 *sat = secondarySat + imageid*secondaryNX*secondaryNY;
        // get sat values for four corners
        float2 czero = make_float2(0.0f, 0.0f);
        float2 topleft = (tx > 0 && ty > 0) ? sat[(tx-1)*secondaryNY+(ty-1)] : czero;
        float2 topright = (tx > 0 ) ? sat[(tx-1)*secondaryNY+(ty+referenceNY-1)] : czero;
        float2 bottomleft = (ty > 0) ? sat[(tx+referenceNX-1)*secondaryNY+(ty-1)] : czero;
        float2 bottomright = sat[(tx+referenceNX-1)*secondaryNY+(ty+referenceNY-1)];
        // get the sum
        float2 secondarySum = bottomright + topleft - topright - bottomleft;
        // sum of value^2
        const float *sat2 = secondarySat2 + imageid*secondaryNX*secondaryNY;
        // get sat2 values for four corners
        float topleft2 = (tx > 0 && ty > 0) ? sat2[(tx-1)*secondaryNY+(ty-1)] : 0.0f;
        float topright2 = (tx > 0 ) ? sat2[(tx-1)*secondaryNY+(ty+referenceNY-1)] : 0.0f;
        float bottomleft2 = (ty > 0) ? sat2[(tx+referenceNX-1)*secondaryNY+(ty-1)] : 0.0f;
        float bottomright2 = sat2[(tx+referenceNX-1)*secondaryNY+(ty+referenceNY-1)];
        float secondarySum2 = bottomright2 + topleft2 - topright2 - bottomleft2;

        // compute the normalization
        float norm2 = (secondarySum2-complexSquare(secondarySum)/(referenceNX*referenceNY))*refSum2;
        // in norm2, secondarySum2 and refSum2 are scaled as imageSize, to be canceled with bare correlation
        // normalize the correlation surface
        correlation[(imageid*corNX+tx)*corNY+ty] *= rsqrtf(norm2 + FLT_EPSILON);
    }
}


void cuCorrNormalizeSAT(
    cuArrays<float> *correlation,
    cuArrays<float2> *reference,
    cuArrays<float2> *secondary,
    cuArrays<float> *referenceSum2,
    cuArrays<float2> *secondarySat,
    cuArrays<float> *secondarySat2,
    cudaStream_t stream)
{
    // compute the std of reference image
    // note that the mean is already subtracted
    int nthreads = 256;
    int sMemSize = nthreads*sizeof(float);
    int nblocks = reference->count;
    sum_square_kernel<<<nblocks, nthreads, sMemSize, stream>>>(referenceSum2->devData, reference->devData,
        reference->width * reference->height, reference->count);
    getLastCudaError("reference image sum_square kernel error");

    // compute the sum area table of the search images
    sat2d_kernel<<<nblocks, nthreads, 0, stream>>>(secondarySat->devData, secondarySat2->devData, secondary->devData,
        secondary->height, secondary->width, secondary->count);
    getLastCudaError("search image sat kernel error");

    nthreads = NTHREADS2D;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(correlation->height,nthreads), IDIVUP(correlation->width,nthreads), correlation->count);
    cuCorrNormalizeSAT_kernel<<<gridSize, blockSize, 0, stream>>>(correlation->devData,
        referenceSum2->devData, secondarySat->devData, secondarySat2->devData,
        correlation->height, correlation->width,
        reference->height, reference->width,
        secondary->height, secondary->width);
    getLastCudaError("cuCorrNormalizeSAT_kernel kernel error");
}
