/*
 * @file cuNormalizer.cu
 * @brief processors to normalize the correlation surface
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

cuNormalizeProcessor*
newCuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    return new cuNormalizeSAT(secondaryNX, secondaryNY, count);
}

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work array
    // reference sum square
    referenceSum2 = new cuArrays<float>(1, 1, count);
    referenceSum2->allocate();

    // secondary sum and sum square
    secondarySAT = new cuArrays<float2>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = new cuArrays<float>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
};

cuNormalizeSAT::~cuNormalizeSAT()
{
    delete referenceSum2;
    delete secondarySAT;
    delete secondarySAT2;
}

void cuNormalizeSAT::execute(cuArrays<float> *correlation,
    cuArrays<float2> *reference, cuArrays<float2> *secondary, cudaStream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        referenceSum2, secondarySAT, secondarySAT2, stream);
}


// end of file
