/*
 * @file cuWindowFilter
 * @brief Apply window function to filter images
 *

 */

#ifndef __CUWINDOWFILTER_H
#define __CUWINDOWFILTER_H

#include "cuArrays.h"
#include "cudaUtil.h"

/**
 * Abstract class interface for window filter
 * with different implementations
 */
class cuWindowFilter {
public:
    // default constructor and destructor
    cuWindowFilter() = default;
    virtual ~cuWindowFilter() = default;
    // execute interface
    virtual void filter(cuArrays<float2> * images, cudaStream_t stream) = 0;
};


/**
 * null window filter - do nothing
 */
class cuNullWindowFilter : public cuWindowFilter
{

public:
    cuNullWindowFilter(int NX, int NY) {}
    ~cuNullWindowFilter() {}
    void filter(cuArrays<float2> * images, cudaStream_t stream) override {}
};

/**
 * Hann Window function
 */

 class cuHannWindowFilter : public cuWindowFilter
 {
 public:
    cuHannWindowFilter(int NX, int NY);
    ~cuHannWindowFilter();
    void filter(cuArrays<float2> * images, cudaStream_t stream) override;
 private:
    cuArrays<float> * hann_coef;
    void init_coef();
 };

#endif
// end of file
