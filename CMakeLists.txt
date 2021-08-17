cmake_minimum_required(VERSION 3.13...3.18)

project(PyCuCohcor LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/.cmake)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    enable_language(CUDA)
    find_package(CUDAToolkit) # TODO added in cmake 3.17 - copy this module
endif()

find_package(Python 3.5 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(OpenMP REQUIRED COMPONENTS C CXX)

# Find these, and create IMPORTED INTERFACE libraries for them if they exist
include(TargetGDAL)
include(UseCython)
include(utils)

# If we're the root cmake project (e.g. not add_subdirectory):
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
    # override this to also test the resulting extension
    function(Python_add_library target)
        _Python_add_library(${target} ${ARGN})
        add_test(NAME import_${target}
                 COMMAND ${Python_EXECUTABLE} -c
                     "import ${target}"
                 )
    endfunction()
endif()

if(NOT DEFINED PYTHON_MODULE_DIR)
    set(PYTHON_MODULE_DIR packages CACHE PATH
        "Python module directory (relative to install prefix)")
endif()



set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

cython_add_module(PyCuCohcor
    src/PyCuCohcor.pyx
    src/GDALImage.cu
    src/cuAmpcorChunk.cu
    src/cuAmpcorController.cu
    src/cuAmpcorParameter.cu
    src/cuArrays.cu
    src/cuArraysCopy.cu
    src/cuArraysPadding.cu
    src/cuCorrFrequency.cu
    src/cuCorrNormalization.cu
    src/cuCorrNormalizationSAT.cu
    src/cuCorrNormalizer.cu
    src/cuCorrTimeDomain.cu
    src/cuDeramp.cu
    src/cuEstimateStats.cu
    src/cuOffset.cu
    src/cuOverSampler.cu
    src/cuSincOverSampler.cu
    )

target_include_directories(PyCuCohcor PRIVATE
    src
    )

target_link_libraries(PyCuCohcor PRIVATE
    CUDA::cufft
    GDAL::GDAL
    Python::NumPy
    )


# Python site-packages
# Get canonical directory for python site packages (relative to install
# location).  It varys from system to system.
pycmd(PYTHON_SITE_PACKAGES "
      import os
      from distutils import sysconfig
      print(sysconfig.get_python_lib(prefix=''))
  ")

install(TARGETS PyCuCohcor
  LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
  )


