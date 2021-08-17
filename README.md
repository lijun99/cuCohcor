# PyCuCohcor - Coherent Cross-Correlation with GPU

## Contents

  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. User Guide](#3-user-guide)
  * [4. List of Parameters](#4-list-of-parameters)
  * [5. List of Procedures](#5-list-of-procedures)

## 1. Introduction

PyCuCohcor uses coherent cross-correlation instead of amplitude(incoherent) cross-correlation in Ampcor. The procedures follow those in Ampcor. 

## 2. Installation

We recommend installing it to a virtual Python environment together with ISCE2. See, e.g., [ISCE2 Installation with GPU](https://github.com/lijun99/isce2-install) for a guide. 

### 2.1 CMake Installation 

This will be similar with ISCE2 CMake installation. 

```bash 
    # clone the github repo
    git clone https://github.com/lijun99/cuCohcor
    # go to cuCohcor directory
    cd cuCohcor
    # create a build directory
    mkdir build && cd build
    # run cmake
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_FLAGS='-arch=sm_60' -DCMAKE_BUILD_TYPE=Release
    # compile and install
    make -j && make install
```
Remember to change CMake options to appropriate settings, e.g., *CMAKE_CUDA_FLAGS* to the target GPU architecture; *CMAKE_BUILD_TYPE* to *DEBUG* for detailed outputs (correlation surfaces, runs much slower).  
     

### 2.2 Standalone Installation

You may install PyCuCohpcor as a standalone package.


**Note** You need to modify *Makefile* and *setup.py* in *src* directory to provide 1) the correct path for GDAL include and lib; 2) the correct GPU architecture; 3) if CUDA is installed in another directory, change the path as well.

```bash
    # go to PyCucor source directory
    cd src
    # edit Makefile to provide the correct gdal include path and gpu architecture to NVCCFLAGS
    # call make to compile
    make -j 
    # install 
    python3 setup.py install  
 ```

## 3. User Guide

The main procedures of PyCuCohcor are implemented with CUDA/C++. A Python interface to configure and run PyCuCohcor is offered.

### 3.1 cuCoherentOffsets.py

*cuCoherentOffsets.py*, included in the *example* directory, serves as a general purpose script to run PyCuCohcor. It has the same structure as *cuDenseOffsets.py* in PyCuAmpcor. 


It uses *argparse* to pass parameters, either from a command line

```bash
cuCoherentOffsets.py -r 20151120.slc.full -s 20151214.slc.full --outprefix ./20151120_20151214/offset --ww 64 --wh 64 --oo 32 --kw 300 --kh 100 --nwac 32 --nwdc 1 --sw  20 --sh 20 --gpuid 2
 ```

 or by a shell script

 ```
#!/bin/bash
reference=./merged/SLC/20151120/20151120.slc.full # reference image name 
secondary=./merged/SLC/20151214/20151214.slc.full # secondary image name
ww=64  # template window width
wh=64  # template window height
sw=20   # (half) search range along width
sh=20   # (half) search range along height
kw=300   # skip between windows along width
kh=100   # skip between windows along height
mm=0   # margin to be neglected 
gross=0  # whether to use a varying gross offset
azshift=0 # constant gross offset along height/azimuth 
rgshift=0 # constant gross offset along width/range
deramp=0 # 0 for mag (TOPS), 1 for complex linear ramp, 2 for complex no deramping  
oo=32  # correlation surface oversampling factor 
outprefix=./merged/20151120_20151214/offset  # output prefix
outsuffix=_ww64_wh64   # output suffix
gpuid=0   # GPU device ID
nstreams=2 # number of CUDA streams
usemmap=1 # whether to use memory-map i/o
mmapsize=8 # buffer size in GB for memory map
nwac=32 # number of windows in a batch along width
nwdc=1  # number of windows in a batch along height

rm $outprefix$outsuffix*
cuCoherentOffsets.py --reference $reference --secondary $secondary --ww $ww --wh $wh --sw $sw --sh $sh --mm $mm --kw $kw --kh $kh --gross $gross --rr $rgshift --aa $azshift --oo $oo --deramp $deramp --outprefix $outprefix --outsuffix $outsuffix --gpuid $gpuid  --usemmap $usemmap --mmapsize $mmapsize --nwac $nwac --nwdc $nwdc 
 ```

Note that in PyCuCohcor, the following names for directions are equivalent:
* row, height, down, azimuth, along the track.
* column, width, across, range, along the sight.

In the above script, the computation starts from the (mm+sh, mm+sw) pixel in the reference image, take a series of template windows of size (wh, ww) with a skip (sh, sw), cross-correlate with the corresponding windows in the secondary image, and iterate till the end of the images. The output offset fields are stored in *outprefix+outputsuffix+'.bip'*, which is in BIP format, i.e., each pixel has two bands of float32 data, (offsetDown, offsetAcross). The total number of pixels is given by the total number of windows (numberWindowDown, numberWindowAcross), which is computed by the script and also saved to the xml file.

If you are interested in a particular region instead of the whole image, you may specify the location of the starting pixel (in reference image) and the number of windows desired by adding

```
--startpixelac $startPixelAcross --startpixeldw $startPixelDown --nwa $numberOfWindowsAcross --nwd $numberOfWindowsDown
```

PyCuCohcor supports two types of gross offset fields,
* static (--gross=0), i.e., a constant shift between reference and secondary images. The static gross offsets can be passed by *--rr $rgshift --aa $azshift*. Note that the margin as well as the starting pixel may be adjusted.
* dynamic (--gross=1), i.e., shifts between reference windows and secondary windows are varying in different locations. This is helpful to reduce the search range if you have a prior knowledge of the estimated offset fields, e.g., the velocity model of glaciers. You may prepare a BIP input file of the varying gross offsets (same format as the output offset fields), and use the option *--gross-file $grossOffsetFilename*. If you need the coordinates of reference windows, you may run *cuCoherentOffsets.py* at first to find out the location of the starting pixel and the total number of windows. The coordinate for the starting pixel of the (iDown, iAcross) window will be (startPixelDown+iDown\*skipDown, startPixelAcross+iAcross\*skipAcross).

### 3.2 Customized Python Scripts

If you need more control of the computation, you may follow the examples to create your own Python script. The general steps are
* create a PyCuCohcor instance
```python
# if standalone
from PyCuCohcor import PyCuCohcor
# create an instance
objOffset = PyCuCohcor()
```
The rest is the same as in *PyCuAmpcor*.

## 4. List of Parameters

**Image Parameters**

| PyCuCohcor           | Notes                     |
| :---                 | :----                     |
| referenceImageName   | The file name of the reference/template image |
| referenceImageHeight | The height of the reference image |
| referenceImageWidth  | The width of the reference image |
| secondaryImageName   | The file name of the secondary/search image   |
| secondaryImageHeight | The height of the secondary image |
| secondaryImageWidth  | The width of the secondary image |
| grossOffsetImageName | The output file name for gross offsets  |
| offsetImageName      | The output file name for dense offsets  |
| snrImageName         | The output file name for signal-noise-ratio of the correlation |
| covImageName         | The output file name for variance of the correlation surface |

PyCuCohcor now uses exclusively the GDAL driver to read images, only single-precision binary data are supported. (Image heights/widths are still required as inputs; they are mainly for dimension checking.  We will update later to read them with the GDAL driver). Multi-band is not currently supported, but can be added if desired.

The offset output is arranged in BIP format, with each pixel (azimuth offset, range offset). In addition to a static gross offset (i.e., a constant for all search windows), PyCuCohcor supports varying gross offsets as inputs (e.g., for glaciers, users can compute the gross offsets with the velocity model for different locations and use them as inputs for PyCuCohcor.

The offsetImage only outputs the (dense) offset values computed from the cross-correlations. Users need to add offsetImage and grossOffsetImage to obtain the total offsets.

The dimension/direction names used in PyCuCohcor are:
* the inner-most dimension x(i): row, height, down, azimuth, along the track.
* the outer-most dimension y(j): column, width, across, range, along the sight.

Note that ampcor.F and GDAL in general use y for rows and x for columns.

Note also PyCuCohcor parameters refer to the names used by the PyCuCohcor Python class. They may be different from those used in C/C++/CUDA, or the cuCoherentOffsets.py args.

**Process Parameters**

| PyCuCohcor           | Notes                     |
| :---                 | :----                     |
| devID                | The CUDA GPU to be used for computation, usually=0, or users can use the CUDA_VISIBLE_DEVICES=n enviromental variable to choose GPU |
| nStreams | The number of CUDA streams to be used, recommended=2, to overlap the CUDA kernels with data copying, more streams require more memory which isn't alway better |
| useMmap              | Whether to use memory map cached file I/O, recommended=1, supported by GDAL vrt driver (needs >=3.1.0) and GeoTIFF |
| mmapSize             | The cache size used for memory map, in units of GB. The larger the better, but not exceed 1/4 the total physical memory. |
| numberWindowDownInChunk |  The number of windows processed in a batch/chunk, along lines |
| numberWindowAcrossInChunk | The number of windows processed in a batch/chunk, along columns |

Many windows are processed together to maximize the usage of GPU cores; which is called as a Chunk. The total number of windows in a chunk is limited by the GPU memory. We recommend
numberWindowDownInChunk=1, numberWindowAcrossInChunk=10, for a window size=64.


**Search Parameters**

| PyCuCohcor           | Notes    |
| :---                 | :----                     |
| skipSampleDown       | The skip in pixels for neighboring windows along height |
| skipSampleAcross     | The skip in pixels for neighboring windows along width |
| numberWindowDown     | the number of windows along height |
| numberWindowAcross   | the number of windows along width  |
| referenceStartPixelDownStatic | the starting pixel location of the first reference window - along height component |
|referenceStartPixelAcrossStatic | the starting pixel location of the first reference window - along width component |

The C/C++/CUDA program accepts inputs with the total number of windows (numberWindowDown, numberWindowAcross) and the starting pixels of each reference window. The purpose is to establish multiple-threads/streams processing. Therefore, users are required to provide/compute these inputs, with tools available from PyCuCohcor python class. The cuCoherentOffsets.py script also does the job.

We provide some examples below, assuming a PyCuCohcor class object is created as

```python
    objOffset = PyCuCohcor()
```

**To compute the total number of windows**

We use the line direction as an example, assuming parameters as

```
   margin # the number of pixels to neglect at edges
   halfSearchRangeDown # the half of the search range
   windowSizeHeight # the size of the reference window for feature tracking
   skipSampleDown # the skip in pixels between two reference windows
   referenceImageHeight # the reference image height, usually the same as the secondary image height
```

and the number of windows may be computed along lines as

```python
   objOffset.numberWindowDown = (referenceImageHeight-2*margin-2*halfSearchRangeDown-windowSizeHeight) // skipSampleDown
```

If there is a gross offset, you may also need to subtract it when computing the number of windows.

The output offset fields will be of size (numberWindowDown, numberWindowAcross). The total number of windows numberWindows = numberWindowDown\*numberWindowAcross.

**To compute the starting pixels of reference/secondary windows**

The starting pixel for the first reference window is usually set as

```python
   objOffset.referenceStartPixelDownStatic = margin + halfSearchRangeDown
   objOffset.referenceStartPixelAcrossStatic = margin + halfSearchRangeAcross
```

you may also choose other values, e.g., for a particular region of the image, or a certain location for debug purposes.


With a constant gross offset, call

```python
   objOffset.setConstantGrossOffset(grossOffsetDown, grossOffsetAcross)
```

to set the starting pixels of all reference and secondary windows.

The starting pixel for the secondary window will be (referenceStartPixelDownStatic-halfSearchRangeDown+grossOffsetDown, referenceStartPixelAcrossStatic-halfSearchRangeAcross+grossOffsetAcross).

For cases you choose a varying grossOffset, you may use two numpy arrays to pass the information to PyCuCohcor, e.g.,

```python
    objOffset.referenceStartPixelDownStatic = objOffset.halfSearchRangeDown + margin
    objOffset.referenceStartPixelAcrossStatic = objOffset.halfSearchRangeAcross + margin
    vD = np.random.randint(0, 10, size =objOffset.numberWindows, dtype=np.int32)
    vA = np.random.randint(0, 1, size = objOffset.numberWindows, dtype=np.int32)
    objOffset.setVaryingGrossOffset(vD, vA)
```

to set all the starting pixels for reference/secondary windows.

Sometimes, adding a large gross offset may cause the windows near the edge to be out of range of the orignal image. To avoid memory access errors, call

```python
   objOffset.checkPixelInImageRange()
```

to verify. If an out-of-range error is reported, you may consider to increase the margin or reduce the number of windows.

## 5. List of Procedures

We follow the same procedure as in Ampcor; 

- take a template window from reference SLC
- take a large search window from secondary SLC 
- deramp both windows (assuming linear-ramp)
- perform the normalized coherent cross-correlation between template and search windows (with frequency-domain algorithm)
- find the max location in the correlation surface
- perform statistics on the correlation surface
- retake a smaller search window from secondary SLC, around the max location 
- oversample both template and search windows by a factor of 2 
- deramp both windows
- perform the normalized coherent cross-correlation again (with a smaller search range)
- find the max location of correlation 
- extract a smaller window from the correlation surface around the max location
- oversample the correlation surface (sub-pixel resolution)
- find the max again
- determine the final offset. 

For more details, see [PyCuAmpcor Documentation](https://github.com/isce-framework/isce2/tree/main/contrib/PyCuAmpcor#5-list-of-procedures).
