[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCL](https://img.shields.io/static/v1?label=OpenCL&message=1.1&color=red&style=flat&logo=data:image/png;base64,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)](https://www.khronos.org/opencl/)

![alt text](https://github.com/LoicMarechal/GMlib/blob/develop/Documentation/GMlib_logo.png "Gmlib logo made with Logo Maker ")

## Overview
The purpose of the **GMlib** is to provide programmers of solvers or automated meshers in the field of scientific computing with an easy, fast and transparent way to port their codes on *GPUs* (Graphic Processing Units).  
This library is based on the *OpenCL* language standard, thus taking advantage of almost every architectures supported by most platforms (*Linux*, *macOS*, *Windows*).  
It is a simple loop parallelization scheme (known as kernels in the realm of GPU computing).  
Provides the programer with pre defined mesh data structures.  
Automatically vectorizes unstructured data like the ball of points or the edge shells.  
It requires some knowledge on OpenCL programing, which akin to C and C++.  
Handles transparently the transfer and vectorization of mesh data structures.

## Build

### Prerequisites for *Linux* or *macOS*
- Install [CMake](https://cmake.org/files/v3.7/cmake-3.7.2-win64-x64.msi)
- A valid C99 compiler
- Open a shell window

### Prerequisites for *Windows*
- You first need to install [CMake](https://cmake.org/files/v3.7/cmake-3.7.2-win64-x64.msi). Do not forget to choose "add cmake to the path for all users", from the install panel.
- Then you need a valid C compiler like the free [Visual Studio Community 2019](https://www.visualstudio.com/vs/visual-studio-express/)
- Open the x64 Native Tools Command Prompt for VS (or x86 if you need to build a 32-bit version)

### Prerequisites for all platforms
You need the right OpenCL development environment specific to your GPU:
- [AMD](https://www.amd.com/en/support)
- [Intel](https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html)
- [Nvidia](https://www.nvidia.com/download/index.aspx)

### Build commands for all platforms
- unarchive the ZIP file
- `cd GMlib-master`
- `mkdir build`
- `cd build`
- `cmake ..`
- `cmake --build . --target install`

### Optional build
Optionally, you may download libMeshb to run the examples:
- you need to install the [libMeshb](https://github.com/LoicMarechal/libMeshb) from GitHub
- cd to /usr/local/GMlib/sample_meshes/
- uncompress them with `lzip -d *.meshb.lz` ([lzip](https://www.nongnu.org/lzip/lzip.html))
- you may now enter /usr/local/GMlib/examples directory and run the various examples

And the Hilbert renumbering command that is necessary to preprocess meshes before processing them with the GMlib
- download it from [LPlib](https://github.com/LoicMarechal/LPlib)
- use the command: "hilbert -in raw.meshb -out renum.meshb -gmlib"

## Usage
The **GMlib** is written in *ANSI C* with some parts in *OpenCL*.  
It is made of a single C file and a header file to be compiled and linked alongside the calling program.  
It may be used in C and C++ programs (Fortran 77 and 90 APIs are under way).  
Tested on *Linux*, *macOS*, *Windows 7-10*.

Here is a basic example that computes some triangles' barycenters on a GPU:

First the "C" part executed by the host CPU:
```C++
// Init the GMLIB with the first available GPU on the system
LibIdx = GmlInit(1);

// Create a vertex and a triangle datatype
VerIdx = GmlNewMeshData(LibIdx, GmlVertices,  NmbVer);
TriIdx = GmlNewMeshData(LibIdx, GmlTriangles, NmbTri);

// Fill the vertices with your mesh coordinates
for(i=0;i<NmbVer;i++)
   GmlSetDataLine(LibIdx, VerIdx, i, coords[i][0], coords[i][1], coords[i][2], VerRef[i]);

// Do the same with the elements
for(i=0;i<NmbTri;i++)
   GmlSetDataLine(LibIdx, TriIdx, i, TriVer[i][0], TriVer[i][1], TriVer[i][2], TriRef[i]);

// Create a raw datatype to store the calculated elements' centers
MidIdx = GmlNewSolutionData(LibIdx, GmlTriangles, 1, GmlFlt4, "TriMid");

// Compile the OpenCL source code with the two needed datatypes:
// the vertex coordinates (read) and the triangles centers (write)
CalMid = GmlCompileKernel( LibIdx,  TriangleCenter, "CalMid", GmlTriangles, 2,
                           VerIdx, GmlReadMode,  NULL,
                           MidIdx, GmlWriteMode, NULL );

// Launch the kernel on the GPU
GmlLaunchKernel(LibIdx, CalMid);

// Get the results back from the GPU and print it
for(i=0;i<NmbTri;i++)
{
   GmlGetDataLine(LibIdx, MidIdx, i, MidTab[i]);
   printf("triangle %d center = %g %g %g\n", i, MidTab[i][0], MidTab[i][1], MidTab[i][2]);
}
```

Then the "OpenCL" part executed by the GPU device:
```C++
TriMid = (TriCrd[0] + TriCrd[1] + TriCrd[2]) / 3.;
```
