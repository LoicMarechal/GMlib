# GMlib version 3.0
Porting meshing tools and solvers that deal with unstructured meshes on GPUs

# Overview
The purpose of the **GMlib** is to provide programmers of solvers or automated meshers in the field of scientific computing with an easy, fast and transparent way to port their codes on *GPUs* (Graphic Processing Units).  
This library is based on the *OpenCL* language standard, thus taking advantage of almost every architectures supported by most platforms (*Linux*, *macOS*, *Windows*).  
It is a simple loop parallelization scheme (known as kernels in the realm of GPU computing).  
Provides the programer with pre defined mesh data structures.  
Automatically vectorizes unstructured data like the ball of points or the edge shells.  
It requires some knowledge on OpenCL programing, which akin to C and C++.  
Handles transparently the transfer and vectorization of mesh data structures.


# Build for *Linux* or *macOS*
Simply follow these steps:
- unarchive the ZIP file
- `cd GMlib-master`
- `mkdir build`
- `cd build`
- `cmake -DCMAKE_INSTALL_PREFIX=$HOME/local ../`
- `make`
- `make install`

Optionally, you may download some sample meshes to run the examples:
- you need to install the [libMeshb](https://github.com/LoicMarechal/libMeshb) from GitHub
- manually download files from the *Git LFS* repository: [sample files](sample_meshes/)
- move them into /opt/GMlib/sample_meshes/
- uncompress them with `lzip -d *.meshb.lz`
- you may now enter /opt/GMlib/examples directory and run the various examples

# Usage
The **GMlib** is written in *ANSI C* with some parts in *OpenCL*.  
It is made of a single C file and a header file to be compiled and linked alongside the calling program.  
It may be used in C and C++ programs (Fortran 77 and 90 APIs are under way).  
Tested on *Linux*, *macOS*, *Windows 7-10*.

Here is a basic example that computes some triangles' barycenters on a GPU:

First the "C" part executed by the host CPU:
```C++
// Init the GMLIB with the first available GPU on the system
GmlInit(1);

// Compile the OpenCL source code
CalMid = GmlNewKernel(TrianglesBasicLoop, "ComputeCenters");

// Create a vertices data type
VerIdx = GmlNewData(GmlVertices, "Crd", NmbVer);

// Fill the datatype with your mesh coordinates
GmlSetDataBlock(VerIdx, VerTab[1], VerTab[ NmbVer ]);

// Do the same with the elements
TriIdx = GmlNewData(GmlTriangles, "Tri", NmbTri);
GmlSetDataBlock(TriIdx, TriTab[1], TriTab[ NmbTri ]);

// Create a raw datatype that will receive the elements' centers
MidIdx = GmlNewData(GmlRawData, "Mid", NmbTri, GmlTriangles, "float4", sizeof(cl_float4));

// Launch the kernel on the GPU passing three arguments to the OpenCL procedure:
// the elements connectivity, the barycenter table and the vertices coordinates
GmlLaunchKernel(CalMid, TriIdx, GmlRead, TriIdx, GmlWrite, MidIdx, GmlRead, VerIdx, GmlEnd);

// Get the results back from the GPU and print it
GmlGetDataBlock(MidIdx, MidTab[1], MidTab[ NmbTri ]);

for(i=0;i<NmbTri;i++)
   printf("triangle %d center = %g %g %g\n", i, MidTab[0], MidTab[1], MidTab[2]);
```

Then the "OpenCL" part executed by the GPU device:
```C++
Mid = (TriCrd[0] + TriCrd[1] + TriCrd[2]) * (float4){1/3,1/3,1/3,0};
```
