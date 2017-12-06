# GMlib version 2.0
Porting meshing tools and solvers that deal with unstructured meshes on GPUs

# Overview
The purpose of the **GMlib** is to provide programmers of solvers or automated meshers in the field of scientific computing with an easy, fast and transparent way to port their codes on *GPUs* (Graphic Processing Units).  
This library is based on the *OpenCL* language standard, thus taking advantage of almost every architectures supported by most platforms (*Linux*, *macOS*, *Windows*).  
It is a simple loop parallelization scheme (known as kernels in the realm of GPU computing).  
Provides the programer with pre defined mesh data structures.  
Automatically vectorizes unstructured data like the ball of points or the edge shells.  
It requires some knowledge on OpenCL programing, which akin to C and C++.  
Handles transparently the transfer and vectorization of mesh data structures.


# Build
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
VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInput);

// Fill the datatype with your mesh coordinates
for(i=0;i<NmbVer;i++)
   GmlSetVertex(VerIdx, i, VerTab[i][0], VerTab[i][1], VerTab[i][2]);

// Transfer the data to the GPU
GmlUploadData(VerIdx);

// Do the same with the elements
TriIdx = GmlNewData(GmlTriangles, NmbTri, 0, GmlInput);
for(i=0;i<NmbTri;i++)
   GmlSetTriangle(TriIdx, i, TriTab[i][0], TriTab[i][1], TriTab[i][2]);
GmlUploadData(TriIdx);

// Create a raw datatype that will receive the elements' centers
MidIdx = GmlNewData(GmlRawData, NmbTri, sizeof(cl_float4), GmlOutput);

// Launch the kernel on the GPU passing three arguments to the OpenCL procedure:
// the elements connectivity, the barycenter table and the vertices coordinates
GmlLaunchKernel(CalMid, NmbTri, 3, TriIdx, MidIdx, VerIdx);

// Get the results back from the GPU and print it
GmlDownloadData(MidIdx);

for(i=0;i<NmbTri;i++)
{
   GmlGetRawData(MidIdx, i, MidTab);
   printf("triangle %d center = %g %g %g\n", i, MidTab[0], MidTab[1], MidTab[2]);
}
```

Then the "OpenCL" part executed by the GPU device:
```C++
__kernel void ComputeCenters(__global int4 *tri, __global float4 *mid, __global float4 *crd)
{
   int i = get_global_id(0);
   int4 idx;

   // Get the three triangle vertex indices stored in one integer vector
   idx = tri[i];

   // Get three vertices coordinates, compute and store the triangle's middle
   mid[i] = (crd[ idx.s0 ] + crd[ idx.s1 ] + crd[ idx.s2 ]) * (float4){1/3,1/3,1/3,0};
}
```
