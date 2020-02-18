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

// Create a vertex data type
VerIdx = GmlNewMeshData(GmlVertices, NmbVer);

// Fill the datatype with your mesh coordinates
for(i=0;i<NmbVer;i++)
   GmlSetDataLine(VerIdx, i, coords[i][0], coords[i][1], coords[i][2], VerRef[i]);

// Do the same with the elements
TriIdx = GmlNewMeshData(GmlTriangles, NmbTri);
for(i=0;i<NmbTri;i++)
   GmlSetDataLine(TriIdx, i, TriVer[i][0], TriVer[i][1], TriVer[i][2], TriRef[i]);

// Create a raw datatype to store the calculated elements' centers
MidIdx = GmlNewSolutionData(GmlTriangles, 1, GmlFlt4, "TriMid");

// Compile the OpenCL source code with the two needed datatypes:
// the vertex coordinates (read) and the triangles centers (write)
CalMid = GmlCompileKernel( TriangleCenter, "CalMid", NULL, GmlTriangles, 2,
                           VerIdx, GmlReadMode,  NULL,
                           MidIdx, GmlWriteMode, NULL );

// Launch the kernel on the GPU
GmlLaunchKernel(CalMid);

// Get the results back from the GPU and print it
for(i=0;i<NmbTri;i++)
{
   GmlGetDataLine(MidIdx, i, MidTab[i]);
   printf("triangle %d center = %g %g %g\n", i, MidTab[i][0], MidTab[i][1], MidTab[i][2]);
}
```

Then the "OpenCL" part executed by the GPU device:
```C++
TriMid = (TriCrd[0] + TriCrd[1] + TriCrd[2]) / 3.;
```
