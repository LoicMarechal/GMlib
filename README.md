# GMlib version 2.0
Porting meshing tools and solvers that deal with unstructured meshes on GPUs

# Overview
The purpose of the GMlib is to provide programmers of solvers or automated meshers in the field of scientific computing with an easy, fast and transparent way to port their codes on GPUs (Graphic Processing Units).  
This library is based on the OpenCL language standard, thus taking advantage of almost every architectures supported by most platforms (Linux, Mac OS X, Windows).  
It is a simple loop parallelization scheme (known as kernels in the realm of GPU computing).  
Provides the programer with pre defined mesh data structures.  
Automatically vectorizes unstructured data like the ball of points or the edge shells.  
It requires some knowledge on OpenCL programing, which akin to C and C++.  
Handles transparently the transfer and vectorization of mesh data structures.


# Build
Simply follow these steps:
- unarchive the ZIP file
- `cd GMlib-master`
- `cmake .`
- `make`
- `make install`

Optionally, you may download some sample meshes to run the examples:
- [edges.meshb.lz] https://github.com/LoicMarechal/GMlib/raw/master/sample_meshes/edges.meshb.lz
- `lzip -d edges.meshb.lz`
- [hexahedra.meshb.lz] https://github.com/LoicMarechal/GMlib/raw/master/sample_meshes/ehexahedra.meshb.lz
- `lzip -d hexahedra.meshb.lz`

# Usage
The GMlib is written in ANSI C with some parts in OpenCL.  
It is made of a single C file and a header file to be compiled and linked alongside the calling program.  
It may be used in C and C++ programs (Fortran 77 and 90 APIs are under way).  
Tested on Linux, Mac OS X, Windows 7-10.
