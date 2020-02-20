
## PHASE 1

### Internal source code

- Split the compile kernel procedure into separate modules to better handle uplink kernels
- Write the neighbors link kernel compilation
- Handle user defined topological links
- Handle the shell uplinks
- Write the tet and triangles shell generation
- Write the ball of edges or triangles
- Write the triangles and tets neighbors generation
- Add the voyeurs information in all the uplink data and kernel sources
- Add a reduction internal kernel with min, max and L1 :heavy_check_mark:
- Add single float, double float and user selectable real float
- Make the GMlib reentrant :heavy_check_mark:
- Revise the parameters structure mechanism :heavy_check_mark:

### Exposed CPU API

- GmlReduce(GMlibIndex, ReductionOpperation, &residual); :heavy_check_mark:

### Examples

- Tetrahedral mesh nodes smoother
- Tetrahedral mesh quality calculation and statistics
- Basic iterative edge based solver with boundary conditions

### Documentation

- Quick reference card
- GitHub page

## PHASE 2

### Internal source code

- Optional include of libMeshb
- Write a .meshb to GMlib import module
- Write a .solb to GMlib import module
- Write a GMlib to .solb export module
- Optional include of LPlib
- Write a renumbering analyzer preprocessor
- Call the LPlib Hilbert renumbering and renumber the whole input data
- Provide a plugin functions mechanism available on the GPU side
- Develop basic geometric functions on tets, hexes, triangles , quads and edges
- length, surface, volume and quality
- Handle hybrid meshes with prisms and pyramids
- Update the ball generation to handle hybrid meshes
- Update the shell generation to handle hybrid meshes
- Update the neighbors generation to handle hybrid meshes

### Exposed CPU API

- `C++ IdxTab[] = GmlImportFile("file.meshb", GmfTetrahedra, GmfTriangles, GmfVertices, 0);`
- IdxTab[] = GmlImportFile("file.solb", GmfSolAtTetrahedra, GmfSolAtVertices, 0);
- GmlExportFile("file.meshb", NmbDat, IdxTab[]);
- GmlExportFile("file.solb", NmbDat, IdxTab[]);
- GmlEvaluateNumbering();
- GmlHilbertRenumbering();

### Exposed GPU API

- CalEdgLen(float4 ver[2]);
- CalTriSrf(float4 ver[3]);
- CalQadSrf(float4 ver[4]);
- CalTetVol(float4 ver[4]);
- CalHexVol(float4 ver[8]);
- CalTriQal(float4 ver[3]);
- CalQadQal(float4 ver[4]);
- CalTetQal(float4 ver[4]);
- CalHexQal(float4 ver[8]);

### Examples

- Update quality and node smoother with the internal geometric functions
- Develop the same two codes working with hexes
- Add a Hilbert check and renumber step to all examples

### Documentation

- Full PDF documentation with quick setup, install, compile, write OpenCL code and procedure lists
- Full GitHub page with examples, sample files, source code and links to OpenCL and GPU programing

## PHASE 3

### Internal source code

- Add P2 and P3 simplicial elements handling except for some geometrical operations
- Develop basic geometric functions on prisms and pyramids
- Distance and intersection
- Develop a CUDA version
- Develop a Bezier to Lagrange converter on the GPU
- Develop a Lagrange to Bezier converter on the GPU

### Exposed CPU API

- GmlInit(idx, CUDA | OpenCL);
- New keywords: GmlEdgesP2, GmlEdgesP3, GmlTrianglesP2, GmlTrianglesP3, GmlTetrahedraP2, GmlTetrahedraP3

### Exposed GPU API

- distance = DisVerVer(float4 a, float4 ver);
- distance = DisVerEdg(float4 a, float4 edg[2]);
- distance = DisVerTri(float4 a, float4 tri[3]);
- coordinates[2] = BarVerEdg(float4 a, float4 edg[2]);
- coordinates[3] = BarVerTri(float4 a, float4 tri[3]);
- coordinates[4] = BarVerTet(float4 a, float4 tet[4]);
- flag = IntEdgTri(float4 edg[2], float4 tri[3], float4 ver);
- edg[ 2] = BezLagEdgP2(float4 edg[ 2]);
- edg[ 3] = BezLagEdgP3(float4 edg[ 3]);
- tri[ 6] = BezLagTriP2(float4 tri[ 6]);
- tri[10] = BezLagTriP3(float4 tri[ 6]);
- tet[10] = BezLagTetP2(float4 tet[10]);
- tet[20] = BezLagTetP3(float4 tet[20]);
- edg[ 2] = LagBezEdgP2(float4 edg[ 2]);
- edg[ 3] = LagBezEdgP3(float4 edg[ 3]);
- tri[ 6] = LagBezTriP2(float4 tri[ 6]);
- tri[10] = LagBezTriP3(float4 tri[ 6]);
- tet[10] = LagBezTetP2(float4 tet[10]);
- tet[20] = LagBezTetP3(float4 tet[20]);
- length  = CalEdgP2Len(float4 edg[ 2]);
- length  = CalEdgP3Len(float4 edg[ 3]);
- surface = CalTriP2Srf(float4 tri[ 6]);
- surface = CalTriP3Srf(float4 tri[10]);
- volume  = CalTetP2Vol(float4 tet[10]);
- volume  = CalTetP3Vol(float4 tet[20]);
- quality = CalTriP2Qal(float4 tri[ 6]);
- quality = CalTriP3Qal(float4 tri[10]);
- quality = CalTetP2Qal(float4 tet[10]);
- quality = CalTetP3Qal(float4 tet[20]);

### Examples

- P2 quality mesh calculator
- P3 quality mesh calculator
- Compute the Hausdorff distance between two meshes

### Documentation

- Update documentation and GitHub with High order elements
- Add a WiKi on how to develop additional plugins
