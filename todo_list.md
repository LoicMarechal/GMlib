
## PHASE 1

### Internal source code

- Split the compile kernel procedure into separate modules to better handle uplink kernels
- Write the neighbors link kernel compilation :heavy_check_mark:
- Handle user defined topological links :heavy_check_mark:
- Handle the shell uplinks :heavy_check_mark:
- Write the tet and triangles shell generation :heavy_check_mark:
- Write the ball of edges or triangles :heavy_check_mark:
- Write the triangles and tets neighbors generation :heavy_check_mark:
- Add the voyeurs information in all the uplink data and kernel sources :heavy_check_mark:
- Add a reduction internal kernel with min, max and L1 :heavy_check_mark:
- Add single float, double float and user selectable real float :heavy_check_mark:
- Make the GMlib reentrant :heavy_check_mark:
- Revise the parameters structure mechanism :heavy_check_mark:

### Exposed CPU API

- `GmlReduce(GMlibIndex, ReductionOpperation, &residual);` :heavy_check_mark:

### Examples

- Tetrahedral mesh nodes smoother :heavy_check_mark:
- Tetrahedral mesh quality calculation and statistics :heavy_check_mark:
- Basic iterative edge based solver with boundary conditions :heavy_check_mark:

### Documentation

- Quick reference card
- GitHub page :heavy_check_mark:

## PHASE 2

### Internal source code

- Optional include of libMeshb :heavy_check_mark:
- Write a .meshb to GMlib import module :heavy_check_mark:
- Write a .solb to GMlib import module
- Write a GMlib to .solb export module :heavy_check_mark:
- Optional include of LPlib
- Write a renumbering analyzer preprocessor :heavy_check_mark:
- Call the LPlib Hilbert renumbering and renumber the whole input data
- Provide a plugin functions mechanism available on the GPU side :heavy_check_mark:
- Develop basic geometric functions on tets, hexes, triangles, quads and edges :heavy_check_mark:
- length, surface, volume and quality :heavy_check_mark:
- Handle hybrid meshes with prisms and pyramids
- Update the ball generation to handle hybrid meshes
- Update the shell generation to handle hybrid meshes
- Update the neighbors generation to handle hybrid meshes
- Add a SetBlock() function for faster upload :heavy_check_mark:
- Add a GetBlock() function for faster download
- Add a GetLinkInfo() function to get sizes of vairable width topolinks :heavy_check_mark:
- Include an optional user's toolkit before a user kernel :heavy_check_mark:

### Exposed CPU API

- `GmlImportMesh("file.meshb", GmfTetrahedra, GmfTriangles, GmfVertices, 0);` :heavy_check_mark:
- `GmlImportSolution("file.solb", GmfSolAtTetrahedra, GmfSolAtVertices, 0);`
- `GmlExportMesh("file.meshb", NmbDat, IdxTab[]);`
- `GmlExportSolution("file.solb", NmbDat, IdxTab[]);` :heavy_check_mark:
- `GmlEvaluateNumbering();` :heavy_check_mark:
- `GmlHilbertRenumbering();`
- `GmlGetLinkInfo(*n, *w, *N, *W);` :heavy_check_mark:
- `GmlIncludeUserToolkit(char *PtrSrc);` :heavy_check_mark:

### Exposed GPU API

- `CalEdgLen();` :heavy_check_mark:
- `CalTriSrf();` :heavy_check_mark:
- `CalQadSrf();` :heavy_check_mark:
- `CalTetVol();` :heavy_check_mark:
- `CalPyrVol();` :heavy_check_mark:
- `CalPriVol();` :heavy_check_mark:
- `CalHexVol();` :heavy_check_mark:
- `CalTriQal();` :heavy_check_mark:
- `CalQadQal();` :heavy_check_mark:
- `CalTetQal();` :heavy_check_mark:
- `CalPyrQal();` :heavy_check_mark:
- `CalPriQal();` :heavy_check_mark:
- `CalHexQal();` :heavy_check_mark:

### Examples

- Update quality and node smoother with the internal geometric functions
- Develop the same two codes working with hexes
- Add a Hilbert check and renumber step to all examples

### Documentation

- Full PDF documentation with quick setup, install, compile, write OpenCL code and procedure lists :heavy_check_mark:
- Full GitHub page with examples, sample files, source code and links to OpenCL and GPU programing :heavy_check_mark:

## PHASE 3

### Internal source code

- Add P2 and P3 simplicial elements handling except for some geometrical operations
- Develop basic geometric functions on prisms and pyramids
- Distance and intersection
- Develop a CUDA version
- Develop a Bezier to Lagrange converter on the GPU
- Develop a Lagrange to Bezier converter on the GPU

### Exposed CPU API

- `GmlInit(idx, CUDA | OpenCL);`
- New keywords: GmlEdgesP2, GmlEdgesP3, GmlTrianglesP2, GmlTrianglesP3, GmlTetrahedraP2, GmlTetrahedraP3

### Exposed GPU API

- `distance = DisVerVer(float4 a, float4 ver);`
- `distance = DisVerEdg(float4 a, float4 edg[2]);`
- `distance = DisVerTri(float4 a, float4 tri[3]);`
- `coordinates[2] = BarVerEdg(float4 a, float4 edg[2]);`
- `coordinates[3] = BarVerTri(float4 a, float4 tri[3]);`
- `coordinates[4] = BarVerTet(float4 a, float4 tet[4]);`
- `flag = IntEdgTri(float4 edg[2], float4 tri[3], float4 ver);`
- `edg[ 2] = BezLagEdgP2(float4 edg[ 2]);`
- `edg[ 3] = BezLagEdgP3(float4 edg[ 3]);`
- `tri[ 6] = BezLagTriP2(float4 tri[ 6]);`
- `tri[10] = BezLagTriP3(float4 tri[ 6]);`
- `tet[10] = BezLagTetP2(float4 tet[10]);`
- `tet[20] = BezLagTetP3(float4 tet[20]);`
- `edg[ 2] = LagBezEdgP2(float4 edg[ 2]);`
- `edg[ 3] = LagBezEdgP3(float4 edg[ 3]);`
- `tri[ 6] = LagBezTriP2(float4 tri[ 6]);`
- `tri[10] = LagBezTriP3(float4 tri[ 6]);`
- `tet[10] = LagBezTetP2(float4 tet[10]);`
- `tet[20] = LagBezTetP3(float4 tet[20]);`
- `length  = CalEdgP2Len(float4 edg[ 2]);`
- `length  = CalEdgP3Len(float4 edg[ 3]);`
- `surface = CalTriP2Srf(float4 tri[ 6]);`
- `surface = CalTriP3Srf(float4 tri[10]);`
- `volume  = CalTetP2Vol(float4 tet[10]);`
- `volume  = CalTetP3Vol(float4 tet[20]);`
- `quality = CalTriP2Qal(float4 tri[ 6]);`
- `quality = CalTriP3Qal(float4 tri[10]);`
- `quality = CalTetP2Qal(float4 tet[10]);`
- `quality = CalTetP3Qal(float4 tet[20]);`

### Examples

- P2 quality mesh calculator
- P3 quality mesh calculator
- Compute the Hausdorff distance between two meshes

### Documentation

- Update documentation and GitHub with High order elements
- Add a WiKi on how to develop additional plugins
