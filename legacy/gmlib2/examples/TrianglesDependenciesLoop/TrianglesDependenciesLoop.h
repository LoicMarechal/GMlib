char *TrianglesDependenciesLoop = "\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*						GMLIB 2.0							*/\n" \
"/*															*/\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*	Description:		Loop over the element, compute some	*/\n" \
"/*						values for each vertices and store	*/\n" \
"/*						them in an element scatter-buffer.	*/\n" \
"/*						Then loop over the vertices and		*/\n" \
"/*						gather the values stored in each	*/\n" \
"/*						buffers of the ball.				*/\n" \
"/*	Author:				Loic MARECHAL						*/\n" \
"/*	Creation date:		nov 26 2012							*/\n" \
"/*	Last modification:	nov 29 2012							*/\n" \
"/*															*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* GMLIB parameters structure								*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"typedef struct\n" \
"{\n" \
"	int empty;\n" \
"}GmlParSct;\n" \
"\n" \
"#ifndef mix\n" \
"#define mix(x,y,a) (x+(y-x)*a)\n" \
"#endif\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Scatter the triangles data in local buffers				*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TrianglesScatter(	__global int4 *TriVer, __global float4 (*TriPos)[3], __global float4 *VerCrd, 								__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int4 idx;\n" \
"	float4 crd[3];\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the three triangle vertex indices */\n" \
"	idx = TriVer[i];\n" \
"\n" \
"	/* Copy the vertices coordinates into a temporary buffer */\n" \
"	crd[0] = VerCrd[ idx.s0 ];\n" \
"	crd[1] = VerCrd[ idx.s1 ];\n" \
"	crd[2] = VerCrd[ idx.s2 ];\n" \
"\n" \
"	/* Compute all three values and store the results in the triangle scatter-buffer */\n" \
"	TriPos[i][0] = ((float4){2,2,2,0}*crd[0] + crd[1] + crd[2]) * (float4){.25, .25, .25, 0};\n" \
"	TriPos[i][1] = ((float4){2,2,2,0}*crd[1] + crd[2] + crd[0]) * (float4){.25, .25, .25, 0};\n" \
"	TriPos[i][2] = ((float4){2,2,2,0}*crd[2] + crd[0] + crd[1]) * (float4){.25, .25, .25, 0};\n" \
"}\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Gather the data stored in the 8 first incident triangles	*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TrianglesGather1(	__global char *VerDeg, __global int8 *VerBal, __global float4 (*TriPos)[3], 								__global float4 *VerCrd, __global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i, deg;\n" \
"	int8 BalCod, TriIdx, VerIdx;\n" \
"	float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	deg = VerDeg[i];		// get the vertex partial degree: maximum 8\n" \
"	BalCod = VerBal[i]; 	// read a vector containing 8 encoded ball data\n" \
"	TriIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices\n" \
"	VerIdx = BalCod & (int8){7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices\n" \
"\n" \
"	// Sum all coordinates\n" \
"	NewCrd += (deg >  0) ? TriPos[ TriIdx.s0 ][ VerIdx.s0 ] : NulCrd;\n" \
"	NewCrd += (deg >  1) ? TriPos[ TriIdx.s1 ][ VerIdx.s1 ] : NulCrd;\n" \
"	NewCrd += (deg >  2) ? TriPos[ TriIdx.s2 ][ VerIdx.s2 ] : NulCrd;\n" \
"	NewCrd += (deg >  3) ? TriPos[ TriIdx.s3 ][ VerIdx.s3 ] : NulCrd;\n" \
"	NewCrd += (deg >  4) ? TriPos[ TriIdx.s4 ][ VerIdx.s4 ] : NulCrd;\n" \
"	NewCrd += (deg >  5) ? TriPos[ TriIdx.s5 ][ VerIdx.s5 ] : NulCrd;\n" \
"	NewCrd += (deg >  6) ? TriPos[ TriIdx.s6 ][ VerIdx.s6 ] : NulCrd;\n" \
"	NewCrd += (deg >  7) ? TriPos[ TriIdx.s7 ][ VerIdx.s7 ] : NulCrd;\n" \
"\n" \
"	// Compute the average value and store it\n" \
"	VerCrd[i] = NewCrd / (float)deg;\n" \
"}\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Gather the data stored in the remaining triangles		*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TrianglesGather2(	__global int (*ExtDeg)[3], __global int *ExtBal, __global float4 (*TriPos)[3], 								__global float4 *VerCrd, __global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i, j, deg, VerIdx, BalCod, BalAdr;\n" \
"	float4 NewCrd;\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	VerIdx = ExtDeg[i][0];	// get the vertex global index\n" \
"	BalAdr = ExtDeg[i][1];	// adress of the first encoded ball data\n" \
"	deg = ExtDeg[i][2];		// extra vertex degree above 8\n" \
"	NewCrd = VerCrd[ VerIdx ] * (float4){8,8,8,0};	// restart from the partial calculation done above\n" \
"\n" \
"	for(j=BalAdr; j<BalAdr + deg; j++)\n" \
"	{\n" \
"		BalCod = ExtBal[j];	// read the the encoded ball data\n" \
"		NewCrd += TriPos[ BalCod >> 3 ][ BalCod & 7 ];	// decode and add the coordinates\n" \
"	}\n" \
"\n" \
"	VerCrd[ VerIdx ] = NewCrd / (float)(8 + deg);	// compute the average value and store it\n" \
"}\n" \
"\n" \

;