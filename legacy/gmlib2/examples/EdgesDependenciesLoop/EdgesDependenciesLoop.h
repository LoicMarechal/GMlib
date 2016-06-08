char *EdgesDependenciesLoop = "\n" \
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
"/* Scatter the elements data in local buffers				*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void EdgesScatter(	__global int2 *EdgVer, __global float4 (*EdgPos)[2], __global float4 *VerCrd, 							__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int2 idx;\n" \
"	float a=1./3., b=2./3.;\n" \
"	float4 crd[2];\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the two element vertex indices */\n" \
"	idx = EdgVer[i];\n" \
"\n" \
"	/* Copy the vertices coordinates into a temporary buffer */\n" \
"	crd[0] = VerCrd[ idx.s0 ];\n" \
"	crd[1] = VerCrd[ idx.s1 ];\n" \
"\n" \
"	/* Compute both values and store the results in the element scatter-buffer */\n" \
"	EdgPos[i][0] = mix(crd[0], crd[1], a);\n" \
"	EdgPos[i][1] = mix(crd[0], crd[1], b);\n" \
"}\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Gather the data stored in the 16 first incident elements	*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void EdgesGather1(	__global char *VerDeg, __global int16 *VerBal, __global float4 (*EdgPos)[2], 							__global float4 *VerCrd, __global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i, deg;\n" \
"	int16 BalCod, EdgIdx, VerIdx;\n" \
"	float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	deg = VerDeg[i];		// get the vertex partial degree: maximum 16\n" \
"	BalCod = VerBal[i]; 	// read a vector containing 16 encoded ball data\n" \
"	EdgIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices\n" \
"	VerIdx = BalCod & (int16){7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices\n" \
"\n" \
"	// Sum all coordinates\n" \
"	NewCrd += (deg >  0) ? EdgPos[ EdgIdx.s0 ][ VerIdx.s0 ] : NulCrd;\n" \
"	NewCrd += (deg >  1) ? EdgPos[ EdgIdx.s1 ][ VerIdx.s1 ] : NulCrd;\n" \
"	NewCrd += (deg >  2) ? EdgPos[ EdgIdx.s2 ][ VerIdx.s2 ] : NulCrd;\n" \
"	NewCrd += (deg >  3) ? EdgPos[ EdgIdx.s3 ][ VerIdx.s3 ] : NulCrd;\n" \
"	NewCrd += (deg >  4) ? EdgPos[ EdgIdx.s4 ][ VerIdx.s4 ] : NulCrd;\n" \
"	NewCrd += (deg >  5) ? EdgPos[ EdgIdx.s5 ][ VerIdx.s5 ] : NulCrd;\n" \
"	NewCrd += (deg >  6) ? EdgPos[ EdgIdx.s6 ][ VerIdx.s6 ] : NulCrd;\n" \
"	NewCrd += (deg >  7) ? EdgPos[ EdgIdx.s7 ][ VerIdx.s7 ] : NulCrd;\n" \
"	NewCrd += (deg >  8) ? EdgPos[ EdgIdx.s8 ][ VerIdx.s8 ] : NulCrd;\n" \
"	NewCrd += (deg >  9) ? EdgPos[ EdgIdx.s9 ][ VerIdx.s9 ] : NulCrd;\n" \
"	NewCrd += (deg > 10) ? EdgPos[ EdgIdx.sa ][ VerIdx.sa ] : NulCrd;\n" \
"	NewCrd += (deg > 11) ? EdgPos[ EdgIdx.sb ][ VerIdx.sb ] : NulCrd;\n" \
"	NewCrd += (deg > 12) ? EdgPos[ EdgIdx.sc ][ VerIdx.sc ] : NulCrd;\n" \
"	NewCrd += (deg > 13) ? EdgPos[ EdgIdx.sd ][ VerIdx.sd ] : NulCrd;\n" \
"	NewCrd += (deg > 14) ? EdgPos[ EdgIdx.se ][ VerIdx.se ] : NulCrd;\n" \
"	NewCrd += (deg > 15) ? EdgPos[ EdgIdx.sf ][ VerIdx.sf ] : NulCrd;\n" \
"\n" \
"	// Compute the average value and store it\n" \
"	VerCrd[i] = NewCrd / (float)deg;\n" \
"}\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Gather the data stored in the remaining elements			*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void EdgesGather2(	__global int (*ExtDeg)[3], __global int *ExtBal, __global float4 (*EdgPos)[2], 							__global float4 *VerCrd, __global GmlParSct *par, const int count )\n" \
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
"	deg = ExtDeg[i][2];		// extra vertex degree above 16\n" \
"	NewCrd = VerCrd[ VerIdx ] * (float4){16,16,16,0};	// restart from the partial calculation done above\n" \
"\n" \
"	for(j=BalAdr; j<BalAdr + deg; j++)\n" \
"	{\n" \
"		BalCod = ExtBal[j];	// read the the encoded ball data\n" \
"		NewCrd += EdgPos[ BalCod >> 3 ][ BalCod & 7 ];	// decode and add the coordinates\n" \
"	}\n" \
"\n" \
"	VerCrd[ VerIdx ] = NewCrd / (float)(16 + deg);	// compute the average value and store it\n" \
"}\n" \
"\n" \

;