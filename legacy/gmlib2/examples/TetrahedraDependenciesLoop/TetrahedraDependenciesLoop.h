char *TetrahedraDependenciesLoop = "\n" \
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
"/* Scatter the tetrahedra data in local buffers				*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TetrahedraScatter(__global int4 *TetVer, __global float4 (*TetPos)[4], __global float4 *VerCrd, 								__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int4 idx;\n" \
"	float4 crd[4];\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the four Tetrahedron vertex indices */\n" \
"	idx = TetVer[i];\n" \
"\n" \
"	/* Copy the vertices coordinates into a temporary buffer */\n" \
"	crd[0] = VerCrd[ idx.s0 ];\n" \
"	crd[1] = VerCrd[ idx.s1 ];\n" \
"	crd[2] = VerCrd[ idx.s2 ];\n" \
"	crd[3] = VerCrd[ idx.s3 ];\n" \
"\n" \
"	/* Compute all three values and store the results in the Tetrahedron scatter-buffer */\n" \
"	TetPos[i][0] = ((float)3*crd[0] + crd[1] + crd[2] + crd[3]) / (float)6;\n" \
"	TetPos[i][1] = ((float)3*crd[1] + crd[2] + crd[3] + crd[0]) / (float)6;\n" \
"	TetPos[i][2] = ((float)3*crd[2] + crd[3] + crd[0] + crd[1]) / (float)6;\n" \
"	TetPos[i][3] = ((float)3*crd[3] + crd[0] + crd[1] + crd[2]) / (float)6;\n" \
"}\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Gather the data stored in the 8 first incident tetrahedra	*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TetrahedraGather1(__global char *VerDeg, __global int16 (*VerBal)[2], __global float4 (*TetPos)[4], 								__global float4 *VerCrd, __global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i, deg;\n" \
"	int16 BalCod, TetIdx, VerIdx;\n" \
"	float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	deg = VerDeg[i];		// get the vertex partial degree: maximum 32\n" \
"	BalCod = VerBal[i][0];	// read a vector containing 16 encoded ball data\n" \
"	TetIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices\n" \
"	VerIdx = BalCod & (int16){7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices\n" \
"\n" \
"	// Sum all coordinates\n" \
"	NewCrd += (deg >  0) ? TetPos[ TetIdx.s0 ][ VerIdx.s0 ] : NulCrd;\n" \
"	NewCrd += (deg >  1) ? TetPos[ TetIdx.s1 ][ VerIdx.s1 ] : NulCrd;\n" \
"	NewCrd += (deg >  2) ? TetPos[ TetIdx.s2 ][ VerIdx.s2 ] : NulCrd;\n" \
"	NewCrd += (deg >  3) ? TetPos[ TetIdx.s3 ][ VerIdx.s3 ] : NulCrd;\n" \
"	NewCrd += (deg >  4) ? TetPos[ TetIdx.s4 ][ VerIdx.s4 ] : NulCrd;\n" \
"	NewCrd += (deg >  5) ? TetPos[ TetIdx.s5 ][ VerIdx.s5 ] : NulCrd;\n" \
"	NewCrd += (deg >  6) ? TetPos[ TetIdx.s6 ][ VerIdx.s6 ] : NulCrd;\n" \
"	NewCrd += (deg >  7) ? TetPos[ TetIdx.s7 ][ VerIdx.s7 ] : NulCrd;\n" \
"	NewCrd += (deg >  8) ? TetPos[ TetIdx.s8 ][ VerIdx.s8 ] : NulCrd;\n" \
"	NewCrd += (deg >  9) ? TetPos[ TetIdx.s9 ][ VerIdx.s9 ] : NulCrd;\n" \
"	NewCrd += (deg > 10) ? TetPos[ TetIdx.sa ][ VerIdx.sa ] : NulCrd;\n" \
"	NewCrd += (deg > 11) ? TetPos[ TetIdx.sb ][ VerIdx.sb ] : NulCrd;\n" \
"	NewCrd += (deg > 12) ? TetPos[ TetIdx.sc ][ VerIdx.sc ] : NulCrd;\n" \
"	NewCrd += (deg > 13) ? TetPos[ TetIdx.sd ][ VerIdx.sd ] : NulCrd;\n" \
"	NewCrd += (deg > 14) ? TetPos[ TetIdx.se ][ VerIdx.se ] : NulCrd;\n" \
"	NewCrd += (deg > 15) ? TetPos[ TetIdx.sf ][ VerIdx.sf ] : NulCrd;\n" \
"\n" \
"	BalCod = VerBal[i][1];	// read a vector containing the next 16 encoded ball data\n" \
"	TetIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices\n" \
"	VerIdx = BalCod & (int16){7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices\n" \
"\n" \
"	// Sum all coordinates\n" \
"	NewCrd += (deg > 16) ? TetPos[ TetIdx.s0 ][ VerIdx.s0 ] : NulCrd;\n" \
"	NewCrd += (deg > 17) ? TetPos[ TetIdx.s1 ][ VerIdx.s1 ] : NulCrd;\n" \
"	NewCrd += (deg > 18) ? TetPos[ TetIdx.s2 ][ VerIdx.s2 ] : NulCrd;\n" \
"	NewCrd += (deg > 19) ? TetPos[ TetIdx.s3 ][ VerIdx.s3 ] : NulCrd;\n" \
"	NewCrd += (deg > 20) ? TetPos[ TetIdx.s4 ][ VerIdx.s4 ] : NulCrd;\n" \
"	NewCrd += (deg > 21) ? TetPos[ TetIdx.s5 ][ VerIdx.s5 ] : NulCrd;\n" \
"	NewCrd += (deg > 22) ? TetPos[ TetIdx.s6 ][ VerIdx.s6 ] : NulCrd;\n" \
"	NewCrd += (deg > 23) ? TetPos[ TetIdx.s7 ][ VerIdx.s7 ] : NulCrd;\n" \
"	NewCrd += (deg > 24) ? TetPos[ TetIdx.s8 ][ VerIdx.s8 ] : NulCrd;\n" \
"	NewCrd += (deg > 25) ? TetPos[ TetIdx.s9 ][ VerIdx.s9 ] : NulCrd;\n" \
"	NewCrd += (deg > 26) ? TetPos[ TetIdx.sa ][ VerIdx.sa ] : NulCrd;\n" \
"	NewCrd += (deg > 27) ? TetPos[ TetIdx.sb ][ VerIdx.sb ] : NulCrd;\n" \
"	NewCrd += (deg > 28) ? TetPos[ TetIdx.sc ][ VerIdx.sc ] : NulCrd;\n" \
"	NewCrd += (deg > 29) ? TetPos[ TetIdx.sd ][ VerIdx.sd ] : NulCrd;\n" \
"	NewCrd += (deg > 30) ? TetPos[ TetIdx.se ][ VerIdx.se ] : NulCrd;\n" \
"	NewCrd += (deg > 31) ? TetPos[ TetIdx.sf ][ VerIdx.sf ] : NulCrd;\n" \
"\n" \
"	// Compute the average value and store it\n" \
"	VerCrd[i] = NewCrd / (float)deg;\n" \
"}\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Gather the data stored in the remaining tetrahedra		*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TetrahedraGather2(__global int (*ExtDeg)[3], __global int *ExtBal, __global float4 (*TetPos)[4], 								__global float4 *VerCrd, __global GmlParSct *par, const int count )\n" \
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
"	deg = ExtDeg[i][2];		// extra vertex degree above 32\n" \
"	NewCrd = VerCrd[ VerIdx ] * (float4){32,32,32,0};	// restart from the partial calculation done above\n" \
"\n" \
"	for(j=BalAdr; j<BalAdr + deg; j++)\n" \
"	{\n" \
"		BalCod = ExtBal[j];	// read the the encoded ball data\n" \
"		NewCrd += TetPos[ BalCod >> 3 ][ BalCod & 7 ];	// decode and add the coordinates\n" \
"	}\n" \
"\n" \
"	VerCrd[ VerIdx ] = NewCrd / (float)(32 + deg);	// compute the average value and store it\n" \
"}\n" \
"\n" \

;