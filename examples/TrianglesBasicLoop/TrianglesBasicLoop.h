char *TrianglesBasicLoop = "\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*						GMLIB 2.0							*/\n" \
"/*															*/\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*	Description:		Loop over triangles and				*/\n" \
"/*	 					access vertices						*/\n" \
"/*	Author:				Loic MARECHAL						*/\n" \
"/*	Creation date:		nov 26 2012							*/\n" \
"/*	Last modification:	nov 30 2012							*/\n" \
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
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Compute each triangles middle position					*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TrianglesBasic(	__global int4 *TriVer, __global float4 *MidTri, __global float4 *VerCrd, 								__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int4 idx;\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the three triangle vertex indices */\n" \
"	idx = TriVer[i];\n" \
"\n" \
"	/* Get three vertices coordinates, compute and store the triangle middle */\n" \
"	MidTri[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ] + VerCrd[ idx.s2 ]) * (float4){.333333,.333333,.333333,0};\n" \
"}\n" \
"\n" \

;