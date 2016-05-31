char *TetrahedraBasicLoop = "\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*						GMLIB 2.0							*/\n" \
"/*															*/\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*	Description:		Loop over tetrahedra				*/\n" \
"/*						and access vertices					*/\n" \
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
"/* Compute each tetrahedra middle position					*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void TetrahedraBasic(	__global int4 *TetVer, __global float4 *MidTet, __global float4 *VerCrd, 								__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int4 idx;\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the four tetrahedra vertex indices */\n" \
"	idx = TetVer[i];\n" \
"\n" \
"	/* Get all four vertices coordinates, compute and store the tetrahedron middle */\n" \
"	MidTet[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ] + VerCrd[ idx.s2 ] + VerCrd[ idx.s3 ]) * (float4){.25,.25,.25,0};\n" \
"}\n" \
"\n" \

;