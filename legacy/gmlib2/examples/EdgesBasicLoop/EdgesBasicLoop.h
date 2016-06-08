char *EdgesBasicLoop = "\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*						GMLIB 2.0							*/\n" \
"/*															*/\n" \
"/*----------------------------------------------------------*/\n" \
"/*															*/\n" \
"/*	Description:		Loop over the elements and read		*/\n" \
"/*						vertices data						*/\n" \
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
"\n" \
"/*----------------------------------------------------------*/\n" \
"/* Compute each elements middle position					*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void EdgesBasic(	__global int2 *EdgVer, __global float4 *MidEdg, __global float4 *VerCrd, 							__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int2 idx;\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the two element vertex indices */\n" \
"	idx = EdgVer[i];\n" \
"\n" \
"	/* Get both vertices coordinates, compute and store the element middle */\n" \
"	MidEdg[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ]) * (float4){.5,.5,.5,0};\n" \
"}\n" \
"\n" \

;