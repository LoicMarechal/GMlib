char *HexahedraBasicLoop = "\n" \
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
"/* Compute each hexahedra middle position					*/\n" \
"/*----------------------------------------------------------*/\n" \
"\n" \
"__kernel void HexahedraBasic(	__global int8 *HexVer, __global float4 *MidHex, __global float4 *VerCrd, 								__global GmlParSct *par, const int count )\n" \
"{\n" \
"	int i;\n" \
"	int8 idx;\n" \
"\n" \
"	i = get_global_id(0);\n" \
"\n" \
"	if(i >= count)\n" \
"		return;\n" \
"\n" \
"	/* Get the eight hexahedra vertex indices */\n" \
"	idx = HexVer[i];\n" \
"\n" \
"	/* Get all eight vertices coordinates, compute and store the hexahedron middle */\n" \
"	MidHex[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ] + VerCrd[ idx.s2 ] + VerCrd[ idx.s3 ] + VerCrd[ idx.s4 ] + 				VerCrd[ idx.s5 ] + VerCrd[ idx.s6 ] + VerCrd[ idx.s7 ]) * (float4){.125,.125,.125,0};\n" \
"}\n" \
"\n" \

;