char *reduce = "\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------------------------*/\n" \
"/*                                                                            */\n" \
"/*                         GPU Meshing Library 2.00                           */\n" \
"/*                                                                            */\n" \
"/*----------------------------------------------------------------------------*/\n" \
"/*                                                                            */\n" \
"/*   Description:       vector reduction procedures                           */\n" \
"/*   Author:            Loic MARECHAL                                         */\n" \
"/*   Creation date:     nov 19 2012                                           */\n" \
"/*   Last modification: feb 06 2017                                           */\n" \
"/*                                                                            */\n" \
"/*----------------------------------------------------------------------------*/\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------------------------*/\n" \
"/* Local defines                                                              */\n" \
"/*----------------------------------------------------------------------------*/\n" \
"\n" \
"#ifndef MAX_WORKGROUP_SIZE\n" \
"#define MAX_WORKGROUP_SIZE 1024\n" \
"#endif\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------------------------*/\n" \
"/* GMLIB parameters structure                                                 */\n" \
"/*----------------------------------------------------------------------------*/\n" \
"\n" \
"typedef struct\n" \
"{\n" \
"   int empty;\n" \
"}GmlParSct;\n" \
"\n" \
"\n" \
"/*----------------------------------------------------------------------------*/\n" \
"/* Reduce the vector through an operator: min,max,L1,L2                       */\n" \
"/*----------------------------------------------------------------------------*/\n" \
"\n" \
"__kernel void reduce_min(  __global float *inp, __global float *out,                            __global GmlParSct *par, int count )\n" \
"{\n" \
"   int i, GloIdx=get_global_id(0), LocIdx=get_local_id(0);\n" \
"   __local float LocVec[ MAX_WORKGROUP_SIZE ];\n" \
"\n" \
"   LocVec[ LocIdx ] = (GloIdx < count) ? inp[ GloIdx ] : 1e37;\n" \
"\n" \
"   barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"   // Then do a partial reduction in local vector\n" \
"   for(i=get_local_size(0)/2; i>0; i=i>>1)\n" \
"   {\n" \
"      if(i > LocIdx)\n" \
"         LocVec[ LocIdx ] = fmin(LocVec[ LocIdx ], LocVec[ LocIdx + i ]);\n" \
"      barrier(CLK_LOCAL_MEM_FENCE);\n" \
"   }\n" \
"\n" \
"   // When the global index reaches the lower part of the vector,\n" \
"   // copy it to global memory\n" \
"   if(!LocIdx)\n" \
"      out[ get_group_id(0) ] = LocVec[0] ? LocVec[0] : 1e37;\n" \
"}\n" \
"\n" \
"\n" \
"__kernel void reduce_max(  __global float *inp, __global float *out,                            __global GmlParSct *par, int count )\n" \
"{\n" \
"   int i, GloIdx=get_global_id(0), LocIdx=get_local_id(0);\n" \
"   __local float LocVec[ MAX_WORKGROUP_SIZE ];\n" \
"\n" \
"   LocVec[ LocIdx ] = (GloIdx < count) ? inp[ GloIdx ] : -1e37;\n" \
"\n" \
"   barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"   // Then do a partial reduction in local vector\n" \
"   for(i=get_local_size(0)/2; i>0; i=i>>1)\n" \
"   {\n" \
"      if(i > LocIdx)\n" \
"         LocVec[ LocIdx ] = fmax(LocVec[ LocIdx ], LocVec[ LocIdx + i ]);\n" \
"      barrier(CLK_LOCAL_MEM_FENCE);\n" \
"   }\n" \
"\n" \
"   // When the global index reaches the lower part of the vector,    // copy it to global memory\n" \
"   if(!LocIdx)\n" \
"      out[ get_group_id(0) ] = LocVec[0] ? LocVec[0] : -1e37;\n" \
"}\n" \
"\n" \
"__kernel void reduce_sum(  __global float *inp, __global float *out,                            __global GmlParSct *par, int count )\n" \
"{\n" \
"   int i, GloIdx=get_global_id(0), LocIdx=get_local_id(0);\n" \
"   __local float LocVec[ MAX_WORKGROUP_SIZE ];\n" \
"\n" \
"   LocVec[ LocIdx ] = (GloIdx < count) ? inp[ GloIdx ] : 0;\n" \
"\n" \
"   barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"   // Then do a partial reduction in local vector\n" \
"   for(i=get_local_size(0)/2; i>0; i=i>>1)\n" \
"   {\n" \
"      LocVec[ LocIdx ] += (i > LocIdx) ? LocVec[ LocIdx + i ] : 0;\n" \
"      barrier(CLK_LOCAL_MEM_FENCE);\n" \
"   }\n" \
"\n" \
"   // When the global index reaches the lower part of the vector,    // copy it to global memory\n" \
"\n" \
"   if(!LocIdx)\n" \
"      out[ get_group_id(0) ] = LocVec[0];\n" \
"}\n" \
"\n" \

;