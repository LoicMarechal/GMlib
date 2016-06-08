

/*----------------------------------------------------------*/
/*															*/
/*					GPU Meshing Library 2.00				*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		vector reduction procedures			*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		nov 19 2012							*/
/*	Last modification:	nov 29 2012							*/
/*															*/
/*----------------------------------------------------------*/


/*----------------------------------------------------------*/
/* Local defines											*/
/*----------------------------------------------------------*/

#ifndef MAX_WORKGROUP_SIZE
#define MAX_WORKGROUP_SIZE 1024
#endif


/*----------------------------------------------------------*/
/* GMLIB parameters structure								*/
/*----------------------------------------------------------*/

typedef struct
{
	int empty;
}GmlParSct;


/*----------------------------------------------------------*/
/* Reduce the vector through an operator: min,max,L1,L2		*/
/*----------------------------------------------------------*/

__kernel void reduce_min(__global float *inp, __global float *out, __global GmlParSct *par, int count)
{
	int i, GloIdx=get_global_id(0), LocIdx=get_local_id(0);
	__local float LocVec[ MAX_WORKGROUP_SIZE ];

	LocVec[ LocIdx ] = (GloIdx < count) ? inp[ GloIdx ] : 1e37;

	barrier(CLK_LOCAL_MEM_FENCE);

	/* Then do a partial reduction in local vector */

	for(i=get_local_size(0)/2; i>0; i=i>>1)
	{
		if(i > LocIdx)
			LocVec[ LocIdx ] = fmin(LocVec[ LocIdx ], LocVec[ LocIdx + i ]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* When the global index reaches the lower part of the vector, copy it to global memory */

	if(!LocIdx)
		out[ get_group_id(0) ] = LocVec[0] ? LocVec[0] : 1e37;
}


__kernel void reduce_max(__global float *inp, __global float *out, __global GmlParSct *par, int count)
{
	int i, GloIdx=get_global_id(0), LocIdx=get_local_id(0);
	__local float LocVec[ MAX_WORKGROUP_SIZE ];

	LocVec[ LocIdx ] = (GloIdx < count) ? inp[ GloIdx ] : -1e37;

	barrier(CLK_LOCAL_MEM_FENCE);

	/* Then do a partial reduction in local vector */

	for(i=get_local_size(0)/2; i>0; i=i>>1)
	{
		if(i > LocIdx)
			LocVec[ LocIdx ] = fmax(LocVec[ LocIdx ], LocVec[ LocIdx + i ]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* When the global index reaches the lower part of the vector, copy it to global memory */

	if(!LocIdx)
		out[ get_group_id(0) ] = LocVec[0] ? LocVec[0] : -1e37;
}

__kernel void reduce_sum(__global float *inp, __global float *out, __global GmlParSct *par, int count)
{
	int i, GloIdx=get_global_id(0), LocIdx=get_local_id(0);
	__local float LocVec[ MAX_WORKGROUP_SIZE ];

	LocVec[ LocIdx ] = (GloIdx < count) ? inp[ GloIdx ] : 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	/* Then do a partial reduction in local vector */

	for(i=get_local_size(0)/2; i>0; i=i>>1)
	{
		LocVec[ LocIdx ] += (i > LocIdx) ? LocVec[ LocIdx + i ] : 0;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* When the global index reaches the lower part of the vector, copy it to global memory */

	if(!LocIdx)
		out[ get_group_id(0) ] = LocVec[0];
}
