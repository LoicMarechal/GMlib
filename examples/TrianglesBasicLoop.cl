

/*----------------------------------------------------------*/
/*															*/
/*						GMLIB 2.0							*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Loop over triangles and				*/
/*	 					access vertices						*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		nov 26 2012							*/
/*	Last modification:	nov 30 2012							*/
/*															*/
/*----------------------------------------------------------*/


/*----------------------------------------------------------*/
/* GMLIB parameters structure								*/
/*----------------------------------------------------------*/

typedef struct
{
	int empty;
}GmlParSct;


/*----------------------------------------------------------*/
/* Compute each triangles middle position					*/
/*----------------------------------------------------------*/

__kernel void TrianglesBasic(	__global int4 *TriVer, __global float4 *MidTri, __global float4 *VerCrd, \
								__global GmlParSct *par, const int count )
{
	int i;
	int4 idx;

	i = get_global_id(0);

	if(i >= count)
		return;

	/* Get the three triangle vertex indices */
	idx = TriVer[i];

	/* Get three vertices coordinates, compute and store the triangle middle */
	MidTri[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ] + VerCrd[ idx.s2 ]) * (float4){.333333,.333333,.333333,0};
}
