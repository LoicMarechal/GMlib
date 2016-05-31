

/*----------------------------------------------------------*/
/*															*/
/*						GMLIB 2.0							*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Loop over quads and					*/
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
/* Compute each quads middle position						*/
/*----------------------------------------------------------*/

__kernel void QuadsBasic(	__global int4 *QadVer, __global float4 *MidQad, __global float4 *VerCrd, \
							__global GmlParSct *par, const int count )
{
	int i;
	int4 idx;

	i = get_global_id(0);

	if(i >= count)
		return;

	/* Get the four quads vertex indices */
	idx = QadVer[i];

	/* Get four vertices coordinates, compute and store the quads middle */
	MidQad[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ] + VerCrd[ idx.s2 ] + VerCrd[ idx.s3 ]) * (float4){.25,.25,.25,0};
}
