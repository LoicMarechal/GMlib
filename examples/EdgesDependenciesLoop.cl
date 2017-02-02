

/*----------------------------------------------------------*/
/*															*/
/*						GMLIB 2.0							*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Loop over the element, compute some	*/
/*						values for each vertices and store	*/
/*						them in an element scatter-buffer.	*/
/*						Then loop over the vertices and		*/
/*						gather the values stored in each	*/
/*						buffers of the ball.				*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		nov 26 2012							*/
/*	Last modification:	nov 29 2012							*/
/*															*/
/*----------------------------------------------------------*/


/*----------------------------------------------------------*/
/* GMLIB parameters structure								*/
/*----------------------------------------------------------*/

typedef struct
{
	int empty;
}GmlParSct;

#ifndef mix
#define mix(x,y,a) (x+(y-x)*a)
#endif


/*----------------------------------------------------------*/
/* Scatter the elements data in local buffers				*/
/*----------------------------------------------------------*/

__kernel void EdgesScatter(	__global int2 *EdgVer, __global float4 (*EdgPos)[2], __global float4 *VerCrd, \
							__global GmlParSct *par, const int count )
{
	int i;
	int2 idx;
	float a=1./3., b=2./3.;
	float4 crd[2];

	i = get_global_id(0);

	if(i >= count)
		return;

	/* Get the two element vertex indices */
	idx = EdgVer[i];

	/* Copy the vertices coordinates into a temporary buffer */
	crd[0] = VerCrd[ idx.s0 ];
	crd[1] = VerCrd[ idx.s1 ];

	/* Compute both values and store the results in the element scatter-buffer */
	EdgPos[i][0] = mix(crd[0], crd[1], a);
	EdgPos[i][1] = mix(crd[0], crd[1], b);
}


/*----------------------------------------------------------*/
/* Gather the data stored in the 16 first incident elements	*/
/*----------------------------------------------------------*/

__kernel void EdgesGather1(	__global char *VerDeg, __global int16 *VerBal, __global float4 (*EdgPos)[2], \
							__global float4 *VerCrd, __global GmlParSct *par, const int count )
{
	int i, deg;
	int16 BalCod, EdgIdx, VerIdx;
	float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};

	i = get_global_id(0);

	if(i >= count)
		return;

	deg = VerDeg[i];		// get the vertex partial degree: maximum 16
	BalCod = VerBal[i]; 	// read a vector containing 16 encoded ball data
	EdgIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices
	VerIdx = BalCod & (int16){7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices

	// Sum all coordinates
	NewCrd += (deg >  0) ? EdgPos[ EdgIdx.s0 ][ VerIdx.s0 ] : NulCrd;
	NewCrd += (deg >  1) ? EdgPos[ EdgIdx.s1 ][ VerIdx.s1 ] : NulCrd;
	NewCrd += (deg >  2) ? EdgPos[ EdgIdx.s2 ][ VerIdx.s2 ] : NulCrd;
	NewCrd += (deg >  3) ? EdgPos[ EdgIdx.s3 ][ VerIdx.s3 ] : NulCrd;
	NewCrd += (deg >  4) ? EdgPos[ EdgIdx.s4 ][ VerIdx.s4 ] : NulCrd;
	NewCrd += (deg >  5) ? EdgPos[ EdgIdx.s5 ][ VerIdx.s5 ] : NulCrd;
	NewCrd += (deg >  6) ? EdgPos[ EdgIdx.s6 ][ VerIdx.s6 ] : NulCrd;
	NewCrd += (deg >  7) ? EdgPos[ EdgIdx.s7 ][ VerIdx.s7 ] : NulCrd;
	NewCrd += (deg >  8) ? EdgPos[ EdgIdx.s8 ][ VerIdx.s8 ] : NulCrd;
	NewCrd += (deg >  9) ? EdgPos[ EdgIdx.s9 ][ VerIdx.s9 ] : NulCrd;
	NewCrd += (deg > 10) ? EdgPos[ EdgIdx.sa ][ VerIdx.sa ] : NulCrd;
	NewCrd += (deg > 11) ? EdgPos[ EdgIdx.sb ][ VerIdx.sb ] : NulCrd;
	NewCrd += (deg > 12) ? EdgPos[ EdgIdx.sc ][ VerIdx.sc ] : NulCrd;
	NewCrd += (deg > 13) ? EdgPos[ EdgIdx.sd ][ VerIdx.sd ] : NulCrd;
	NewCrd += (deg > 14) ? EdgPos[ EdgIdx.se ][ VerIdx.se ] : NulCrd;
	NewCrd += (deg > 15) ? EdgPos[ EdgIdx.sf ][ VerIdx.sf ] : NulCrd;

	// Compute the average value and store it
	VerCrd[i] = NewCrd / (float)deg;
}


/*----------------------------------------------------------*/
/* Gather the data stored in the remaining elements			*/
/*----------------------------------------------------------*/

__kernel void EdgesGather2(	__global int (*ExtDeg)[3], __global int *ExtBal, __global float4 (*EdgPos)[2], \
							__global float4 *VerCrd, __global GmlParSct *par, const int count )
{
	int i, j, deg, VerIdx, BalCod, BalAdr;
	float4 NewCrd;

	i = get_global_id(0);

	if(i >= count)
		return;

	VerIdx = ExtDeg[i][0];	// get the vertex global index
	BalAdr = ExtDeg[i][1];	// adress of the first encoded ball data
	deg = ExtDeg[i][2];		// extra vertex degree above 16
	NewCrd = VerCrd[ VerIdx ] * (float4){16,16,16,0};	// restart from the partial calculation done above

	for(j=BalAdr; j<BalAdr + deg; j++)
	{
		BalCod = ExtBal[j];	// read the the encoded ball data
		NewCrd += EdgPos[ BalCod >> 3 ][ BalCod & 7 ];	// decode and add the coordinates
	}

	VerCrd[ VerIdx ] = NewCrd / (float)(16 + deg);	// compute the average value and store it
}
