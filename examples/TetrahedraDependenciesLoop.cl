

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
/* Scatter the tetrahedra data in local buffers				*/
/*----------------------------------------------------------*/

__kernel void TetrahedraScatter(__global int4 *TetVer, __global float4 (*TetPos)[4], __global float4 *VerCrd, \
								__global GmlParSct *par, const int count )
{
	int i;
	int4 idx;
	float4 crd[4];

	i = get_global_id(0);

	if(i >= count)
		return;

	/* Get the four Tetrahedron vertex indices */
	idx = TetVer[i];

	/* Copy the vertices coordinates into a temporary buffer */
	crd[0] = VerCrd[ idx.s0 ];
	crd[1] = VerCrd[ idx.s1 ];
	crd[2] = VerCrd[ idx.s2 ];
	crd[3] = VerCrd[ idx.s3 ];

	/* Compute all three values and store the results in the Tetrahedron scatter-buffer */
	TetPos[i][0] = ((float)3*crd[0] + crd[1] + crd[2] + crd[3]) / (float)6;
	TetPos[i][1] = ((float)3*crd[1] + crd[2] + crd[3] + crd[0]) / (float)6;
	TetPos[i][2] = ((float)3*crd[2] + crd[3] + crd[0] + crd[1]) / (float)6;
	TetPos[i][3] = ((float)3*crd[3] + crd[0] + crd[1] + crd[2]) / (float)6;
}


/*----------------------------------------------------------*/
/* Gather the data stored in the 8 first incident tetrahedra	*/
/*----------------------------------------------------------*/

__kernel void TetrahedraGather1(__global char *VerDeg, __global int16 (*VerBal)[2], __global float4 (*TetPos)[4], \
								__global float4 *VerCrd, __global GmlParSct *par, const int count )
{
	int i, deg;
	int16 BalCod, TetIdx, VerIdx;
	float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};

	i = get_global_id(0);

	if(i >= count)
		return;

	deg = VerDeg[i];		// get the vertex partial degree: maximum 32
	BalCod = VerBal[i][0];	// read a vector containing 16 encoded ball data
	TetIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices
	VerIdx = BalCod & (int16){7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices

	// Sum all coordinates
	NewCrd += (deg >  0) ? TetPos[ TetIdx.s0 ][ VerIdx.s0 ] : NulCrd;
	NewCrd += (deg >  1) ? TetPos[ TetIdx.s1 ][ VerIdx.s1 ] : NulCrd;
	NewCrd += (deg >  2) ? TetPos[ TetIdx.s2 ][ VerIdx.s2 ] : NulCrd;
	NewCrd += (deg >  3) ? TetPos[ TetIdx.s3 ][ VerIdx.s3 ] : NulCrd;
	NewCrd += (deg >  4) ? TetPos[ TetIdx.s4 ][ VerIdx.s4 ] : NulCrd;
	NewCrd += (deg >  5) ? TetPos[ TetIdx.s5 ][ VerIdx.s5 ] : NulCrd;
	NewCrd += (deg >  6) ? TetPos[ TetIdx.s6 ][ VerIdx.s6 ] : NulCrd;
	NewCrd += (deg >  7) ? TetPos[ TetIdx.s7 ][ VerIdx.s7 ] : NulCrd;
	NewCrd += (deg >  8) ? TetPos[ TetIdx.s8 ][ VerIdx.s8 ] : NulCrd;
	NewCrd += (deg >  9) ? TetPos[ TetIdx.s9 ][ VerIdx.s9 ] : NulCrd;
	NewCrd += (deg > 10) ? TetPos[ TetIdx.sa ][ VerIdx.sa ] : NulCrd;
	NewCrd += (deg > 11) ? TetPos[ TetIdx.sb ][ VerIdx.sb ] : NulCrd;
	NewCrd += (deg > 12) ? TetPos[ TetIdx.sc ][ VerIdx.sc ] : NulCrd;
	NewCrd += (deg > 13) ? TetPos[ TetIdx.sd ][ VerIdx.sd ] : NulCrd;
	NewCrd += (deg > 14) ? TetPos[ TetIdx.se ][ VerIdx.se ] : NulCrd;
	NewCrd += (deg > 15) ? TetPos[ TetIdx.sf ][ VerIdx.sf ] : NulCrd;

	BalCod = VerBal[i][1];	// read a vector containing the next 16 encoded ball data
	TetIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices
	VerIdx = BalCod & (int16){7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices

	// Sum all coordinates
	NewCrd += (deg > 16) ? TetPos[ TetIdx.s0 ][ VerIdx.s0 ] : NulCrd;
	NewCrd += (deg > 17) ? TetPos[ TetIdx.s1 ][ VerIdx.s1 ] : NulCrd;
	NewCrd += (deg > 18) ? TetPos[ TetIdx.s2 ][ VerIdx.s2 ] : NulCrd;
	NewCrd += (deg > 19) ? TetPos[ TetIdx.s3 ][ VerIdx.s3 ] : NulCrd;
	NewCrd += (deg > 20) ? TetPos[ TetIdx.s4 ][ VerIdx.s4 ] : NulCrd;
	NewCrd += (deg > 21) ? TetPos[ TetIdx.s5 ][ VerIdx.s5 ] : NulCrd;
	NewCrd += (deg > 22) ? TetPos[ TetIdx.s6 ][ VerIdx.s6 ] : NulCrd;
	NewCrd += (deg > 23) ? TetPos[ TetIdx.s7 ][ VerIdx.s7 ] : NulCrd;
	NewCrd += (deg > 24) ? TetPos[ TetIdx.s8 ][ VerIdx.s8 ] : NulCrd;
	NewCrd += (deg > 25) ? TetPos[ TetIdx.s9 ][ VerIdx.s9 ] : NulCrd;
	NewCrd += (deg > 26) ? TetPos[ TetIdx.sa ][ VerIdx.sa ] : NulCrd;
	NewCrd += (deg > 27) ? TetPos[ TetIdx.sb ][ VerIdx.sb ] : NulCrd;
	NewCrd += (deg > 28) ? TetPos[ TetIdx.sc ][ VerIdx.sc ] : NulCrd;
	NewCrd += (deg > 29) ? TetPos[ TetIdx.sd ][ VerIdx.sd ] : NulCrd;
	NewCrd += (deg > 30) ? TetPos[ TetIdx.se ][ VerIdx.se ] : NulCrd;
	NewCrd += (deg > 31) ? TetPos[ TetIdx.sf ][ VerIdx.sf ] : NulCrd;

	// Compute the average value and store it
	VerCrd[i] = NewCrd / (float)deg;
}


/*----------------------------------------------------------*/
/* Gather the data stored in the remaining tetrahedra		*/
/*----------------------------------------------------------*/

__kernel void TetrahedraGather2(__global int (*ExtDeg)[3], __global int *ExtBal, __global float4 (*TetPos)[4], \
								__global float4 *VerCrd, __global GmlParSct *par, const int count )
{
	int i, j, deg, VerIdx, BalCod, BalAdr;
	float4 NewCrd;

	i = get_global_id(0);

	if(i >= count)
		return;

	VerIdx = ExtDeg[i][0];	// get the vertex global index
	BalAdr = ExtDeg[i][1];	// adress of the first encoded ball data
	deg = ExtDeg[i][2];		// extra vertex degree above 32
	NewCrd = VerCrd[ VerIdx ] * (float4){32,32,32,0};	// restart from the partial calculation done above

	for(j=BalAdr; j<BalAdr + deg; j++)
	{
		BalCod = ExtBal[j];	// read the the encoded ball data
		NewCrd += TetPos[ BalCod >> 3 ][ BalCod & 7 ];	// decode and add the coordinates
	}

	VerCrd[ VerIdx ] = NewCrd / (float)(32 + deg);	// compute the average value and store it
}
