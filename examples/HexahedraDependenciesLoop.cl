

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

__kernel void HexahedraScatter(	__global int8 *HexVer, __global float4 (*HexPos)[8], __global float4 *VerCrd, \
								__global GmlParSct *par, const int count )
{
	int i;
	int8 idx;
	float4 crd[8];

	i = get_global_id(0);

	if(i >= count)
		return;

	/* Get the eight Hexahedron vertex indices */
	idx = HexVer[i];

	/* Copy the vertices coordinates into a temporary buffer */
	crd[0] = VerCrd[ idx.s0 ];
	crd[1] = VerCrd[ idx.s1 ];
	crd[2] = VerCrd[ idx.s2 ];
	crd[3] = VerCrd[ idx.s3 ];
	crd[4] = VerCrd[ idx.s4 ];
	crd[5] = VerCrd[ idx.s5 ];
	crd[6] = VerCrd[ idx.s6 ];
	crd[7] = VerCrd[ idx.s7 ];

	/* Compute all three values and store the results in the Hexahedron scatter-buffer */
	HexPos[i][0] = ((float)3*crd[0] + crd[1] + crd[3] + crd[4]) / (float)6;
	HexPos[i][1] = ((float)3*crd[1] + crd[2] + crd[5] + crd[0]) / (float)6;
	HexPos[i][2] = ((float)3*crd[2] + crd[3] + crd[6] + crd[1]) / (float)6;
	HexPos[i][3] = ((float)3*crd[3] + crd[0] + crd[7] + crd[2]) / (float)6;
	HexPos[i][4] = ((float)3*crd[4] + crd[0] + crd[5] + crd[7]) / (float)6;
	HexPos[i][5] = ((float)3*crd[5] + crd[1] + crd[4] + crd[6]) / (float)6;
	HexPos[i][6] = ((float)3*crd[6] + crd[2] + crd[5] + crd[7]) / (float)6;
	HexPos[i][7] = ((float)3*crd[7] + crd[3] + crd[4] + crd[6]) / (float)6;
}


/*----------------------------------------------------------*/
/* Gather the data stored in the 8 first incident hexahedra	*/
/*----------------------------------------------------------*/

__kernel void HexahedraGather1(__global char *VerDeg, __global int8 *VerBal, __global float4 (*HexPos)[8], \
								__global float4 *VerCrd, __global GmlParSct *par, const int count )
{
	int i, deg;
	int8 BalCod, HexIdx, VerIdx;
	float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};

	i = get_global_id(0);

	if(i >= count)
		return;

	deg = VerDeg[i];		// get the vertex partial degree: maximum 8
	BalCod = VerBal[i]; 	// read a vector containing 8 encoded ball data
	HexIdx = BalCod >> 3;	// divide each codes by 8 to get the elements indices
	VerIdx = BalCod & (int8){7,7,7,7,7,7,7,7};	// do a logical and to extract the local vertex indices

	// Sum all coordinates
	NewCrd += (deg >  0) ? HexPos[ HexIdx.s0 ][ VerIdx.s0 ] : NulCrd;
	NewCrd += (deg >  1) ? HexPos[ HexIdx.s1 ][ VerIdx.s1 ] : NulCrd;
	NewCrd += (deg >  2) ? HexPos[ HexIdx.s2 ][ VerIdx.s2 ] : NulCrd;
	NewCrd += (deg >  3) ? HexPos[ HexIdx.s3 ][ VerIdx.s3 ] : NulCrd;
	NewCrd += (deg >  4) ? HexPos[ HexIdx.s4 ][ VerIdx.s4 ] : NulCrd;
	NewCrd += (deg >  5) ? HexPos[ HexIdx.s5 ][ VerIdx.s5 ] : NulCrd;
	NewCrd += (deg >  6) ? HexPos[ HexIdx.s6 ][ VerIdx.s6 ] : NulCrd;
	NewCrd += (deg >  7) ? HexPos[ HexIdx.s7 ][ VerIdx.s7 ] : NulCrd;

	// Compute the average value and store it
	VerCrd[i] = NewCrd / (float)deg;
}


/*----------------------------------------------------------*/
/* Gather the data stored in the remaining hexahedra		*/
/*----------------------------------------------------------*/

__kernel void HexahedraGather2(	__global int (*ExtDeg)[3], __global int *ExtBal, __global float4 (*HexPos)[8], \
								__global float4 *VerCrd, __global GmlParSct *par, const int count )
{
	int i, j, deg, VerIdx, BalCod, BalAdr;
	float4 NewCrd;

	i = get_global_id(0);

	if(i >= count)
		return;

	VerIdx = ExtDeg[i][0];	// get the vertex global index
	BalAdr = ExtDeg[i][1];	// adress of the first encoded ball data
	deg = ExtDeg[i][2];		// extra vertex degree above 8
	NewCrd = VerCrd[ VerIdx ] * (float4){8,8,8,0};	// restart from the partial calculation done above

	for(j=BalAdr; j<BalAdr + deg; j++)
	{
		BalCod = ExtBal[j];	// read the the encoded ball data
		NewCrd += HexPos[ BalCod >> 3 ][ BalCod & 7 ];	// decode and add the coordinates
	}

	VerCrd[ VerIdx ] = NewCrd / (float)(8 + deg);	// compute the average value and store it
}
