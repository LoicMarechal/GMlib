

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                                  GMLIB 2.0                                 */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Loop over the element, compute some values for each   */
/*                      vertices and store them in an element scatter-buffer. */
/*                      Then loop over the vertices and gather the values     */
/*                      stored in each buffers of the ball.                   */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     nov 26 2012                                           */
/*   Last modification: feb 05 2017                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* GMLIB parameters structure                                                 */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int empty;
}GmlParSct;

#ifndef mix
#define mix(x,y,a) (x+(y-x)*a)
#endif


/*----------------------------------------------------------------------------*/
/* Scatter the triangles data in local buffers                                */
/*----------------------------------------------------------------------------*/

__kernel void TrianglesScatter(  __global int4 *TriVer, __global float4 (*TriPos)[3], \
                                 __global float4 *VerCrd, __global GmlParSct *par, \
                                 const int count )
{
   int i;
   int4 idx;
   float4 crd[3];

   i = get_global_id(0);

   if(i >= count)
      return;

   // Get the three triangle vertex indices
   idx = TriVer[i];

   // Copy the vertices coordinates into a temporary buffer
   crd[0] = VerCrd[ idx.s0 ];
   crd[1] = VerCrd[ idx.s1 ];
   crd[2] = VerCrd[ idx.s2 ];

   // Compute all three values and store the results in the triangle scatter-buffer
   TriPos[i][0] = ((float4){2,2,2,0}*crd[0] + crd[1] + crd[2]) * (float4){.25, .25, .25, 0};
   TriPos[i][1] = ((float4){2,2,2,0}*crd[1] + crd[2] + crd[0]) * (float4){.25, .25, .25, 0};
   TriPos[i][2] = ((float4){2,2,2,0}*crd[2] + crd[0] + crd[1]) * (float4){.25, .25, .25, 0};
}


/*----------------------------------------------------------------------------*/
/* Gather the data stored in the 8 first incident triangles                   */
/*----------------------------------------------------------------------------*/

__kernel void TrianglesGather1(  __global char *VerDeg, __global int8 *VerBal, \
                                 __global float4 (*TriPos)[3], __global float4 *VerCrd, \
                                 __global GmlParSct *par, const int count )
{
   int i, deg;
   int8 BalCod, TriIdx, VerIdx;
   float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};

   i = get_global_id(0);

   if(i >= count)
      return;

   deg = VerDeg[i];        // get the vertex partial degree: maximum 8
   BalCod = VerBal[i];     // read a vector containing 8 encoded ball data
   TriIdx = BalCod >> 3;   // divide each codes by 8 to get the elements indices
   // do a logical and to extract the local vertex indices
   VerIdx = BalCod & (int8){7,7,7,7,7,7,7,7};

   // Sum all coordinates
   NewCrd += (deg >  0) ? TriPos[ TriIdx.s0 ][ VerIdx.s0 ] : NulCrd;
   NewCrd += (deg >  1) ? TriPos[ TriIdx.s1 ][ VerIdx.s1 ] : NulCrd;
   NewCrd += (deg >  2) ? TriPos[ TriIdx.s2 ][ VerIdx.s2 ] : NulCrd;
   NewCrd += (deg >  3) ? TriPos[ TriIdx.s3 ][ VerIdx.s3 ] : NulCrd;
   NewCrd += (deg >  4) ? TriPos[ TriIdx.s4 ][ VerIdx.s4 ] : NulCrd;
   NewCrd += (deg >  5) ? TriPos[ TriIdx.s5 ][ VerIdx.s5 ] : NulCrd;
   NewCrd += (deg >  6) ? TriPos[ TriIdx.s6 ][ VerIdx.s6 ] : NulCrd;
   NewCrd += (deg >  7) ? TriPos[ TriIdx.s7 ][ VerIdx.s7 ] : NulCrd;

   // Compute the average value and store it
   VerCrd[i] = NewCrd / (float)deg;
}


/*----------------------------------------------------------------------------*/
/* Gather the data stored in the remaining triangles                          */
/*----------------------------------------------------------------------------*/

__kernel void TrianglesGather2(  __global int (*ExtDeg)[3], __global int *ExtBal, \
                                 __global float4 (*TriPos)[3], __global float4 *VerCrd, \
                                 __global GmlParSct *par, const int count )
{
   int i, j, deg, VerIdx, BalCod, BalAdr;
   float4 NewCrd;

   i = get_global_id(0);

   if(i >= count)
      return;

   VerIdx = ExtDeg[i][0];   // get the vertex global index
   BalAdr = ExtDeg[i][1];   // adress of the first encoded ball data
   deg = ExtDeg[i][2];      // extra vertex degree above 8
   // restart from the partial calculation done above
   NewCrd = VerCrd[ VerIdx ] * (float4){8,8,8,0};

   for(j=BalAdr; j<BalAdr + deg; j++)
   {
      BalCod = ExtBal[j];   // read the the encoded ball data
      NewCrd += TriPos[ BalCod >> 3 ][ BalCod & 7 ];   // decode and add the coordinates
   }

   VerCrd[ VerIdx ] = NewCrd / (float)(8 + deg);   // compute the average value and store it
}
