

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
/* Scatter the quads data in local buffers                                    */
/*----------------------------------------------------------------------------*/

__kernel void QuadsScatter(__global int4 *QadVer, __global float4 (*QadPos)[4],\
                           __global float4 *VerCrd, __global GmlParSct *par, \
                           const int count )
{
   int i;
   int4 idx;
   float4 crd[4];

   i = get_global_id(0);

   if(i >= count)
      return;

   // Get the four Qadrahedron vertex indices
   idx = QadVer[i];

   // Copy the vertices coordinates into a temporary buffer
   crd[0] = VerCrd[ idx.s0 ];
   crd[1] = VerCrd[ idx.s1 ];
   crd[2] = VerCrd[ idx.s2 ];
   crd[3] = VerCrd[ idx.s3 ];

   // Compute all three values and store the results in the quad scatter-buffer
   QadPos[i][0] = ((float)3*crd[0] + crd[1] + crd[2] + crd[3]) / (float)6;
   QadPos[i][1] = ((float)3*crd[1] + crd[2] + crd[3] + crd[0]) / (float)6;
   QadPos[i][2] = ((float)3*crd[2] + crd[3] + crd[0] + crd[1]) / (float)6;
   QadPos[i][3] = ((float)3*crd[3] + crd[0] + crd[1] + crd[2]) / (float)6;
}


/*----------------------------------------------------------------------------*/
/* Gather the data stored in the 4 first incident quads                       */
/*----------------------------------------------------------------------------*/

__kernel void QuadsGather1(__global char *VerDeg, __global int4 *VerBal, \
                           __global float4 (*QadPos)[4], __global float4 *VerCrd,\
                           __global GmlParSct *par, const int count )
{
   int i, deg;
   int4 BalCod, QadIdx, VerIdx;
   float4 NewCrd = (float4){0,0,0,0}, NulCrd = (float4){0,0,0,0};

   i = get_global_id(0);

   if(i >= count)
      return;

   deg = VerDeg[i];        // get the vertex partial degree: maximum 4
   BalCod = VerBal[i];     // read a vector containing 4 encoded ball data
   QadIdx = BalCod >> 3;   // divide each codes by 8 to get the elements indices
   VerIdx = BalCod & (int4){7,7,7,7};   // do a logical and to extract the local vertex indices

   // Sum all coordinates
   NewCrd += (deg >  0) ? QadPos[ QadIdx.s0 ][ VerIdx.s0 ] : NulCrd;
   NewCrd += (deg >  1) ? QadPos[ QadIdx.s1 ][ VerIdx.s1 ] : NulCrd;
   NewCrd += (deg >  2) ? QadPos[ QadIdx.s2 ][ VerIdx.s2 ] : NulCrd;
   NewCrd += (deg >  3) ? QadPos[ QadIdx.s3 ][ VerIdx.s3 ] : NulCrd;

   // Compute the average value and store it
   VerCrd[i] = NewCrd / (float)deg;
}


/*----------------------------------------------------------------------------*/
/* Gather the data stored in the remaining quads                              */
/*----------------------------------------------------------------------------*/

__kernel void QuadsGather2(__global int (*ExtDeg)[3], __global int *ExtBal, \
                           __global float4 (*QadPos)[4], __global float4 *VerCrd, \
                           __global GmlParSct *par, const int count )
{
   int i, j, deg, VerIdx, BalCod, BalAdr;
   float4 NewCrd;

   i = get_global_id(0);

   if(i >= count)
      return;

   VerIdx = ExtDeg[i][0];   // get the vertex global index
   BalAdr = ExtDeg[i][1];   // adress of the first encoded ball data
   deg = ExtDeg[i][2];      // extra vertex degree above 4
   NewCrd = VerCrd[ VerIdx ] * (float4){4,4,4,0};   // restart from the partial calculation done above

   for(j=BalAdr; j<BalAdr + deg; j++)
   {
      BalCod = ExtBal[j];   // read the the encoded ball data
      NewCrd += QadPos[ BalCod >> 3 ][ BalCod & 7 ];   // decode and add the coordinates
   }

   VerCrd[ VerIdx ] = NewCrd / (float)(4 + deg);   // compute the average value and store it
}
