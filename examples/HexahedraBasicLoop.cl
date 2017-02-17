

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                                  GMLIB 2.0                                 */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Loop over the elements and read vertices data         */
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


/*----------------------------------------------------------------------------*/
/* Compute each hexahedra middle position                                     */
/*----------------------------------------------------------------------------*/

__kernel void HexahedraBasic( __global int8 *HexVer, __global float4 *MidHex, \
                              __global float4 *VerCrd, __global GmlParSct *par, \
                              const int count )
{
   int i;
   int8 idx;

   i = get_global_id(0);

   if(i >= count)
      return;

   // Get the eight hexahedra vertex indices
   idx = HexVer[i];

   // Get all eight vertices coordinates, compute and store the hexahedron middle
   MidHex[i] = (  VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ] + VerCrd[ idx.s2 ] \
               +  VerCrd[ idx.s3 ] + VerCrd[ idx.s4 ] + VerCrd[ idx.s5 ] \
               +  VerCrd[ idx.s6 ] + VerCrd[ idx.s7 ]) * (float4){.125,.125,.125,0};
}
