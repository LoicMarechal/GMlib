

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
/* Compute each elements middle position                                      */
/*----------------------------------------------------------------------------*/

__kernel void EdgesBasic(  __global int2 *EdgVer, __global float4 *MidEdg, \
                           __global float4 *VerCrd, __global GmlParSct *par, \
                           const int count )
{
   int i;
   int2 idx;

   i = get_global_id(0);

   if(i >= count)
      return;

   // Get the two element vertex indices
   idx = EdgVer[i];

   // Get both vertices coordinates, compute and store the element middle
   MidEdg[i] = (VerCrd[ idx.s0 ] + VerCrd[ idx.s1 ]) * (float4){.5,.5,.5,0};
}
