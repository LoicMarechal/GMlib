

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.00                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on elements                                */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     dec 03 2012                                           */
/*   Last modification: jul 16 2018                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmeshb7.h>
#include <gmlib3.h>
#include "HexahedraBasicLoop.h"


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements middle and get back the results.                              */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int i, j, NmbVer, NmbHex, ver=0, dim=0, ref, (*HexTab)[8] = NULL;
   int VerIdx, HexIdx, MidIdx, CalMid, GpuIdx=0;
   int64_t InpMsh;
   float (*VerTab)[4] = NULL, (*MidTab)[4], dummy, chk=0.;
   double GpuTim;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nHexahedraBasicLoop GPU_index");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
      GpuIdx = atoi(ArgVec[1]);


   /*--------------*/
   /* MESH READING */
   /*--------------*/

   // Open the mesh
   if( !(InpMsh = GmfOpenMesh("hexahedra.meshb", GmfRead, &ver, &dim)) \
   || (ver != GmfFloat) || (dim != 3) )
   {
      puts("Could not open hexahedra.meshb");
      puts("Please run the command in the same directory this mesh is located");
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) \
   || !(VerTab = malloc((NmbVer+1) * 4 * sizeof(float))) )
   {
      return(1);
   }

   if( !(NmbHex = GmfStatKwd(InpMsh, GmfHexahedra)) \
   || !(HexTab = malloc((NmbHex+1) * 8 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGotoKwd(InpMsh, GmfVertices);
   for(i=1;i<=NmbVer;i++)
      GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

   // Read the elements
   GmfGotoKwd(InpMsh, GmfHexahedra);
   for(i=1;i<=NmbHex;i++)
   {
      GmfGetLin(  InpMsh, GmfHexahedra, \
                  &HexTab[i][0], &HexTab[i][1], &HexTab[i][2], &HexTab[i][3], \
                  &HexTab[i][4], &HexTab[i][5], &HexTab[i][6], &HexTab[i][7], &ref );

      for(j=0;j<8;j++)
         HexTab[i][j]--;
   }

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

   if(!(CalMid = GmlNewKernel(HexahedraBasicLoop)))
      return(1);

   // Create a vertices data type and transfer the data to the GPU
   if(!(VerIdx = GmlNewData(GmlVertices, "crd", NmbVer)))
      return(1);

   GmlSetDataBlock(VerIdx, VerTab[1], VerTab[ NmbVer ]);

   // Do the same with the elements
   if(!(HexIdx = GmlNewData(GmlHexahedra, "hex", NmbHex)))
      return(1);

   GmlSetDataBlock(HexIdx, HexTab[1], HexTab[ NmbHex ]);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewData(GmlRawData, "mid", NmbHex, GmlHexahedra, 1, "float4", sizeof(cl_float4))))
      return(1);

   if(!(MidTab = malloc( (NmbHex+1) * 4 * sizeof(float))))
      return(1);

   // Launch the kernel on the GPU
   GpuTim = GmlLaunchKernel(CalMid, HexIdx, GmlRead, HexIdx, GmlWrite, MidIdx, GmlRead, VerIdx, GmlEnd);

   if(GpuTim < 0)
      return(1);

   // Get the results back and print some stats
   GmlGetDataBlock(MidIdx, MidTab[1], MidTab[ NmbHex ]);

   for(i=1;i<=NmbHex;i++)
   {
      chk += sqrt(MidTab[i][0] * MidTab[i][0] \
               +  MidTab[i][1] * MidTab[i][1] \
               +  MidTab[i][2] * MidTab[i][2]);
   }

   printf("%d hex middles computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbHex, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbHex );


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(HexIdx);
   GmlFreeData(MidIdx);
   GmlStop();

   free(MidTab);
   free(HexTab);
   free(VerTab);

   return(0);
}
