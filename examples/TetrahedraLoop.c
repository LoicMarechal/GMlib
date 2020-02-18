

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.00                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on tetrahedra                              */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     nov 21 2019                                           */
/*   Last modification: feb 07 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libmeshb7.h"
#include "gmlib3.h"
#include "Parameters.h"
#include "TetrahedraLoop.h"
#include "VertexGather.h"


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements middle and get back the results.                              */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int i, j, NmbVer, NmbTet, ver = 0, dim = 0, *VerRef, (*TetTab)[5];
   int VerIdx, TetIdx, BalIdx, MidIdx, SolIdx, CalMid, OptVer, GpuIdx = 0;
   int64_t InpMsh;
   float MidTab[4], SolTab[8], dummy, TetChk = 0., VerChk = 0., (*VerTab)[3];
   float IniSol[8] = {.125, .125, .125, .125, .125, .125, .125, .125};
   double TetTim = 0., VerTim = 0., res;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nTetrahedraLoop GPU_index");
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
   if( !(InpMsh = GmfOpenMesh("tetrahedra.meshb", GmfRead, &ver, &dim)) || (dim != 3) )
   {
      puts("Could not open tetrahedra.meshb");
      puts("Please run the command in the same directory this mesh is located");
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices))
   ||  !(VerTab = malloc( (NmbVer+1) * 3 * sizeof(float)))
   ||  !(VerRef = malloc( (NmbVer+1)     * sizeof(int))) )
   {
      return(1);
   }

   if( !(NmbTet = GmfStatKwd(InpMsh, GmfTetrahedra))
   ||  !(TetTab = malloc( (NmbTet+1) * 5 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGetBlock(InpMsh, GmfVertices, 2, NmbVer, 0, NULL, NULL,
               GmfFloatVec, 3, VerTab[1],  VerTab[ NmbVer ],
               GmfInt,        &VerRef[1], &VerRef[ NmbVer ]);

   // Read the elements
   GmfGetBlock(InpMsh, GmfTetrahedra, 1, NmbTet, 0, NULL, NULL,
               GmfIntVec, 5, TetTab[1], TetTab[ NmbTet ]);

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

   GmlDebugOn();

   // Create a vertices data type and transfer the data to the GPU
   if(!(VerIdx = GmlNewMeshData(GmlVertices, NmbVer)))
      return(1);

   for(i=1;i<=NmbVer;i++)
      GmlSetDataLine(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2], VerRef[i]);

   // Do the same with the elements
   if(!(TetIdx = GmlNewMeshData(GmlTetrahedra, NmbTet)))
      return(1);

   for(i=1;i<=NmbTet;i++)
      GmlSetDataLine(TetIdx, i-1,
                     TetTab[i][0]-1, TetTab[i][1]-1,
                     TetTab[i][2]-1, TetTab[i][3]-1, TetTab[i][4]);

   // Create a raw datatype to store some value at vertices.
   if(!(SolIdx = GmlNewSolutionData(GmlVertices, 2, GmlFlt4, "SolAtVer")))
      return(1);

   // Fill the initial field with crap
   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(SolIdx, i, &IniSol);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewSolutionData(GmlTetrahedra, 1, GmlFlt4, "TetMid")))
      return(1);

   // Assemble and compile the scatter kernel
   CalMid = GmlCompileKernel( TetrahedraLoop, "TetrahedraBasic",
                              Parameters, GmlTetrahedra, 3,
                              VerIdx, GmlReadMode | GmlRefFlag,  NULL,
                              SolIdx, GmlReadMode,  NULL,
                              MidIdx, GmlWriteMode, NULL );

   if(!CalMid)
      return(1);

   // Assemble and compile the gather kernel
   OptVer = GmlCompileKernel( VertexGather, "VertexGather",
                              Parameters, GmlVertices, 2,
                              SolIdx, GmlWriteMode, NULL,
                              MidIdx, GmlReadMode,  NULL );

   if(!OptVer)
      return(1);

   for(i=1;i<=100;i++)
   {
      // Launch the tetrahedra kernel on the GPU
      res  = GmlLaunchKernel(CalMid);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %g\n", CalMid, res);
         exit(0);
      }

      TetTim += res;

      // Launch the vertex kernel on the GPU
      res = GmlLaunchKernel(OptVer);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %g\n", OptVer, res);
         exit(0);
      }

      VerTim += res;
   }


   /*-----------------*/
   /* GET THE RESULTS */
   /*-----------------*/

   // Get back the MidTet data from the GPU memory and compute a checksum
   for(i=0;i<NmbTet;i++)
   {
      GmlGetDataLine(MidIdx, i, MidTab);
      TetChk += MidTab[0] + MidTab[1] + MidTab[2];
   }

   // Get back the SolAtVer data from the GPU memory and compute a checksum
   for(i=0;i<NmbVer;i++)
   {
      GmlGetDataLine(SolIdx, i, SolTab);
      VerChk += SolTab[0] + SolTab[1] + SolTab[2] + SolTab[3];
   }

   printf("%d tets processed in %g seconds, scater=%g, gather=%g\n",
          NmbTet, TetTim + VerTim, TetTim, VerTim);

   printf("%ld MB used, %ld MB transfered\n",
          GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576);

   printf("MidTet checksum = %g, SolAtVer checksum = %g\n",
         TetChk / NmbTet, VerChk / NmbVer);


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(TetIdx);
   GmlFreeData(MidIdx);
   GmlStop();

   free(TetTab);
   free(VerTab);

   return(0);
}
