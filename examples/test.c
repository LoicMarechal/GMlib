

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
   int i, j, NmbVer, NmbTet, ver=0, dim=0, *VerRef, (*TetTab)[5]=NULL;
   int VerIdx, TetIdx, BalIdx, MidIdx, SolIdx, CalMid, OptVer, GpuIdx=0;
   int64_t InpMsh;
   float (*VerTab)[3]=NULL, (*MidTab)[4], (*SolTab)[8], dummy, chk=0.;
   float NulSol[8] = {.125, .125, .125, .125, .125, .125, .125, .125};
   double GpuTim = 0., res;


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

   /*for(i=1;i<=NmbTet;i++)
      printf(  "read ele %d: %d %d %d %d\n", i,
               TetTab[i][0],TetTab[i][1],TetTab[i][2],TetTab[i][3] );*/

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

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
      GmlSetDataLine(SolIdx, i, &NulSol);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewSolutionData(GmlTetrahedra, 1, GmlFlt4, "TetMid")))
      return(1);

   // Assemble and compile the scatter kernel
   CalMid = GmlCompileKernel( TetrahedraLoop, "TetrahedraBasic",
                              Parameters, GmlTetrahedra, 3,
                              VerIdx, GmlReadMode,  NULL,
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

   //exit(0);

   // Launch the tetrahedra kernel on the GPU
   res  = GmlLaunchKernel(CalMid);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %g\n", CalMid, res);
      exit(0);
   }

   GpuTim += res;

   // Launch the vertex kernel on the GPU
   res = GmlLaunchKernel(OptVer);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %g\n", OptVer, res);
      exit(0);
   }

   GpuTim += res;

   // Get the data back from the GPU memory
   if(!(MidTab = malloc( NmbTet * 4 * sizeof(float))) )
      return(1);

   for(i=0;i<NmbTet;i++)
   {
      GmlGetDataLine(MidIdx, i, MidTab[i]);
      chk += sqrt(MidTab[i][0] * MidTab[i][0]
               +  MidTab[i][1] * MidTab[i][1]
               +  MidTab[i][2] * MidTab[i][2]);
   }

   printf("%d tet centers computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbTet, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbTet );

   if(!(SolTab = calloc( NmbVer, 8 * sizeof(float))) )
      return(1);

   chk = 0.;

   for(i=0;i<NmbVer;i++)
   {
      GmlGetDataLine(SolIdx, i, SolTab[i]);
      /*printf("vertex %d : deg=%d, chk=%d %d %d %d\n", i+1,
               (int)SolTab[i][0], (int)SolTab[i][4]+1, (int)SolTab[i][5]+1, (int)SolTab[i][6]+1, (int)SolTab[i][7]+1);*/

      for(j=0;j<8;j++)
         chk += SolTab[i][j];
   }

   printf("SolAtVertex checksum = %g\n",chk / NmbVer);


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(TetIdx);
   GmlFreeData(MidIdx);
   GmlStop();

   free(MidTab);
   free(TetTab);
   free(VerTab);

   return(0);
}
