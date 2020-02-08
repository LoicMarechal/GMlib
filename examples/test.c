

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
   int i, NmbVer, NmbTet, ver=0, dim=0, ref, (*TetTab)[4]=NULL;
   int VerIdx, TetIdx, BalIdx, MidIdx, SolIdx, CalMid, OptVer, GpuIdx=0;
   int64_t InpMsh;
   float (*VerTab)[3]=NULL, (*MidTab)[4], dummy, chk=0.;
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
   if( !(InpMsh = GmfOpenMesh("tetrahedra.meshb", GmfRead, &ver, &dim))
   || (ver != GmfFloat) || (dim != 3) )
   {
      puts("Could not open tetrahedra.meshb");
      puts("Please run the command in the same directory this mesh is located");
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) \
   || !(VerTab = malloc(NmbVer * 3 * sizeof(float))) )
   {
      return(1);
   }

   if( !(NmbTet = GmfStatKwd(InpMsh, GmfTetrahedra)) \
   || !(TetTab = malloc(NmbTet * 4 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGotoKwd(InpMsh, GmfVertices);
   for(i=0;i<NmbVer;i++)
      GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

   // Read the elements
   GmfGotoKwd(InpMsh, GmfTetrahedra);
   for(i=0;i<NmbTet;i++)
      GmfGetLin(  InpMsh, GmfTetrahedra, &TetTab[i][0], &TetTab[i][1], \
                  &TetTab[i][2], &TetTab[i][3], &ref );

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

   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(VerIdx, i, VerTab[i][0], VerTab[i][1], VerTab[i][2], 0);

   // Do the same with the elements
   if(!(TetIdx = GmlNewMeshData(GmlTetrahedra, NmbTet)))
      return(1);

   for(i=0;i<NmbTet;i++)
      GmlSetDataLine(TetIdx, i,
                     TetTab[i][0]-1, TetTab[i][1]-1,
                     TetTab[i][2]-1, TetTab[i][3]-1, 0);

   // Create a raw datatype to store some value at vertices.
   if(!(SolIdx = GmlNewSolutionData(GmlVertices, 2, GmlFlt4, "SolAtVer")))
      return(1);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewSolutionData(GmlTetrahedra, 1, GmlFlt4, "TetMid")))
      return(1);

   if(!(MidTab = malloc((NmbTet+1)*4*sizeof(float))))
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

   // Launch the kernel on the GPU
   res  = GmlLaunchKernel(CalMid);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %g\n", CalMid, res);
      exit(0);
   }

   GpuTim += res;

   /*res = GmlLaunchKernel(OptVer);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %g\n", OptVer, res);
      exit(0);
   }

   GpuTim += res;*/

   for(i=0;i<NmbTet;i++)
   {
      GmlGetDataLine(MidIdx, i, MidTab[i]);
      chk += sqrt(MidTab[i][0] * MidTab[i][0] \
               +  MidTab[i][1] * MidTab[i][1] \
               +  MidTab[i][2] * MidTab[i][2]);
   }

   printf("%d tet centers computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbTet, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbTet );


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
