

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 2.00                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       loop on elements with indirect writes to vertices     */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     dec 03 2012                                           */
/*   Last modification: feb 04 2017                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmeshb7.h>
#include <gmlib2.h>
#include "QuadsDependenciesLoop.h"


/*----------------------------------------------------------------------------*/
/* Read a mesh, send the data on the GPU, smooth the                          */
/* coordinates and get back the results.                                      */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int iter, i, j, NmbVer, NmbQad, ver=0, dim=0, ref;
   int (*QadTab)[4]=NULL, GpuIdx=0, VerIdx, QadIdx, CalPos;
   int CalCrd1, CalCrd2, BalIdx, PosIdx, NmbItr=100;
   int64_t InpMsh;
   float (*VerTab)[3]=NULL, dummy, FltTab[8], chk=0.;
   double GpuTim, total=0;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nQuadsDependenciesLoop GPU_index");
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
   if( !(InpMsh = GmfOpenMesh("quads.meshb", GmfRead, &ver, &dim)) \
   || (ver != GmfFloat) || (dim != 3) )
   {
      puts("Could not open quads.meshb");
      puts("Please run the command in the same directory this mesh is located");
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) \
   || !(VerTab = malloc((NmbVer+1) * 3 * sizeof(float))) )
   {
      return(1);
   }

   if( !(NmbQad = GmfStatKwd(InpMsh, GmfQuadrilaterals)) \
   || !(QadTab = malloc((NmbQad+1) * 4 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGotoKwd(InpMsh, GmfVertices);
   for(i=1;i<=NmbVer;i++)
      GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

   // Read the elements
   GmfGotoKwd(InpMsh, GmfQuadrilaterals);
   for(i=1;i<=NmbQad;i++)
      GmfGetLin(  InpMsh, GmfQuadrilaterals, &QadTab[i][0], &QadTab[i][1], \
                  &QadTab[i][2], &QadTab[i][3], &ref );

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

   if(!(CalPos = GmlNewKernel(QuadsDependenciesLoop, "QuadsScatter")))
      return(1);

   if(!(CalCrd1 = GmlNewKernel(QuadsDependenciesLoop, "QuadsGather1")))
      return(1);

   if(!(CalCrd2 = GmlNewKernel(QuadsDependenciesLoop, "QuadsGather2")))
      return(1);

   // Create a vertices data type and transfer the data to the GPU
   if(!(VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInout)))
      return(1);

   for(i=1;i<=NmbVer;i++)
      GmlSetVertex(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2]);

   GmlUploadData(VerIdx);

   // Do the same with the elements
   if(!(QadIdx = GmlNewData(GmlQuadrilaterals, NmbQad, 0, GmlInput)))
      return(1);

   for(i=1;i<=NmbQad;i++)
      GmlSetQuadrilateral( QadIdx, i-1, QadTab[i][0]-1, QadTab[i][1]-1, \
                           QadTab[i][2]-1, QadTab[i][3]-1 );

   GmlUploadData(QadIdx);

   // Create a data type with the list of incident elements to each vertices
   BalIdx = GmlNewBall(VerIdx, QadIdx);
   GmlUploadBall(BalIdx);

   // Create a raw datatype to store the elements scatter data.
   // It does not need to be tranfered to the GPU
   if(!(PosIdx = GmlNewData(GmlRawData, NmbQad, 4*sizeof(cl_float4), GmlOutput)))
      return(1);

   // Smooth the coordinates a 100 times
   for(iter=1;iter<=NmbItr;iter++)
   {
      // SCATTER: Compute the new verties coordinates on the GPU
      // but store them in an element based local buffer
      GpuTim = GmlLaunchKernel(CalPos, NmbQad, 3, QadIdx, PosIdx, VerIdx);
    
      if(GpuTim < 0)
         return(1);
    
      total += GpuTim;
    
      // GATHER: compute the average local element coordinates for each vertices
      GpuTim = GmlLaunchBallKernel(CalCrd1, CalCrd2, BalIdx, 2, PosIdx, VerIdx);
    
      if(GpuTim < 0)
         return(1);
    
      total += GpuTim;
   }

   // Get the results back and print some stats
   GmlDownloadData(VerIdx);

   for(i=1;i<=NmbVer;i++)
   {
      GmlGetVertex(VerIdx, i-1, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2]);
      chk += sqrt(VerTab[i][0] * VerTab[i][0] \
               +  VerTab[i][1] * VerTab[i][1] \
               +  VerTab[i][2] * VerTab[i][2]);
   }

   printf("%d coordinates smoothed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbVer*NmbItr, total, GmlGetMemoryUsage()/1048576, \
      GmlGetMemoryTransfer()/1048576, chk / NmbVer );


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(QadIdx);
   GmlFreeData(PosIdx);
   GmlFreeBall(BalIdx);
   GmlStop();

   free(QadTab);
   free(VerTab);

   return(0);
}
