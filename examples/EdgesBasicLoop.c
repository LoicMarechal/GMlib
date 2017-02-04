

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                      GPU Meshing Library 2.00                              */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on elements                                */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     dec 03 2012                                           */
/*   Last modification: feb 03 2017                                           */
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
#include "EdgesBasicLoop.h"


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements middle and get back the results.                              */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int i, NmbVer, NmbEdg, ver=0, dim=0, ref, (*EdgTab)[2]=NULL;
   int VerIdx, EdgIdx, MidIdx, CalMid, GpuIdx=0;
   int64_t InpMsh;
   float (*VerTab)[3]=NULL, (*MidTab)[4], dummy, chk=0.;
   double GpuTim;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nEdgesBasicLoop GPU_index");
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
   if( !(InpMsh = GmfOpenMesh("edges.meshb", GmfRead, &ver, &dim)) \
   || (ver != GmfFloat) || (dim != 3) )
   {
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) \
   || !(VerTab = malloc((NmbVer+1) * 3 * sizeof(float))) )
   {
      return(1);
   }

   if( !(NmbEdg = GmfStatKwd(InpMsh, GmfEdges)) \
   || !(EdgTab = malloc((NmbEdg+1) * 2 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGotoKwd(InpMsh, GmfVertices);
   for(i=1;i<=NmbVer;i++)
      GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

   // Read the elements
   GmfGotoKwd(InpMsh, GmfEdges);
   for(i=1;i<=NmbEdg;i++)
      GmfGetLin(InpMsh, GmfEdges, &EdgTab[i][0], &EdgTab[i][1], &ref);

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

   if(!(CalMid = GmlNewKernel(EdgesBasicLoop, "EdgesBasic")))
      return(1);

   // Create a vertices data type and transfer the data to the GPU
   if(!(VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInput)))
      return(1);

   for(i=1;i<=NmbVer;i++)
      GmlSetVertex(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2]);

   GmlUploadData(VerIdx);

   /* Do the same with the elements */
   if(!(EdgIdx = GmlNewData(GmlEdges, NmbEdg, 0, GmlInput)))
      return(1);

   for(i=1;i<=NmbEdg;i++)
      GmlSetEdge(EdgIdx, i-1, EdgTab[i][0]-1, EdgTab[i][1]-1);

   GmlUploadData(EdgIdx);

   // Create a raw datatype to store the element middles
    // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewData(GmlRawData, NmbEdg, sizeof(cl_float4), GmlOutput)))
      return(1);

   if(!(MidTab = malloc((NmbEdg+1)*4*sizeof(float))))
      return(1);

   // Launch the kernel on the GPU
   GpuTim = GmlLaunchKernel(CalMid, NmbEdg, 3, EdgIdx, MidIdx, VerIdx);

   if(GpuTim < 0)
      return(1);

   // Get the results back and print some stats
   GmlDownloadData(MidIdx);

   for(i=1;i<=NmbEdg;i++)
   {
      GmlGetRawData(MidIdx, i-1, MidTab[i]);
      chk += sqrt(   MidTab[i][0] * MidTab[i][0] \
                  +  MidTab[i][1] * MidTab[i][1] \
                  +  MidTab[i][2] * MidTab[i][2]);
   }

   printf("%d edge centers computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbEdg, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbEdg );


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(EdgIdx);
   GmlFreeData(MidIdx);
   GmlStop();

   free(MidTab);
   free(EdgTab);
   free(VerTab);

   return(0);
}
