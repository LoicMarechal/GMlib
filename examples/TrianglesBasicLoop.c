

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 2.00                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on elements                                */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     dec 03 2012                                           */
/*   Last modification: feb 05 2017                                           */
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
#include "TrianglesBasicLoop.h"


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements middle and get back the results.                              */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int i, NmbVer, NmbTri, ver=0, dim=0, ref, (*TriTab)[3]=NULL;
   int VerIdx, TriIdx, MidIdx, CalMid, GpuIdx=0;
   int64_t InpMsh;
   float (*VerTab)[3]=NULL, (*MidTab)[4], dummy, chk=0.;
   double GpuTim;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nTrianglesBasicLoop GPU_index");
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
   if( !(InpMsh = GmfOpenMesh("triangles.meshb", GmfRead, &ver, &dim)) \
   || (ver != GmfFloat) || (dim != 3) )
   {
      puts("Could not open triangles.meshb");
      puts("Please run the command in the same directory this mesh is located");
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) \
   || !(VerTab = malloc((NmbVer+1) * 3 * sizeof(float))) )
   {
      return(1);
   }

   if( !(NmbTri = GmfStatKwd(InpMsh, GmfTriangles)) \
   || !(TriTab = malloc((NmbTri+1) * 3 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGotoKwd(InpMsh, GmfVertices);
   for(i=1;i<=NmbVer;i++)
      GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

   // Read the elements
   GmfGotoKwd(InpMsh, GmfTriangles);
   for(i=1;i<=NmbTri;i++)
      GmfGetLin(  InpMsh, GmfTriangles, &TriTab[i][0], \
                  &TriTab[i][1], &TriTab[i][2], &ref );

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

   if(!(CalMid = GmlNewKernel(TrianglesBasicLoop, "TrianglesBasic")))
      return(1);

   // Create a vertices data type and transfer the data to the GPU
   if(!(VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInput)))
      return(1);

   for(i=1;i<=NmbVer;i++)
      GmlSetVertex(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2]);

   GmlUploadData(VerIdx);

   /* Do the same with the elements */
   if(!(TriIdx = GmlNewData(GmlTriangles, NmbTri, 0, GmlInput)))
      return(1);

   for(i=1;i<=NmbTri;i++)
      GmlSetTriangle(TriIdx, i-1, TriTab[i][0]-1, \
                     TriTab[i][1]-1, TriTab[i][2]-1);

   GmlUploadData(TriIdx);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewData(GmlRawData, NmbTri, sizeof(cl_float4), GmlOutput)))
      return(1);

   if(!(MidTab = malloc((NmbTri+1)*4*sizeof(float))))
      return(1);

   // Launch the kernel on the GPU
   GpuTim = GmlLaunchKernel(CalMid, NmbTri, 3, TriIdx, MidIdx, VerIdx);

   if(GpuTim < 0)
      return(1);

   // Get the results back and print some stats
   GmlDownloadData(MidIdx);

   for(i=1;i<=NmbTri;i++)
   {
      GmlGetRawData(MidIdx, i-1, MidTab[i]);
      chk += sqrt(MidTab[i][0] * MidTab[i][0] \
               +  MidTab[i][1] * MidTab[i][1] \
               +  MidTab[i][2] * MidTab[i][2]);
   }

   printf("%d triangle middles computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbTri, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbTri );


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(TriIdx);
   GmlFreeData(MidIdx);
   GmlStop();

   free(MidTab);
   free(TriTab);
   free(VerTab);

   return(0);
}
