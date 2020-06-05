

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.29                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Get the min triangle quality for each vertex ball     */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     may 26 2020                                           */
/*   Last modification: jun 05 2020                                           */
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

#include "parameters.h"
#include "triqal.h"
#include "verqal.h"


/*----------------------------------------------------------------------------*/
/* This structure definition must be exactly the same as the OpenCL one       */
/*----------------------------------------------------------------------------*/

typedef struct {
   int   foo;
   float res;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements quality and get minimal and mean values                       */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         i, res, NmbVer, VerIdx, NmbTri, TriIdx, GpuIdx = 0, ParIdx;
   int         TriQalKrn, VerQalKrn, TriQalIdx, VerQalIdx;
   float       qal;
   size_t      GmlIdx;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nBallQuality GPU_index");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
      GpuIdx = atoi(ArgVec[1]);

   // Init the GMLIB and compile the OpenCL source code
   if(!(GmlIdx = GmlInit(GpuIdx)))
      return(1);

   //GmlDebugOn(GmlIdx);

   GmlImportMesh( GmlIdx, "../sample_meshes/naca012_2D_triangles.meshb",
                  GmfVertices, GmfTriangles, 0 );

   if(!GmlGetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTriangles, &NmbTri, &TriIdx))
      return(1);

   printf("Imported %d vertices and %d triangles\n", NmbVer, NmbTri);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   if(!(TriQalIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 1, GmlFlt, "tq")))
      return(1);

   if(!(VerQalIdx = GmlNewSolutionData(GmlIdx, GmlVertices,  1, GmlFlt, "vq")))
      return(1);

   // Assemble and compile a neighbours kernel
   TriQalKrn = GmlCompileKernel( GmlIdx, triqal, "triqal", GmlTriangles, 2,
                                 VerIdx, GmlReadMode, NULL,
                                 TriQalIdx, GmlWriteMode, NULL );

   if(!TriQalKrn)
      return(1);

   // Assemble and compile a neighbours kernel
   VerQalKrn = GmlCompileKernel( GmlIdx, verqal, "verqal", GmlVertices, 3,
                                 TriIdx,    GmlReadMode, NULL,
                                 TriQalIdx, GmlReadMode, NULL,
                                 VerQalIdx, GmlWriteMode, NULL );

   if(!VerQalKrn)
      return(1);

   // Launch the tetrahedra kernel on the GPU
   res = GmlLaunchKernel(GmlIdx, TriQalKrn);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %d\n", TriQalKrn, res);
      exit(0);
   }

   // Launch the tetrahedra kernel on the GPU
   res = GmlLaunchKernel(GmlIdx, VerQalKrn);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %d\n", VerQalKrn, res);
      exit(0);
   }

   for(i=0;i<NmbTri;i++)
   {
      GmlGetDataLine(GmlIdx, TriQalIdx, i, &qal);
      printf("triangle %d quality = %g\n", i, qal);
   }

   for(i=0;i<NmbVer;i++)
   {
      GmlGetDataLine(GmlIdx, VerQalIdx, i, &qal);
      printf("ball of vertex %d: min quality = %g\n", i, qal);
   }

   GmlStop(GmlIdx);

   return(0);
}
