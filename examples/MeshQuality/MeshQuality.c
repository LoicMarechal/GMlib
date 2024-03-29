

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.30                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Compute a tet mesh mean and min qualities             */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     mar 24 2020                                           */
/*   Last modification: aug 05 2021                                           */
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
#include "compute_quality.h"


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
   int         NmbVer, NmbTet, res;
   int         QalIdx, QalKrn, VerIdx, TetIdx;
   int         GpuIdx = 0;
   size_t      GmlIdx;
   double      QalTim, AvgTim, MinTim, AvgQal, MinQal;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nMeshQuality GPU_index");
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

   GmlImportMesh( GmlIdx, "../sample_meshes/tetrahedra.meshb",
                  GmfVertices, GmfTetrahedra, 0 );

   if(!GmlGetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf("Imported %d vertices and %d tets from the mesh file\n", NmbVer, NmbTet);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(QalIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "qal")))
      return(1);

   // Assemble and compile a neighbours kernel
   QalKrn = GmlCompileKernel( GmlIdx, compute_quality, "compute_quality",
                              GmlTetrahedra, 2,
                              VerIdx, GmlReadMode, NULL,
                              QalIdx, GmlWriteMode, NULL );

   if(!QalKrn)
      return(1);

   // Launch the tetrahedra kernel on the GPU
   res = GmlLaunchKernel(GmlIdx, QalKrn);

   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %d\n", QalKrn, res);
      exit(0);
   }

   // Launch the reduction kernel on the GPU
   res = GmlReduceVector(GmlIdx, QalIdx, GmlSum, &AvgQal);

   if(res < 0)
   {
      printf("Launch reduction kernel failled with error: %d\n", res);
      exit(0);
   }

   // Launch the reduction kernel on the GPU
   res = GmlReduceVector(GmlIdx, QalIdx, GmlMin, &MinQal);

   if(res < 0)
   {
      printf("Launch reduction kernel failled with error: %d\n", res);
      exit(0);
   }


   /*-----------------*/
   /* GET THE RESULTS */
   /*-----------------*/

   QalTim = GmlGetKernelRunTime(GmlIdx, QalKrn);
   AvgTim = GmlGetReduceRunTime(GmlIdx, GmlSum);
   MinTim = GmlGetReduceRunTime(GmlIdx, GmlMin);

   printf("%d tets processed in %g seconds, quality=%g s, min=%g s, mean=%g s\n",
          NmbTet, QalTim + AvgTim + MinTim, QalTim, AvgTim, MinTim);

   printf("%zd MB used, %zd MB transfered\n",
          GmlGetMemoryUsage   (GmlIdx) / 1048576,
          GmlGetMemoryTransfer(GmlIdx) / 1048576);

   printf("QalTet checksum = min quality=%g, mean quality=%g\n",
         MinQal, AvgQal / NmbTet);


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(GmlIdx, VerIdx);
   GmlFreeData(GmlIdx, TetIdx);
   GmlFreeData(GmlIdx, QalIdx);
   GmlStop(GmlIdx);

   return(0);
}
