

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.23                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       tet mesh quality improvement with nodes smoothing     */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     mar 27 2020                                           */
/*   Last modification: may 06 2020                                           */
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

#include "Parameters.h"
#include "NS_quality.h"
#include "NS_scatter.h"
#include "NS_gather.h"


/*----------------------------------------------------------------------------*/
/* This structure definition must be exactly the same as the OpenCL one       */
/*----------------------------------------------------------------------------*/

typedef struct {
   int   foo;
   float res;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* Some very basic kernel launch error checking                               */
/*----------------------------------------------------------------------------*/

int CheckLaunch(int KrnIdx, int res)
{
   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %d\n", KrnIdx, res);
      exit(0);
   }

   return(res);
}


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, perform several smoothing  */
/* steps, compute the resulting mesh quality and save it                      */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         i, j, res, NmbVer, NmbTet;
   int         QalIdx, QalKrn, VerIdx, TetIdx, ParIdx, OptIdx, ResIdx;
   int         GpuIdx = 0, TetKrn, VerKrn;
   size_t      GmlIdx;
   double      RedTim, TetTim, VerTim, WalTim, AvgQal, OptRes;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nMeshSmoother GPU_index");
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

   GmlImportMesh(GmlIdx, "../sample_meshes/tetrahedra.meshb", GmfVertices, GmfTetrahedra, 0);

   if(!GetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf("Imported %d vertices and %d tets from the mesh file\n", NmbVer, NmbTet);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), Parameters)))
      return(1);

   // A vector to store the tets' quality
   if(!(QalIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "qal")))
      return(1);

   // Create a raw datatype to store the element middles.
   if(!(OptIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 4, GmlFlt4, "OptCrd")))
      return(1);

   // A residual vector to store the nodes' displacement
   if(!(ResIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1, GmlFlt, "res")))
      return(1);

   // Assemble and compile the tet quality kernel
   QalKrn = GmlCompileKernel( GmlIdx, NS_quality, "NS_quality",
                              GmlTetrahedra, 2,
                              VerIdx, GmlReadMode,  NULL,
                              QalIdx, GmlWriteMode, NULL );

   if(!QalKrn)
      return(1);

   // Assemble and compile the tet optimizing kernel
   TetKrn = GmlCompileKernel( GmlIdx, NS_scatter, "NS_scatter",
                              GmlTetrahedra, 2,
                              VerIdx, GmlReadMode,  NULL,
                              OptIdx, GmlWriteMode, NULL );

   if(!TetKrn)
      return(1);

   // Assemble and compile gather coordinates kernel
   VerKrn = GmlCompileKernel( GmlIdx, NS_gather, "NS_gather",
                              GmlVertices, 3,
                              ResIdx, GmlWriteMode,               NULL,
                              OptIdx, GmlReadMode | GmlVoyeurs,   NULL,
                              VerIdx, GmlReadMode | GmlWriteMode, NULL );

   if(!VerKrn)
      return(1);

   WalTim = GmlGetWallClock();

   // Launch the tetrahedra quality kernel on the GPU
   res = GmlLaunchKernel(GmlIdx, QalKrn);
   CheckLaunch(QalKrn, res);

   // Launch the reduction kernel on the GPU
   res = GmlReduceVector(GmlIdx, QalIdx, GmlSum, &AvgQal);
   CheckLaunch(0, res);
   printf("Before smoothing: mean quality=%g\n", AvgQal / NmbTet);

   for(int itr=1; itr<=10; itr++)
   {
      // Launch the tetrahedra optimizer kernel on the GPU
      res = GmlLaunchKernel(GmlIdx, TetKrn);
      CheckLaunch(TetKrn, res);

      // Launch the vertex gather kernel on the GPU
      res = GmlLaunchKernel(GmlIdx, VerKrn);
      CheckLaunch(VerKrn, res);

      // Compute the residual displacement value
      res = GmlReduceVector(GmlIdx, ResIdx, GmlSum, &OptRes);
      CheckLaunch(0, res);
      printf("Smoothing step %2d: residual=%g\n", itr, OptRes);
   }

   // Launch the tetrahedra quality kernel on the GPU
   res  = GmlLaunchKernel(GmlIdx, QalKrn);
   CheckLaunch(QalKrn, res);

   // Launch the reduction kernel on the GPU
   res = GmlReduceVector(GmlIdx, QalIdx, GmlSum, &AvgQal);
   CheckLaunch(0, res);
   printf("After  smoothing: mean quality=%g\n", AvgQal / NmbTet);

   WalTim = GmlGetWallClock() - WalTim;


   /*-----------------*/
   /* GET THE RESULTS */
   /*-----------------*/

   TetTim = GmlGetKernelRunTime(GmlIdx, TetKrn);
   VerTim = GmlGetKernelRunTime(GmlIdx, VerKrn);
   RedTim = GmlGetReduceRunTime(GmlIdx, GmlSum);

   printf(  "%d tets optimized in %g seconds (scatter=%g, gather=%g, reduce=%g), wall clock = %g\n",
            NmbTet, RedTim + TetTim + VerTim, TetTim, VerTim, RedTim, WalTim );

   printf("%ld MB used, %ld MB transfered\n",
          GmlGetMemoryUsage   (GmlIdx) / 1048576,
          GmlGetMemoryTransfer(GmlIdx) / 1048576);


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(GmlIdx, VerIdx);
   GmlFreeData(GmlIdx, TetIdx);
   GmlFreeData(GmlIdx, QalIdx);
   GmlFreeData(GmlIdx, OptIdx);
   GmlFreeData(GmlIdx, ResIdx);
   GmlFreeData(GmlIdx, ParIdx);
   GmlStop(GmlIdx);

   return(0);
}
