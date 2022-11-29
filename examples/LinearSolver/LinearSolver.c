

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.34                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Linear solver                                         */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     mar 22 2022                                           */
/*   Last modification: mar 31 2022                                           */
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
#include "mulmatvec.h"
#include "residual.h"
#include "updrhs.h"


/*----------------------------------------------------------------------------*/
/* This structure definition must be exactly the same as the OpenCL one       */
/*----------------------------------------------------------------------------*/

typedef struct {
   int   foo;
   float res;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* Read a tet mesh, send the data on the GPU, compute the volumes, assemble   */
/* the matrix and solve the system Ax = b with b = 0                          */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         i, j, k, ret, NmbVer, VerIdx, NmbTet, TetIdx, DegIdx, BalIdx;
   int         SolIdx, RhsIdx, MatIdx, ResIdx, MatVecKrn, UpdRhsKrn, CalResKrn;
   int         TetDat[4], *DegTab, (*BalTab)[16], idx0, idx1, GpuIdx = 0;
   int         TetEdg[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
   int         BalFlg, ref;
   float       sol[256] = {1};
   double      tim, res, TotRes = 0.;
   size_t      GmlIdx;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nLinearSolver GPU_index");
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


   // Inport the volume mesh
   GmlImportMesh( GmlIdx, "../sample_meshes/tetrahedra.meshb",
                  GmfVertices, GmfTetrahedra, 0 );

   if(!GmlGetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf("Imported  %8d vertices and %d tetrahedra\n", NmbVer, NmbTet);


   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   if(!(DegIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1,  GmlInt,   "Deg")))
      return(1);

   if(!(BalIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1,  GmlInt16, "Bal")))
      return(1);

   if(!(MatIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 16, GmlFlt16, "Mat")))
      return(1);

   if(!(SolIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1,  GmlFlt4,  "Sol")))
      return(1);

   if(!(RhsIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1,  GmlFlt4,  "Rhs")))
      return(1);

   if(!(ResIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1,  GmlFlt,   "Res")))
      return(1);


   // Init the matrix and vectors with 1
   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, SolIdx, i, &sol);

   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, RhsIdx, i, &sol);

   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, MatIdx, i, &sol);

   DegTab = calloc(NmbVer, sizeof(int));
   BalTab = malloc(NmbVer * 16 * sizeof(int));

   for(i=0;i<NmbTet;i++)
   {
      GmlGetDataLine(GmlIdx, TetIdx, i, &TetDat[0], &TetDat[1],
                     &TetDat[2], &TetDat[3], &ref);

      for(j=0;j<6;j++)
      {
         idx0 = TetDat[ TetEdg[j][0] ];
         idx1 = TetDat[ TetEdg[j][1] ];

         BalFlg = 0;

         for(k=0;k<DegTab[ idx0 ];k++)
            if(BalTab[ idx0 ][k] == idx1)
            {
               BalFlg = 1;
               break;
            }

         if(!BalFlg && DegTab[ idx0 ] < 16)
            BalTab[ idx0 ][ DegTab[ idx0 ]++ ] = idx1;

         BalFlg = 0;

         for(k=0;k<DegTab[ idx1 ];k++)
            if(BalTab[ idx1 ][k] == idx0)
            {
               BalFlg = 1;
               break;
            }

         if(!BalFlg && DegTab[ idx1 ] < 16)
            BalTab[ idx1 ][ DegTab[ idx1 ]++ ] = idx0;
      }
   }

   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, DegIdx, i, &DegTab[i]);

   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, BalIdx, i, BalTab[i]);

   // Assemble and compile the kernels
   UpdRhsKrn = GmlCompileKernel( GmlIdx, updrhs, "updrhs", GmlVertices, 1,
                                 RhsIdx, GmlManual, NULL );

   if(!UpdRhsKrn)
      return(1);

   // Assemble and compile a neighbours kernel
   MatVecKrn = GmlCompileKernel( GmlIdx, mulmatvec, "matvec", GmlVertices, 5,
                                 DegIdx, GmlManual, NULL,
                                 BalIdx, GmlManual, NULL,
                                 MatIdx, GmlManual, NULL,
                                 RhsIdx, GmlManual, NULL,
                                 SolIdx, GmlManual, NULL );

   if(!MatVecKrn)
      return(1);

   // Assemble and compile a  kernel
   CalResKrn = GmlCompileKernel( GmlIdx, residual, "residual", GmlVertices, 2,
                                 RhsIdx, GmlManual, NULL,
                                 ResIdx, GmlManual, NULL );

   if(!CalResKrn)
      return(1);


   // Start the resolution loop
   tim = GmlGetWallClock();

   for(i=1;i<=1000;i++)
   {
      // Launch the vertex kernel on the GPU
      ret = GmlLaunchKernel(GmlIdx, UpdRhsKrn);

      if(ret < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", UpdRhsKrn, ret);
         exit(0);
      }

      // Launch the vertex kernel on the GPU
      ret = GmlLaunchKernel(GmlIdx, MatVecKrn);

      if(ret < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", MatVecKrn, ret);
         exit(0);
      }

      // Launch the vertex kernel on the GPU
      ret = GmlLaunchKernel(GmlIdx, CalResKrn);

      if(ret < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", CalResKrn, ret);
         exit(0);
      }

      GmlReduceVector(GmlIdx, ResIdx, GmlL1, &res);
      TotRes += res;
      printf("\r checksum = %g ", TotRes);
      fflush(stdout);
   }

   printf("\n wall clock = %g\n", GmlGetWallClock() - tim);

   /*for(i=0;i<10;i++)
   {
      GmlGetDataLine(GmlIdx, VerSolIdx, i, &sol);
      printf("solution of vertex %d: %g\n", i, sol);
   }*/

   GmlStop(GmlIdx);

   return(0);
}
