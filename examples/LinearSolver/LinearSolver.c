

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.34                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Linear solver                                         */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     mar 22 2022                                           */
/*   Last modification: may 02 2024                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <libmeshb7.h>
#include <gmlib3.h>

#include "parameters.h"
#include "residual.h"
#include "updrhs.h"


/*----------------------------------------------------------------------------*/
/* Macro instructions                                                         */
/*----------------------------------------------------------------------------*/

#define MIN(a,b)     ((a) < (b) ? (a) : (b))
#define MAX(a,b)     ((a) > (b) ? (a) : (b))
#define POW(a)       ((a)*(a))
#define CUB(a)       ((a)*(a)*(a))


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
   int         i, j, k, l, ret, NmbVer, VerIdx, NmbTet, TetIdx, DegIdx, BalIdx;
   int         SolIdx, RhsIdx, MatIdx, ResIdx, MatVecKrn, UpdRhsKrn, CalResKrn;
   int         TetDat[4], *DegTab, (*BalTab)[16], idx0, idx1, GpuIdx = 0;
   int         TetEdg[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
   int         BalFlg, ref, NmbEdg, EdgIdx, deg[32] = {0}, NmbDeg, DegMax = 0;
   int         NmbItr, LowDeg, LowMax=0, UprDeg, UprMax=0, LowUprDeg[32][32]={0};
   int         (*TmpDeg)[2], *LinTab, *ColTab;
   float       sol[256] = {1};
   double      tim, byt=0, opp=0, res, TotRes = 0., *ValTab;
   size_t      GmlIdx, nnz = 0;
   char        *InpNam;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt != 4)
   {
      puts("\nLinearSolver tetmesh_name GPU_index NB_loops");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
   {
      InpNam = ArgVec[1];
      GpuIdx = atoi(ArgVec[2]);
      NmbItr = atoi(ArgVec[3]);

   }


   // Init the GMLIB and compile the OpenCL source code
   if(!(GmlIdx = GmlInit(GpuIdx)))
      return(1);

   //GmlDebugOn(GmlIdx);


   // Inport the volume mesh
   GmlImportMesh(GmlIdx, InpNam, GmfVertices, GmfTetrahedra, 0);

   if(!GmlGetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf(" Imported %d vertices and %d tetrahedra\n", NmbVer, NmbTet);

   // Extract all edges
   GmlExtractEdges(GmlIdx);
   GmlGetMeshInfo(GmlIdx, GmlEdges, &NmbEdg, &EdgIdx);
   printf(" %d edges extracted from the volume\n", NmbEdg);
   printf(" Mesh numbering factor = %g%%\n", GmlEvaluateNumbering(GmlIdx));

   DegTab = calloc(NmbVer+1, sizeof(int));
   assert(DegTab);

   LinTab = calloc(NmbVer+1, sizeof(int));
   assert(LinTab);

   // Build the degree tab
   for(i=0;i<NmbEdg;i++)
   {
      GmlGetDataLine(GmlIdx, EdgIdx, i, &j, &k, &ref);
      DegTab[j]++;
      DegTab[k]++;
   }

   for(i=0;i<NmbVer;i++)
   {
      if(DegTab[i] > 255)
      {
         puts("A vertex degree exceeds 256.");
         exit(1);
      }

      LinTab[i] = nnz;
      nnz += DegTab[i];
      DegTab[i] = 0;
   }

   LinTab[ NmbVer ] = nnz;

   ColTab = calloc(nnz, sizeof(int));
   assert(ColTab);

   ValTab = calloc(nnz, 16 * sizeof(double));
   assert(ValTab);

   for(i=0;i<NmbEdg;i++)
   {
      GmlGetDataLine(GmlIdx, EdgIdx, i, &j, &k, &ref);
      ColTab[ LinTab[j] + DegTab[j] ] = k;
      ColTab[ LinTab[k] + DegTab[k] ] = j;

      for(l=0;l<64;l++)
      {
         ValTab[ (LinTab[j] + DegTab[j]) * 16 + l ] = 1.;
         ValTab[ (LinTab[k] + DegTab[k]) * 16 + l ] = 1.;
      }

      DegTab[j]++;
      DegTab[k]++;
   }

   MatIdx = GmlNewMatrix(GmlIdx, NmbVer, nnz, 4, GmlFlt, ValTab, ColTab, LinTab, "matrix");

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
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


   // Assemble and compile the kernels
   UpdRhsKrn = GmlCompileKernel( GmlIdx, updrhs, "updrhs", GmlVertices, 1,
                                 RhsIdx, GmlManual, NULL );

   if(!UpdRhsKrn)
      return(1);

   // Assemble and compile a  kernel
   CalResKrn = GmlCompileKernel( GmlIdx, residual, "residual", GmlVertices, 2,
                                 RhsIdx, GmlManual, NULL,
                                 ResIdx, GmlManual, NULL );

   if(!CalResKrn)
      return(1);


   // Start the resolution loop
   tim = GmlGetWallClock();

   for(i=1;i<=NmbItr;i++)
   {
      // Launch the vertex kernel on the GPU
      ret = GmlLaunchKernel(GmlIdx, UpdRhsKrn);

      if(ret < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", UpdRhsKrn, ret);
         exit(0);
      }

      // Launch the vertex kernel on the GPU
      ret = GmlMultMatVec(GmlIdx, MatIdx, SolIdx, RhsIdx);

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
      printf("\r iter = %4d, checksum = %g ", i, TotRes);
      fflush(stdout);
   }

   // Get the total physical runtime
   tim = GmlGetWallClock() - tim;

   puts("");
   printf(" GPU memory used: %.2f GBytes\n", (float)GmlGetMemoryUsage(GmlIdx) / 1073741824.);
   printf(" wall clock = %g\n", tim);
   printf(" %8.2f GBytes/s,",  (float)GmlGetMemoryAccess(GmlIdx) / (tim * 1E9));
   printf(" %8.2f GFlops/s\n", (float)GmlGetFlops(GmlIdx) / (tim * 1E9));

   /*for(i=0;i<10;i++)
   {
      GmlGetDataLine(GmlIdx, VerSolIdx, i, &sol);
      printf("solution of vertex %d: %g\n", i, sol);
   }*/

   GmlStop(GmlIdx);

   return(0);
}
