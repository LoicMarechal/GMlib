

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.34                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Linear solver                                         */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     mar 22 2022                                           */
/*   Last modification: may 27 2024                                           */
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
   int         i, j, k, l, ret, NmbVer, VerIdx, NmbTet, TetIdx, SolIdx, RhsIdx;
   int         MatIdx, ResIdx, UpdRhsKrn, CalResKrn, *DegTab, GpuIdx = 0, ref;
   int         NmbEdg, EdgIdx, NmbItr, *LinTab, *ColTab, BlkSiz, FltSiz, FltTyp;
   int         VecTyp;
   float       MemByt, FltOpp, res;
   double      tim, TotRes = 0.;
   float       SolFlt[8] = {-1,-2,-3,-4,-5,-6,-7,-8}, *ValTabFlt;
   double      SolDbl[8] = {-1,-2,-3,-4,-5,-6,-7,-8}, *ValTabDbl;
   size_t      GmlIdx, nnz = 0;
   void        *ValTab, *sol;
   char        *InpNam;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt != 6)
   {
      puts("\nLinearSolver tetmesh_name GPU_index NB_loops Block_size (4,5 or 7) Float_size (32 or 64)");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
   {
      InpNam = ArgVec[1];
      GpuIdx = atoi(ArgVec[2]);
      NmbItr = atoi(ArgVec[3]);
      BlkSiz = atoi(ArgVec[4]);
      FltSiz = atoi(ArgVec[5]);
   }

   if(BlkSiz != 4 && BlkSiz != 5 && BlkSiz != 7)
   {
      printf("Invalid block size %d\n", BlkSiz);
      exit(1);
   }

   if(FltSiz != 32 && FltSiz != 64)
   {
      printf("Invalid float size %d\n", FltSiz);
      exit(1);
   }

   if(FltSiz == 32)
   {
      FltTyp = GmlFlt;
      VecTyp = (BlkSiz <= 4) ? GmlFlt4 : GmlFlt8;
      sol = (void *)SolFlt;
   }
   else
   {
      FltTyp = GmlDbl;
      VecTyp = (BlkSiz <= 4) ? GmlDbl4 : GmlDbl8;
      sol = (void *)SolDbl;
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

   if(FltTyp == GmlFlt)
   {
      ValTabFlt = calloc(nnz, POW(BlkSiz) * sizeof(float));
      assert(ValTabFlt);
      ValTab = (void *)ValTabFlt;
   }
   else
   {
      ValTabDbl = calloc(nnz, POW(BlkSiz) * sizeof(double));
      assert(ValTabDbl);
      ValTab = (void *)ValTabDbl;
   }

   for(i=0;i<NmbEdg;i++)
   {
      GmlGetDataLine(GmlIdx, EdgIdx, i, &j, &k, &ref);
      ColTab[ LinTab[j] + DegTab[j] ] = k;
      ColTab[ LinTab[k] + DegTab[k] ] = j;

      for(l=0;l<POW(BlkSiz);l++)
      {
         if(FltTyp == GmlFlt)
         {
            ValTabFlt[ (LinTab[j] + DegTab[j]) * POW(BlkSiz) + l ] += 1.;
            ValTabFlt[ (LinTab[k] + DegTab[k]) * POW(BlkSiz) + l ] += 1.;
         }
         else
         {
            ValTabDbl[ (LinTab[j] + DegTab[j]) * POW(BlkSiz) + l ] += 1.;
            ValTabDbl[ (LinTab[k] + DegTab[k]) * POW(BlkSiz) + l ] += 1.;
         }
      }

      DegTab[j]++;
      DegTab[k]++;
   }

   MatIdx = GmlNewMatrix(GmlIdx, NmbVer, nnz, BlkSiz, ValTab, ColTab, LinTab, FltTyp);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   if(!(SolIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1, VecTyp,  "Sol")))
      return(1);

   if(!(RhsIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1, VecTyp,  "Rhs")))
      return(1);

   if(!(ResIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1,  GmlFlt, "Res")))
      return(1);

   // Init the matrix and vectors with 1
   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, SolIdx, i, sol);

   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, RhsIdx, i, sol);

   if(BlkSiz == 4)
   {
      if(FltTyp == GmlFlt)
         GmlSetCompilerOptions(GmlIdx, " -DVECTYP=float4 -DVECINC=0.1,0.2,0.3,0.4 ");
      else
         GmlSetCompilerOptions(GmlIdx, " -DVECTYP=double4 -DVECINC=0.1,0.2,0.3,0.4 ");
   }
   else if(BlkSiz == 5)
   {
      if(FltTyp == GmlFlt)
         GmlSetCompilerOptions(GmlIdx, " -DVECTYP=float8 -DVECINC=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 ");
      else
         GmlSetCompilerOptions(GmlIdx, " -DVECTYP=double8 -DVECINC=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 ");
   }

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
         printf("Launch kernel GmlMultMatVec failled with error: %d\n", ret);
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

   MemByt = GmlGetMemoryAccess(GmlIdx) + 56. * NmbVer;
   FltOpp = GmlGetFlops(GmlIdx) + 8. * NmbVer;

   printf(" GPU memory used: %.2f GBytes\n", (float)GmlGetMemoryUsage(GmlIdx) / 1073741824.);
   printf(" wall clock = %g\n", tim);
   printf(" %8.2f GBytes/s,",  MemByt / (tim * 1E9));
   printf(" %8.2f GFlops/s\n", FltOpp / (tim * 1E9));

   for(i=0;i<10;i++)
   {
      GmlGetDataLine(GmlIdx, SolIdx, i, sol);
      printf("solution of vertex %d:", i);
      for(j=0;j<BlkSiz;j++)
      {
         if(FltTyp == GmlFlt)
            printf(" %g", SolFlt[j]);
         else
            printf(" %g", SolDbl[j]);
      }
      puts("");
   }

   GmlStop(GmlIdx);

   return(0);
}
