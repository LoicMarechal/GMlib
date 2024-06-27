

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
/*----------------------------------------------------------------------------*/

void ChkGmlErr(int ret, char *KrnNam)
{
   if(ret < 0)
   {
      printf("Launch kernel %s failled with error: %d\n", KrnNam, ret);
      exit(0);
   }
}


/*----------------------------------------------------------------------------*/
/* Read a tet mesh, send the data on the GPU, compute the volumes, assemble   */
/* the matrix and solve the system Ax = b with b = 0                          */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         i, j, k, l, ret, NmbVer, NmbTet, RhsIdx, DiaIdx, Xk0Idx, Xk1Idx;
   int         MatIdx, ResIdx, *DegTab, GpuIdx = 0, ref, VerIdx, TetIdx, tmp;
   int         NmbEdg, EdgIdx, NmbItr, *LinTab, *ColTab, BlkSiz, FltSiz, FltTyp;
   int         VecTyp, RedIdx;
   float       res, MemByt, FltOpp, *ValTabFlt;
   double      tim, TotRes = 0., *ValTabDbl;
   size_t      GmlIdx, nnz = 0;
   void        *ValTab, *sol;
   char        *InpNam;
   GmlParSct   *GmlPar;


   // --------------------
   // COMMAND LINE PARSING
   // --------------------

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
   }
   else
   {
      FltTyp = GmlDbl;
      VecTyp = (BlkSiz <= 4) ? GmlDbl4 : GmlDbl8;
   }

   // Init the GMLIB and compile the OpenCL source code
   if(!(GmlIdx = GmlInit(GpuIdx)))
      return(1);

   //GmlDebugOn(GmlIdx);


   // -----------------------
   // MESH READING AND PARSING
   // ------------------------

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


   // -------------------
   // SPARSE MATRIX SETUP
   // -------------------

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

   // Allocate and transfer the L+U sparse, sliced, block matrix
   MatIdx = GmlNewMatrix(GmlIdx, NmbVer, nnz, BlkSiz, ValTab, ColTab, LinTab, FltTyp);
   assert(MatIdx);

   free(DegTab);
   free(LinTab);
   free(ColTab);


   // ---------------------
   // DIAGONAL MATRIX SETUP
   // ---------------------

   for(i=0;i<NmbVer;i++)
      for(j=0;j<POW(BlkSiz);j++)
         if(FltTyp == GmlFlt)
            ValTabFlt[ i * POW(BlkSiz) + j ] = i + j;
         else
            ValTabDbl[ i * POW(BlkSiz) + j ] = i + j;

   // Allocate and setup the diagonal matrix as a vector
   DiaIdx = GmlNewVector(GmlIdx, NmbVer, POW(BlkSiz), ValTab, FltTyp);
   assert(DiaIdx);


   // ------------------------------------
   // ALLOCATE AND SETUP THE THREE VECTORS
   // ------------------------------------

   for(i=0;i<NmbVer;i++)
      for(j=0;j<BlkSiz;j++)
         if(FltTyp == GmlFlt)
            ValTabFlt[ i * BlkSiz + j ] = 1.;
         else
            ValTabDbl[ i * BlkSiz + j ] = 1.;

   // Allocate and setup the even iteration solution vector
   Xk0Idx = GmlNewVector(GmlIdx, NmbVer, BlkSiz, ValTab, FltTyp);
   assert(Xk0Idx);

   for(i=0;i<NmbVer;i++)
      for(j=0;j<BlkSiz;j++)
         if(FltTyp == GmlFlt)
            ValTabFlt[ i * BlkSiz + j ] = 0.;
         else
            ValTabDbl[ i * BlkSiz + j ] = 0.;

   // Allocate and setup the odd iteration solution vector
   Xk1Idx = GmlNewVector(GmlIdx, NmbVer, BlkSiz, ValTab, FltTyp);
   assert(Xk1Idx);

   for(i=0;i<NmbVer;i++)
      for(j=0;j<BlkSiz;j++)
         if(FltTyp == GmlFlt)
            ValTabFlt[ i * BlkSiz + j ] = .1;
         else
            ValTabDbl[ i * BlkSiz + j ] = .1;

   // Allocate and setup the right handside vector
   RhsIdx = GmlNewVector(GmlIdx, NmbVer, BlkSiz, ValTab, FltTyp);
   assert(RhsIdx);

   // Allocate a reduction vector stored in a vertex based solution field
   RedIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1, GmlFlt, "reduction");
   assert(RedIdx);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   if(FltTyp == GmlFlt)
      free(ValTabFlt);
   else
      free(ValTabDbl);

   // Start the resolution loop
   tim = GmlGetWallClock();

   for(i=1;i<=NmbItr;i++)
   {
      // Launch the A.X kernel on the GPU
      ret = GmlMultMatVec(GmlIdx, MatIdx, Xk0Idx, Xk1Idx);
      ChkGmlErr(ret, "GmlMultMatVec");

      // Launch the X+B kernel on the GPU
      ret = GmlAddVec(GmlIdx, RhsIdx, Xk1Idx);
      ChkGmlErr(ret, "GmlAddVec");

      // Launch the D.X kernel on the GPU
      ret = GmlMultDiagMatVec(GmlIdx, DiaIdx, Xk1Idx);
      ChkGmlErr(ret, "GmlMultDiagMatVec");

      // Compute and print the residual value
      ret = GmlNormVec(GmlIdx, Xk1Idx, RedIdx, &res);
      ChkGmlErr(ret, "GmlNormVec");

      TotRes += res;
      printf("\r iter = %4d, checksum = %g ", i, TotRes);
      fflush(stdout);

      // Swap the odd and even iteration vectors
      tmp = Xk0Idx;
      Xk0Idx = Xk1Idx;
      Xk1Idx = tmp;
   }

   // Get the total physical runtime
   tim = GmlGetWallClock() - tim;

   printf(" GPU memory used: %.2f GBytes\n", (float)GmlGetMemoryUsage(GmlIdx) / 1073741824.);
   printf(" wall clock = %g\n", tim);
   printf(" %8.2f GBytes/s,",  GmlGetMemoryAccess(GmlIdx) / (tim * 1E9));
   printf(" %8.2f GFlops/s\n", GmlGetFlops(GmlIdx) / (tim * 1E9));

/*   for(i=0;i<10;i++)
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
*/
   GmlStop(GmlIdx);

   return(0);
}
