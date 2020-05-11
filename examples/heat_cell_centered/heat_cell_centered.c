/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Cell-centered heat equation solver                    */
/*   Author:            Julien VANHAREN                                       */
/*   Creation date:     apr 15 2020                                           */
/*   Last modification: apr 27 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmeshb7.h>
#include <gmlib3.h>

#include "param.h"
#include "ini_tet.h"
#include "sol_ext.h"
#include "grd_tet.h"
#include "grd_ext.h"
#include "flx_bal.h"
#include "tim_int.h"

enum BC_TYPE
{
   DIRICHLET,
   NEUMANN
};
typedef struct
{
   int BoCo[6];
   int BoCo_Ref[6];
   float BoCo_Val[6];
} GmlParSct;

void Write_SolTet(char *out_fn, int NbrTet, float *SolTet)
{
   int64_t fid;
   int Ver = 1, Dim = 3;
   int NmbTyp = 1, TypTab[1] = {GmfSca};

   fid = GmfOpenMesh(out_fn, GmfWrite, Ver, Dim);
   GmfSetKwd(fid, GmfSolAtTetrahedra, NbrTet, NmbTyp, TypTab);
   GmfSetBlock(fid, GmfSolAtTetrahedra, 1, NbrTet, 0, NULL, NULL,
               GmfFloat, &SolTet[0], &SolTet[NbrTet - 1]);
   GmfCloseMesh(fid);
}

void Heat_Init(int argc, char *argv[], size_t *GmlIdx)
{
   int dbg = 0, GpuIdx;

   if (argc == 1)
   {
      puts("Cell-centered heat equation solver");
      puts("Choose the GPU index from the following list:");
      GmlListGPU();
      exit(EXIT_SUCCESS);
   }
   else
      GpuIdx = atoi(argv[1]);
   if (argc == 3)
      dbg = 1;
   if (!(*GmlIdx = GmlInit(GpuIdx)))
      exit(EXIT_SUCCESS);
   if (dbg)
      GmlDebugOn(*GmlIdx);
}

int main(int argc, char *argv[])
{
   int i, j, NmbItr = 0, NbrVer = 0, NbrTri = 0, NbrTet = 0;
   int ParIdx, VerIdx = 0, TriIdx = 0, TetIdx = 0, BalIdx, MidIdx, SolIdx, FlxIdx, CalMid, GatherIdx, OptVer;
   int GpuIdx = 0, ResIdx, NgbIdx, NgbKrn, FlxKrn, F64Idx, F64Krn;
   size_t GmlIdx;
   float MidTab[4], SolTab[8], TetChk = 0., VerChk = 0.;
   float Zero[4] = {0.f, 0.f, 0.f, 0.f};
   double NgbTim = 0, TetTim = 0, VerTim = 0, RedTim = 0, F64Tim = 0;
   double TotalTime = 0., res, FlxTim = 0., GtrTim = 0., ResTim = 0., WallTime;
   float *Sol, *XGrdTet, *YGrdTet, *ZGrdTet, *Laplacian;
   GmlParSct *GmlPar;

   int n;
   int IniTetKrn, SolTetIdx, SolTetTmpIdx, GrdTetIdx, SolExtIdx, GrdExtIdx, RhsIdx;
   int SolExtKrn, GrdTetKrn, GrdExtKrn, FlxBalKrn, TimKrn;
   double Time[6]={0.}, InitRes, Res;
   float *SolTet, Tmp[4];

   /* Library initialization. */
   Heat_Init(argc, argv, &GmlIdx);

   /* Define boundary conditions. */
   GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), param);
   GmlPar->BoCo[0] = NEUMANN;
   GmlPar->BoCo[1] = NEUMANN;
   GmlPar->BoCo[2] = NEUMANN;
   GmlPar->BoCo[3] = DIRICHLET;
   GmlPar->BoCo[4] = DIRICHLET;
   GmlPar->BoCo[5] = NEUMANN;
   for (i = 0; i < 6; i++)
      GmlPar->BoCo_Ref[i] = i + 1;
   for (i = 0; i < 6; i++)
      GmlPar->BoCo_Val[i] = 0.;
   GmlPar->BoCo_Val[3] = +300.f;
   GmlPar->BoCo_Val[4] = +250.f;

   /* Import mesh and print statistics. */
   GmlImportMesh(GmlIdx, "../sample_meshes/cube.meshb", GmfVertices, GmfTriangles, GmfTetrahedra);
   GetMeshInfo(GmlIdx, GmlVertices, &NbrVer, &VerIdx);
   GetMeshInfo(GmlIdx, GmlTetrahedra, &NbrTet, &TetIdx);
   GetMeshInfo(GmlIdx, GmlTriangles, &NbrTri, &TriIdx);
   printf("+++ Imported %d vertices, %d triangles and %d tetrahedra from the mesh file\n", NbrVer, NbrTri, NbrTet);

   /* Extract all faces connectivity. */
   GmlExtractFaces(GmlIdx);
   GetMeshInfo(GmlIdx, GmlTriangles, &NbrTri, &TriIdx);
   printf("+++ %d triangles extracted from the volume\n", NbrTri);

   /* Fields declaration. */
   SolTetIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "SolTet");
   SolTetTmpIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "SolTetTmp");
   GrdTetIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt4, "GrdTet");
   SolExtIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 1, GmlFlt, "SolExt");
   GrdExtIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 1, GmlFlt4, "GrdExt");
   RhsIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "Rhs");
   
   for (i = 0; i < NbrTet; i++)
   {
      GmlSetDataLine(GmlIdx, SolTetIdx, i, Zero);
      GmlSetDataLine(GmlIdx, SolTetTmpIdx, i, Zero);
      GmlSetDataLine(GmlIdx, GrdTetIdx, i, Zero);
      GmlSetDataLine(GmlIdx, RhsIdx, i, Zero);
   }
   for (i = 0; i < NbrTri; i++)
   {
      GmlSetDataLine(GmlIdx, SolExtIdx, i, Zero);
      GmlSetDataLine(GmlIdx, GrdExtIdx, i, Zero);
   }

   /* Kernels compilation. */
   IniTetKrn = GmlCompileKernel(GmlIdx, ini_tet, "ini_tet", GmlTetrahedra, 2,
                                VerIdx, GmlReadMode, NULL,
                                SolTetIdx, GmlWriteMode, NULL);
   SolExtKrn = GmlCompileKernel(GmlIdx, sol_ext, "sol_ext", GmlTriangles, 3,
                                TriIdx, GmlReadMode | GmlRefFlag, NULL,
                                SolTetIdx, GmlReadMode, NULL,
                                SolExtIdx, GmlWriteMode, NULL);
   GrdTetKrn = GmlCompileKernel(GmlIdx, grd_tet, "grd_tet", GmlTetrahedra, 3,
                                VerIdx, GmlReadMode, NULL,
                                SolExtIdx, GmlReadMode, NULL,
                                GrdTetIdx, GmlWriteMode, NULL);
   GrdExtKrn = GmlCompileKernel(GmlIdx, grd_ext, "grd_ext", GmlTriangles, 3,
                                TriIdx, GmlReadMode | GmlRefFlag, NULL,
                                GrdTetIdx, GmlReadMode, NULL,
                                GrdExtIdx, GmlWriteMode, NULL);
   FlxBalKrn = GmlCompileKernel(GmlIdx, flx_bal, "flx_bal", GmlTetrahedra, 3,
                                VerIdx, GmlReadMode, NULL,
                                GrdExtIdx, GmlReadMode, NULL,
                                RhsIdx, GmlWriteMode, NULL);
   TimKrn = GmlCompileKernel(GmlIdx, tim_int, "tim_int", GmlTetrahedra, 2,
                             RhsIdx, GmlReadMode, NULL,
                             SolTetIdx, GmlReadMode | GmlWriteMode, NULL);

   /* Solution initialization. */
   // Time = GmlLaunchKernel(GmlIdx, IniTetKrn);
   /* Begin resolution. */

   WallTime = GmlGetWallClock();
   GmlUploadParameters(GmlIdx);

   for (n = 1; n <= 1; n++)
   {
      GmlLaunchKernel(GmlIdx, SolExtKrn);
      GmlLaunchKernel(GmlIdx, GrdTetKrn);
      GmlLaunchKernel(GmlIdx, GrdExtKrn);
      GmlLaunchKernel(GmlIdx, FlxBalKrn);
      GmlLaunchKernel(GmlIdx, TimKrn);
      GmlReduceVector(GmlIdx, RhsIdx, GmlSum, &Res);
      printf("\r+++ Iteration %4d Residual = %.12f", n, Res);
      fflush(stdout);
   }

   GmlDownloadParameters(GmlIdx);

   Time[0] = GmlGetKernelRunTime(GmlIdx, SolExtKrn);
   Time[1] = GmlGetKernelRunTime(GmlIdx, GrdTetKrn);
   Time[2] = GmlGetKernelRunTime(GmlIdx, GrdExtKrn);
   Time[3] = GmlGetKernelRunTime(GmlIdx, FlxBalKrn);
   Time[4] = GmlGetKernelRunTime(GmlIdx, TimKrn);
   Time[5] = GmlGetReduceRunTime(GmlIdx, GmlSum);

   puts("");
   for(i=0;i<6;i++)
   {
      TotalTime += Time[i];
      printf("Total time for kernel %d = %g seconds\n", i, Time[i]);
   }

   printf("GPU execution time = %g seconds\n", TotalTime);
   printf("Wall clock         = %g seconds\n", GmlGetWallClock() - WallTime);

   // do
   // {
   //    // LOOP OVER THE TRIANGLES
   //    res = GmlLaunchKernel(GmlIdx, FlxKrn);

   //    if(res < 0)
   //    {
   //       printf("Error %d in flux kernel\n", (int)res);
   //       exit(1);
   //    }

   //    FlxTim += res;

   //    // LOOP OVER THE TRETRAHEDRA
   //    res = GmlLaunchKernel(GmlIdx, GatherIdx);

   //    if(res < 0)
   //    {
   //       printf("Error %d in gather kernel\n", (int)res);
   //       exit(2);
   //    }

   //    GtrTim += res;

   //    // COMPUTE THE RESIDUAL
   //    res = GmlReduceVector(GmlIdx, SolIdx, GmlSum, &residual_i);

   //    if(res < 0)
   //    {
   //       printf("Error %d in reduction kernel\n", (int)res);
   //       exit(3);
   //    }

   //    ResTim += res;

   //    printf("\rIteration %4d: residual=%g", NmbItr, residual_i / InitRes);
   //    fflush(stdout);
   // }while(NmbItr++ < 1000 && (residual_i / InitRes > .0001));

   // printf("\nTotal run time = %gs (flux=%g, gather=%g, residual=%g)\n",
   //          FlxTim + GtrTim + ResTim, FlxTim, GtrTim, ResTim);

   // printf("Boundary triangles %d\n", GmlPar->Cnt);

   SolTet = malloc(NbrTet * sizeof(float));
   XGrdTet = malloc(NbrTet * sizeof(float));
   YGrdTet = malloc(NbrTet * sizeof(float));
   ZGrdTet = malloc(NbrTet * sizeof(float));
   Laplacian = malloc(NbrTet * sizeof(float));
   for (i = 0; i < NbrTet; i++)
   {
      GmlGetDataLine(GmlIdx, SolTetIdx, i, &SolTet[i]);
      // GmlGetDataLine(GmlIdx, SolTetTmpIdx, i, &SolTet[i]);
      GmlGetDataLine(GmlIdx, GrdTetIdx, i, Tmp);
      XGrdTet[i] = Tmp[0];
      YGrdTet[i] = Tmp[1];
      ZGrdTet[i] = Tmp[2];
      GmlGetDataLine(GmlIdx, RhsIdx, i, Tmp);
      Laplacian[i] = Tmp[0];
   }
   Write_SolTet("solution.solb", NbrTet, SolTet);
   Write_SolTet("gradx.solb", NbrTet, XGrdTet);
   Write_SolTet("grady.solb", NbrTet, YGrdTet);
   Write_SolTet("gradz.solb", NbrTet, ZGrdTet);
   Write_SolTet("laplacian.solb", NbrTet, Laplacian);
   free(SolTet);
   free(XGrdTet);
   free(Laplacian);
   GmlStop(GmlIdx);

   return 0;
}
