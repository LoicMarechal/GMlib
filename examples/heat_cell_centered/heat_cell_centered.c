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
#include "dt.h"

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
   double dt;
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
   size_t GmlIdx;
<<<<<<< HEAD
   float MidTab[4], SolTab[8], TetChk = 0., VerChk = 0.;
   float Zero[4] = {0.f, 0.f, 0.f, 0.f};
   double NgbTim = 0, TetTim = 0, VerTim = 0, RedTim = 0, F64Tim = 0;
   double TotalTime = 0., res, FlxTim = 0., GtrTim = 0., ResTim = 0.;
   float *Sol, *XGrdTet, *YGrdTet, *ZGrdTet, *Laplacian;
   GmlParSct *GmlPar;

   int n;
   int IniTetKrn, SolTetIdx, SolTetTmpIdx, GrdTetIdx, SolExtIdx, GrdExtIdx, RhsIdx;
   int SolExtKrn, GrdTetKrn, GrdExtKrn, FlxBalKrn, TimKrn;
   double Time[6]={0.}, InitRes, Res;
   float *SolTet, Tmp[4];
=======
   GmlParSct *GmlPar;
   int i, n, NbrVer, NbrTri, NbrTet;
   float Tmp[4], Zero[4] = {0.f, 0.f, 0.f, 0.f};
   float *SolTet, *XGrdTet, *YGrdTet, *ZGrdTet, *Laplacian;
   double TotalTime = 0., WallTime, Time[6] = {0.}, InitRes, Res;
   /* Indexes */
   int VerIdx, TriIdx, TetIdx, SolTetIdx, GrdTetIdx, SolExtIdx, GrdExtIdx, RhsIdx, dtIdx;
   /* Kernels */
   int IniTetKrn, SolExtKrn, GrdTetKrn, GrdExtKrn, FlxBalKrn, TimKrn, dtKrn;
>>>>>>> c2ac3a64426a7bc971446fd2706f74231c9b79b1

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
   GrdTetIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt4, "GrdTet");
   SolExtIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 1, GmlFlt, "SolExt");
   GrdExtIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 1, GmlFlt4, "GrdExt");
   RhsIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "Rhs");
   dtIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "dt");

   for (i = 0; i < NbrTet; i++)
   {
      GmlSetDataLine(GmlIdx, SolTetIdx, i, Zero);
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
   dtKrn = GmlCompileKernel(GmlIdx, dt, "dt", GmlTetrahedra, 2,
                            VerIdx, GmlReadMode, NULL,
                            dtIdx, GmlWriteMode, NULL);

   /* Solution initialization. */
   /* Time = GmlLaunchKernel(GmlIdx, IniTetKrn); */
   /* Begin resolution. */
<<<<<<< HEAD
   for (n = 1; n <= 1000; n++)
   {
      Time[0] += GmlLaunchKernel(GmlIdx, SolExtKrn);
      Time[1] += GmlLaunchKernel(GmlIdx, GrdTetKrn);
      Time[2] += GmlLaunchKernel(GmlIdx, GrdExtKrn);
      Time[3] += GmlLaunchKernel(GmlIdx, FlxBalKrn);
      Time[4] += GmlLaunchKernel(GmlIdx, TimKrn);
      Time[5] += GmlReduceVector(GmlIdx, RhsIdx, GmlSum, &Res);
      printf("\r+++ Iteration %4d Residual = %.12f", n, Res);
      fflush(stdout);
   }

   puts("");
   for(i=0;i<6;i++)
   {
      TotalTime += Time[i];
      printf("Total time for kernel %d = %g seconds\n", i, Time[i]);
   }

   printf("Total runing time       = %g seconds\n", TotalTime);

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
=======
>>>>>>> c2ac3a64426a7bc971446fd2706f74231c9b79b1

   WallTime = GmlGetWallClock();
   GmlLaunchKernel(GmlIdx, dtKrn);
   GmlReduceVector(GmlIdx, dtIdx, GmlMin, &GmlPar->dt);
   printf("+++ Time step = %.3E\n", GmlPar->dt);
   GmlUploadParameters(GmlIdx);

   GmlLaunchKernel(GmlIdx, SolExtKrn);
   GmlLaunchKernel(GmlIdx, GrdTetKrn);
   GmlLaunchKernel(GmlIdx, GrdExtKrn);
   GmlLaunchKernel(GmlIdx, FlxBalKrn);
   GmlLaunchKernel(GmlIdx, TimKrn);
   GmlReduceVector(GmlIdx, RhsIdx, GmlSum, &InitRes);

   for (n = 1; n <= 100000; n++)
   {
      GmlLaunchKernel(GmlIdx, SolExtKrn);
      GmlLaunchKernel(GmlIdx, GrdTetKrn);
      GmlLaunchKernel(GmlIdx, GrdExtKrn);
      GmlLaunchKernel(GmlIdx, FlxBalKrn);
      GmlLaunchKernel(GmlIdx, TimKrn);
      GmlReduceVector(GmlIdx, RhsIdx, GmlSum, &Res);
      if (!(n % 1000))
         printf("+++ Iteration %6d Residual = %.3E\n", n, Res / InitRes);
   }

   GmlDownloadParameters(GmlIdx);

   Time[0] = GmlGetKernelRunTime(GmlIdx, SolExtKrn);
   Time[1] = GmlGetKernelRunTime(GmlIdx, GrdTetKrn);
   Time[2] = GmlGetKernelRunTime(GmlIdx, GrdExtKrn);
   Time[3] = GmlGetKernelRunTime(GmlIdx, FlxBalKrn);
   Time[4] = GmlGetKernelRunTime(GmlIdx, TimKrn);
   Time[5] = GmlGetReduceRunTime(GmlIdx, GmlSum);

   puts("\n\n");
   for (i = 0; i < 6; i++)
   {
      TotalTime += Time[i];
      printf("Total time for kernel %d = %g seconds\n", i, Time[i]);
   }

   printf("GPU execution time = %g seconds\n", TotalTime);
   printf("Wall clock         = %g seconds\n", GmlGetWallClock() - WallTime);

   SolTet = malloc(NbrTet * sizeof(float));
   XGrdTet = malloc(NbrTet * sizeof(float));
   YGrdTet = malloc(NbrTet * sizeof(float));
   ZGrdTet = malloc(NbrTet * sizeof(float));
   Laplacian = malloc(NbrTet * sizeof(float));
   for (i = 0; i < NbrTet; i++)
   {
      GmlGetDataLine(GmlIdx, SolTetIdx, i, &SolTet[i]);
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
   free(YGrdTet);
   free(ZGrdTet);
   free(Laplacian);
   GmlStop(GmlIdx);

   return 0;
}
