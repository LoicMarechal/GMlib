/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Cell-centered heat equation solver                    */
/*   Author:            Julien VANHAREN                                       */
/*   Creation date:     apr 15 2020                                           */
/*   Last modification: jun 12 2020                                           */
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
   float dt;
} GmlParSct;

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
   GmlParSct *GmlPar;
   int i, n=1, NbrVer, NbrTri, NbrTet;
   int MemByt[6] = {480,768,576,984,72,24};
   int IntOpp[6] = {312,0,24,24,0,0};
   int FltOpp[6] = {24,1710,432,918,96,198};
   float Zero[4] = {0.f,0.f,0.f,0.f};
   double TotalTime = 0., TotalByte = 0., TotalFlop = 0.;
   double WallTime, PhysTime, Time[6], InitRes, Res, Dbldt;
   /* Indexes */
   int VerIdx, TriIdx, TetIdx, SolTetIdx, GrdTetIdx, SolExtIdx, GrdExtIdx, RhsIdx, dtIdx;
   /* Kernels */
   int IniTetKrn, SolExtKrn, GrdTetKrn, GrdExtKrn, FlxBalKrn, TimKrn, dtKrn;

   /* Library initialization. */
   Heat_Init(argc, argv, &GmlIdx);

   /* Define boundary conditions. */
   GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), param);
   *GmlPar = (GmlParSct){NEUMANN, NEUMANN, NEUMANN, DIRICHLET, DIRICHLET, NEUMANN, 1,2,3,4,5,6,0.,0.,0.,300.,250.,0.,0.};

   /* Import mesh and print statistics. */
   GmlImportMesh(GmlIdx, "../sample_meshes/cube.meshb", GmfVertices, GmfTriangles, GmfTetrahedra);
   GmlGetMeshInfo(GmlIdx, GmlVertices,   &NbrVer, &VerIdx);
   GmlGetMeshInfo(GmlIdx, GmlTetrahedra, &NbrTet, &TetIdx);
   GmlGetMeshInfo(GmlIdx, GmlTriangles,  &NbrTri, &TriIdx);
   printf("+++ Imported %d vertices, %d triangles and %d tetrahedra from the mesh file\n", NbrVer, NbrTri, NbrTet);

   /* Extract all faces connectivity. */
   GmlExtractFaces(GmlIdx);
   GmlGetMeshInfo(GmlIdx, GmlTriangles, &NbrTri, &TriIdx);
   printf("+++ %d triangles extracted from the volume\n", NbrTri);
   printf("+++ Mesh numbering factor = %g%%\n", GmlEvaluateNumbering(GmlIdx));

   /* Fields declaration. */
   SolTetIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt,  "SolTet");
   GrdTetIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt4, "GrdTet");
   SolExtIdx = GmlNewSolutionData(GmlIdx, GmlTriangles,  1, GmlFlt,  "SolExt");
   GrdExtIdx = GmlNewSolutionData(GmlIdx, GmlTriangles,  1, GmlFlt4, "GrdExt");
   RhsIdx    = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt,  "Rhs");
   dtIdx     = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt,  "dt");

   if(!SolTetIdx || !GrdTetIdx || !SolExtIdx || !GrdExtIdx || !RhsIdx || !dtIdx)
   {
      printf(  "Failed to allocate some data: %d %d %d %d %d %d\n",
               SolTetIdx, GrdTetIdx, SolExtIdx, GrdExtIdx, RhsIdx, dtIdx );
      exit(1);
   }

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
                                VerIdx,    GmlReadMode,  NULL,
                                SolTetIdx, GmlWriteMode, NULL);
   SolExtKrn = GmlCompileKernel(GmlIdx, sol_ext, "sol_ext", GmlTriangles, 3,
                                TriIdx,    GmlReadMode | GmlRefFlag, NULL,
                                SolTetIdx, GmlReadMode,  NULL,
                                SolExtIdx, GmlWriteMode, NULL);
   GrdTetKrn = GmlCompileKernel(GmlIdx, grd_tet, "grd_tet", GmlTetrahedra, 3,
                                VerIdx,    GmlReadMode,  NULL,
                                SolExtIdx, GmlReadMode,  NULL,
                                GrdTetIdx, GmlWriteMode, NULL);
   GrdExtKrn = GmlCompileKernel(GmlIdx, grd_ext, "grd_ext", GmlTriangles, 3,
                                TriIdx,    GmlReadMode | GmlRefFlag, NULL,
                                GrdTetIdx, GmlReadMode,  NULL,
                                GrdExtIdx, GmlWriteMode, NULL);
   FlxBalKrn = GmlCompileKernel(GmlIdx, flx_bal, "flx_bal", GmlTetrahedra, 3,
                                VerIdx,    GmlReadMode,  NULL,
                                GrdExtIdx, GmlReadMode,  NULL,
                                RhsIdx,    GmlWriteMode, NULL);
   TimKrn    = GmlCompileKernel(GmlIdx, tim_int, "tim_int", GmlTetrahedra, 2,
                                RhsIdx,    GmlReadMode,  NULL,
                                SolTetIdx, GmlReadMode | GmlWriteMode, NULL);
   dtKrn     = GmlCompileKernel(GmlIdx, dt, "dt",           GmlTetrahedra, 2,
                                VerIdx,    GmlReadMode,  NULL,
                                dtIdx,     GmlWriteMode, NULL);

   if(!IniTetKrn || !SolExtKrn || !GrdTetKrn || !GrdExtKrn || !FlxBalKrn || !TimKrn || !dtKrn)
   {
      printf(  "Failed to compile some kernels: %d %d %d %d %d %d %d\n",
               IniTetKrn, SolExtKrn, GrdTetKrn, GrdExtKrn, FlxBalKrn, TimKrn, dtKrn );
      exit(1);
   }

   /* Solution initialization. */
   /* Time = GmlLaunchKernel(GmlIdx, IniTetKrn); */
   /* Begin resolution. */

   WallTime = GmlGetWallClock();
   GmlLaunchKernel(GmlIdx, dtKrn);
   GmlReduceVector(GmlIdx, dtIdx, GmlMin, &Dbldt);
	GmlPar->dt = (float)Dbldt;
   printf("+++ Time step = %.3E\n", GmlPar->dt);
   GmlUploadParameters(GmlIdx);

   do
   {
      GmlLaunchKernel(GmlIdx, SolExtKrn);
      GmlLaunchKernel(GmlIdx, GrdTetKrn);
      GmlLaunchKernel(GmlIdx, GrdExtKrn);
      GmlLaunchKernel(GmlIdx, FlxBalKrn);
      GmlLaunchKernel(GmlIdx, TimKrn);
      GmlReduceVector(GmlIdx, RhsIdx, GmlL2, &Res);
      if(n==1)
         InitRes = Res;
      if (!(n % 1000))
         printf("+++ Iteration %6d Residual = %.3E\n", n, Res / InitRes);
      n++;
   }while( (Res / InitRes > 1e-6) && (n < 100000) );

   printf("+++ Iteration %6d Residual = %.3E\n\n", n, Res / InitRes);

   GmlDownloadParameters(GmlIdx);

   Time[0] = GmlGetKernelRunTime(GmlIdx, SolExtKrn);
   Time[1] = GmlGetKernelRunTime(GmlIdx, GrdTetKrn);
   Time[2] = GmlGetKernelRunTime(GmlIdx, GrdExtKrn);
   Time[3] = GmlGetKernelRunTime(GmlIdx, FlxBalKrn);
   Time[4] = GmlGetKernelRunTime(GmlIdx, TimKrn);
   Time[5] = GmlGetReduceRunTime(GmlIdx, GmlL2);

   for (i = 0; i < 6; i++)
   {
      TotalTime += Time[i];
      TotalByte += (double)NbrVer * n * MemByt[i];
      TotalFlop += (double)NbrVer * n * (IntOpp[i]+FltOpp[i]);
      printf("Profiling time for kernel %d = %6.2f seconds,", i, Time[i]);
      printf(" %8.2f GB/s,",    ((double)NbrVer * n * MemByt[i]) / (Time[i] * 1E9));
      printf(" %8.2f Gflops\n", ((double)NbrVer * n * (IntOpp[i]+FltOpp[i])) / (Time[i] * 1E9));
   }

   PhysTime = GmlGetWallClock() - WallTime;
   printf("Total profiling time        = %6.2f seconds\n", TotalTime);
   printf("Wall clock time             = %6.2f seconds,", PhysTime);
   printf(" %8.2f GB/s,",    TotalByte / (PhysTime * 1E9));
   printf(" %8.2f Gflops\n", TotalFlop / (PhysTime * 1E9));

   GmlExportSolution(GmlIdx, "../sample_meshes/cube.solb", SolTetIdx, GrdTetIdx, RhsIdx, 0);
   GmlStop(GmlIdx);

   return 0;
}
