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

enum BC_TYPE { DIRICHLET, NEUMANN };
typedef struct {
  int Cnt;
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
   float Ini[4] = {0., 0., 0., 0.};
   double NgbTim = 0, TetTim = 0, VerTim = 0, RedTim = 0, F64Tim = 0;
   double res, FlxTim = 0., GtrTim = 0., ResTim = 0.;
   float *Sol, *XGrdTet;
   GmlParSct *GmlPar;

   int n;
   int IniTetKrn, SolTetIdx, GrdTetIdx, SolExtIdx, GrdExtIdx, RhsIdx;
   int SolExtKrn, GrdTetKrn, GrdExtKrn, FlxBalKrn, TimKrn;
   double Time, InitRes, Res;
   float *SolTet, Tmp[4];

   /* Library initialization. */
   Heat_Init(argc, argv, &GmlIdx);

   /* Define boundary conditions. */
   GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), param);
   GmlPar->Cnt = 0;
   for (i = 0; i < 6; i++)
      GmlPar->BoCo[i] = DIRICHLET;
   for (i = 0; i < 6; i++)
      GmlPar->BoCo_Ref[i] = i + 1;
   for (i = 0; i < 6; i++)
      GmlPar->BoCo_Val[i] = 1.234;

   /* Import mesh and print statistics. */
<<<<<<< HEAD
   GmlImportMesh(GmlIdx, "../../sample_meshes/tetrahedra.meshb", GmfVertices, GmfTriangles, GmfTetrahedra);
=======
   GmlImportMesh(GmlIdx, "../sample_meshes/cube.meshb", GmfVertices, GmfTriangles, GmfTetrahedra);
>>>>>>> solver
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

   /* Kernels compilation. */
   IniTetKrn = GmlCompileKernel(GmlIdx, ini_tet, "ini_tet", GmlTetrahedra, 2,
                                VerIdx, GmlReadMode, NULL,
                                SolTetIdx, GmlWriteMode, NULL);
   SolExtKrn = GmlCompileKernel(GmlIdx, sol_ext, "sol_ext", GmlTriangles, 3,
                                TriIdx, GmlReadMode | GmlRefFlag, NULL,
                                SolTetIdx, GmlReadMode, NULL,
                                SolExtIdx, GmlWriteMode, NULL);
   // GrdTetKrn = GmlCompileKernel(GmlIdx, grd_tet, "grd_tet", GmlTetrahedra, 3,
   //                              VerIdx, GmlReadMode, NULL,
   //                              SolExtIdx, GmlReadMode, NULL,
   //                              GrdTetIdx, GmlWriteMode, NULL);
   // GrdExtKrn = GmlCompileKernel(GmlIdx, grd_ext, "grd_ext", GmlTriangles, 3,
   //                              TriIdx, GmlReadMode | GmlRefFlag, NULL,
   //                              GrdTetIdx, GmlReadMode, NULL,
   //                              GrdExtIdx, GmlWriteMode, NULL);
   // FlxBalKrn = GmlCompileKernel(GmlIdx, flx_bal, "flx_bal", GmlTetrahedra, 3,
   //                              VerIdx, GmlReadMode, NULL,
   //                              GrdExtIdx, GmlReadMode, NULL,
   //                              RhsIdx, GmlWriteMode, NULL);
   // TimKrn = GmlCompileKernel(GmlIdx, tim_int, "tim_int", GmlTetrahedra, 2,
   //                           RhsIdx, GmlReadMode, NULL,
   //                           SolTetIdx, GmlWriteMode, NULL);

   Time = GmlLaunchKernel(GmlIdx, IniTetKrn);
   Time = GmlLaunchKernel(GmlIdx, SolExtKrn);
   // Time = GmlLaunchKernel(GmlIdx, GrdTetKrn);

   /* Begin resolution. */
   // Time = GmlReduceVector(GmlIdx, RhsIdx, GmlSum, &InitRes);
   // for (n = 1; n <= 1; n++)
   // {
   //    Time = GmlLaunchKernel(GmlIdx, SolExtKrn);
   //    Time = GmlLaunchKernel(GmlIdx, GrdTetKrn);
   //    // Time = GmlLaunchKernel(GmlIdx, GrdExtKrn);
   //    // Time = GmlLaunchKernel(GmlIdx, FlxBalKrn);
   //    // Time = GmlLaunchKernel(GmlIdx, TimKrn);
   //    // Time = GmlReduceVector(GmlIdx, RhsIdx, GmlSum, &Res);
   //    // printf("+++ Iteration %4d Residual = %.12f\n", n, Res / InitRes);
   // }
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

   for (i = 0; i < NbrTri; i++)
   {
      GmlGetDataLine(GmlIdx, SolExtIdx, i, Tmp);
      if (Tmp[0] > 1.235 || Tmp[0] < 1.233)
         printf("SolExt %.12f iTri %d\n", Tmp[0], i);
   }

   // SolTet = malloc(NbrTet * sizeof(float));
   // XGrdTet = malloc(NbrTet * sizeof(float));
   // for (i = 0; i < NbrTet; i++)
   // {
   //    GmlGetDataLine(GmlIdx, SolTetIdx, i, &SolTet[i]);
   //    GmlGetDataLine(GmlIdx, GrdTetIdx, i, Tmp);
   //    XGrdTet[i] = Tmp[0];
   // }
   // Write_SolTet("solution.solb", NbrTet, SolTet);
   // Write_SolTet("gradx.solb", NbrTet, XGrdTet);
   // free(SolTet);
   // free(XGrdTet);

   return 0;
}
