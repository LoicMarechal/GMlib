/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Cell-centered heat equation solver                    */
/*   Author:            Julien VANHAREN                                       */
/*   Creation date:     apr 15 2020                                           */
/*   Last modification: apr 15 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmeshb7.h>
#include <gmlib3.h>

#include "param.h"
#include "scatter.h"
#include "gather.h"

typedef struct
{
   int foo;
   float res;
} GmlParSct;

void WriteSolution(char *out_fn, int NmbTet, float *Sol)
{
   int64_t fid;
   int iTet, NmbTyp = 1, TypTab[1] = {GmfSca};

   fid = GmfOpenMesh(out_fn, GmfWrite, 1, 3);
   if (NmbTet)
   {
      GmfSetKwd(fid, GmfSolAtTetrahedra, NmbTet, NmbTyp, TypTab);
      for (iTet = 0; iTet < NmbTet; iTet++)
         GmfSetLin(fid, GmfSolAtTetrahedra, &(Sol[iTet]));
   }
   GmfCloseMesh(fid);
}

int main(int ArgCnt, char **ArgVec)
{
   int i, j, NmbVer = 0, NmbTri = 0, NmbTet = 0;
   int ParIdx, VerIdx = 0, TriIdx = 0, TetIdx = 0, BalIdx, MidIdx, SolIdx, FlxIdx, CalMid, ScatterIdx, GatherIdx, OptVer;
   int GpuIdx = 0, ResIdx, NgbIdx, NgbKrn, F64Idx, F64Krn;
   size_t GmlIdx;
   float MidTab[4], SolTab[8], TetChk = 0., VerChk = 0.;
   float IniSol[1] = {10.};
   double NgbTim = 0, TetTim = 0, VerTim = 0, RedTim = 0, F64Tim = 0;
   double res, residual;
   float *Flx;
   GmlParSct *GmlPar;

   if (ArgCnt == 1)
   {
      puts("Cell-centered heat equation solver");
      puts("Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
      GpuIdx = atoi(ArgVec[1]);

   if (!(GmlIdx = GmlInit(GpuIdx)))
      return (1);

   // GmlDebugOn(GmlIdx);

   GmlImportMesh(GmlIdx, "../sample_meshes/tetrahedra.meshb", GmfVertices, GmfTetrahedra, GmfTriangles, 0);
   GetMeshInfo(GmlIdx, GmlVertices, &NmbVer, &VerIdx);
   GetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx);
   GetMeshInfo(GmlIdx, GmlTriangles, &NmbTri, &TriIdx);
   printf("Imported %d vertices, %d triangles and %d tets from the mesh file\n", NmbVer, NmbTri, NmbTet);

   GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), param);
   NgbIdx = GmlSetNeighbours(GmlIdx, GmlTetrahedra);
   SolIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 2, GmlFlt, "Sol");
   // FlxIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "Flx");

   for (i = 0; i < NmbTet; i++)
      GmlSetDataLine(GmlIdx, SolIdx, i, &IniSol);

   ResIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "Res");

   // ScatterIdx = GmlCompileKernel(GmlIdx, scatter, "scatter",
   //                               GmlTetrahedra, 2,
   //                               SolIdx, GmlReadMode, NULL,
   //                               FlxIdx, GmlWriteMode, NULL);

   GatherIdx = GmlCompileKernel(GmlIdx, gather, "gather", GmlTriangles, 2,
                                SolIdx, GmlWriteMode, NULL,
                                TriIdx, GmlReadMode | GmlVoyeurs, NULL);

   res = GmlLaunchKernel(GmlIdx, GatherIdx);

   // Flx = malloc(NmbTet * sizeof(float));
   // for (i = 0; i < NmbTet; i++)
   //    GmlGetDataLine(GmlIdx, FlxIdx, i, &(Flx[i]));
   // printf("After GPU %.3f\n", Flx[0]);

   // WriteSolution("output.sol", NmbTet, Flx);

   //    if (res < 0)
   //    {
   //       printf("Launch kernel %d failled with error: %g\n", CalMid, res);
   //       exit(0);
   //    }

   //    TetTim += res;

   //    // Launch the vertex kernel on the GPU
   //    res = GmlLaunchKernel(GmlIdx, OptVer);

   //    if (res < 0)
   //    {
   //       printf("Launch kernel %d failled with error: %g\n", OptVer, res);
   //       exit(0);
   //    }

   //    VerTim += res;

   //    // Launch the reduction kernel on the GPU
   //    res = GmlReduceVector(GmlIdx, ResIdx, GmlSum, &residual);

   //    if (res < 0)
   //    {
   //       printf("Launch reduction kernel failled with error: %g\n", res);
   //       exit(0);
   //    }

   //    RedTim += res;
   //    printf("Iteration: %3d, residual: %g\n", i, residual);
   // }

   // /*-----------------*/
   // /* GET THE RESULTS */
   // /*-----------------*/

   // // Get back the MidTet data from the GPU memory and compute a checksum
   // for (i = 0; i < NmbTet; i++)
   // {
   //    GmlGetDataLine(GmlIdx, MidIdx, i, MidTab);
   //    TetChk += MidTab[0] + MidTab[1] + MidTab[2];
   // }

   // // Get back the SolAtVer data from the GPU memory and compute a checksum
   // for (i = 0; i < NmbVer; i++)
   // {
   //    GmlGetDataLine(GmlIdx, SolIdx, i, SolTab);
   //    VerChk += SolTab[0] + SolTab[1] + SolTab[2] + SolTab[3];
   // }

   // printf("%d tets processed in %g seconds, FP64=%g, ngb access=%g, scater=%g, gather=%g, reduction=%g\n",
   //        NmbTet, F64Tim + NgbTim + TetTim + VerTim + RedTim, F64Tim, NgbTim, TetTim, VerTim, RedTim);

   // printf("%ld MB used, %ld MB transfered\n",
   //        GmlGetMemoryUsage(GmlIdx) / 1048576,
   //        GmlGetMemoryTransfer(GmlIdx) / 1048576);

   // printf("MidTet checksum = %g, SolAtVer checksum = %g\n",
   //        TetChk / NmbTet, VerChk / NmbVer);

   // /*-----*/
   // /* END */
   // /*-----*/

   // GmlFreeData(GmlIdx, VerIdx);
   // GmlFreeData(GmlIdx, TetIdx);
   // GmlFreeData(GmlIdx, MidIdx);
   // GmlFreeData(GmlIdx, ParIdx);
   // GmlStop(GmlIdx);

   return (0);
}
