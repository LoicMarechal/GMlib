/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Node-centered advection equation solver               */
/*   Author:            Julien VANHAREN                                       */
/*   Creation date:     may 13 2020                                           */
/*   Last modification: may 13 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/

#include <gmlib3.h>
#include <libmeshb7.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bal_crd.h"
#include "crd.h"
#include "param.h"

typedef struct {
  int toto;
} GmlParSct;

void Adv_Init(int argc, char *argv[], size_t *GmlIdx) {
  int dbg = 0, GpuIdx;

  if (argc == 1) {
    puts("Node-centered advection equation solver");
    puts("Choose the GPU index from the following list:");
    GmlListGPU();
    exit(EXIT_SUCCESS);
  } else
    GpuIdx = atoi(argv[1]);
  if (argc == 3) dbg = 1;
  if (!(*GmlIdx = GmlInit(GpuIdx))) exit(EXIT_SUCCESS);
  if (dbg) GmlDebugOn(*GmlIdx);
}

int main(int argc, char *argv[]) {
  size_t GmlIdx;
  GmlParSct *GmlPar;
  int flag;
  int NbrVer, NbrEdg, NbrTri;
  int VerIdx, EdgIdx, TriIdx;

  /* Library initialization. */
  Adv_Init(argc, argv, &GmlIdx);

  /* Import mesh and print statistics. */
  GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), param);
  GmlImportMesh(GmlIdx, "../sample_meshes/square.meshb", GmfVertices,
                GmfTriangles, 0);
  GetMeshInfo(GmlIdx, GmlVertices, &NbrVer, &VerIdx);
  GetMeshInfo(GmlIdx, GmlEdges, &NbrEdg, &EdgIdx);
  GetMeshInfo(GmlIdx, GmlTriangles, &NbrTri, &TriIdx);
  printf("+++ Imported %d vertices and %d triangles\n", NbrVer, NbrTri);
  GmlExtractEdges(GmlIdx);
  GetMeshInfo(GmlIdx, GmlEdges, &NbrEdg, &EdgIdx);
  printf("+++ %d edges extracted from the surface\n", NbrEdg);

  /* Begin: For each triangles, store the coordinates of its three vertices. */
  // int CrdIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 3, GmlFlt4, "Crd");
  // int CrdKrn = GmlCompileKernel(GmlIdx, crd, "crd", GmlTriangles, 2, VerIdx,
  //                               GmlReadMode, NULL, CrdIdx, GmlWriteMode,
  //                               NULL);
  // printf("+++ Kernel compilation return: %d\n", CrdKrn);
  // flag = GmlLaunchKernel(GmlIdx, CrdKrn);
  // printf("+++ Kernel launch return: %d\n", flag);
  // // int i, j;
  // // float tmp[12];
  // // for (i = 0; i < NbrTri; i++) {
  // //   GmlGetDataLine(GmlIdx, CrdIdx, i, tmp);
  // //   for (j = 0; j < 12; j++) printf("%.3f ", tmp[j]);
  // //   printf("\n");
  // // }
  // /* End: For each triangles, store the coordinates of its three vertices. */

  // int BalCrdIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1, GmlFlt4,
  // "BalCrd");

  // int BalCrdKrn =
  //     GmlCompileKernel(GmlIdx, bal_crd, "bal_crd", GmlVertices, 2, CrdIdx,
  //                      GmlReadMode, NULL, BalCrdIdx, GmlWriteMode, NULL);
  // GmlLaunchKernel(GmlIdx, BalCrdKrn);

  // int i, j;
  // float tmp[4];
  // float ini[4] = {-99.f, -99.f, -99.f, -99.f};

  // for (i = 0; i < NbrVer; i++) GmlSetDataLine(GmlIdx, BalCrdIdx, i, &ini);

  // for (i = 0; i < NbrVer; i++) {
  //   GmlGetDataLine(GmlIdx, BalCrdIdx, i, &tmp);
  //   for (j = 0; j < 4; j++) printf("%.3f ", tmp[j]);
  //   printf("\n");
  // }

  int BalCrdKrn = GmlCompileKernel(GmlIdx, bal_crd, "bal_crd", GmlVertices, 1,
                                   TriIdx, GmlReadMode, NULL);
  printf("+++ Kernel compilation return: %d\n", BalCrdKrn);
  flag = GmlLaunchKernel(GmlIdx, BalCrdKrn);
  printf("+++ Kernel launch return: %d\n", flag);

  // printf("\n");
  // printf("\n");
  // printf("\n");
  // for (i = 0; i < NbrVer; i++) {
  //   GmlGetDataLine(GmlIdx, BalCrdIdx, i, &tmp);
  //   for (j = 0; j < 4; j++) printf("%.3f ", tmp[j]);
  //   printf("\n");
  // }

  GmlStop(GmlIdx);
  return 0;
}
