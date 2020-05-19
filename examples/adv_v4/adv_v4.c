/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Node-centered advection equation solver               */
/*   Author:            Julien VANHAREN                                       */
/*   Creation date:     may 13 2020                                           */
/*   Last modification: may 13 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmlib3.h>
#include <libmeshb7.h>

void Adv_Init(int argc, char *argv[], size_t *GmlIdx)
{
    int dbg = 0, GpuIdx;

    if (argc == 1)
    {
        puts("Node-centered advection equation solver");
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
    int NbrVer, NbrEdg, NbrTri;
    int VerIdx, EdgIdx, TriIdx;

    /* Library initialization. */
    Adv_Init(argc, argv, &GmlIdx);

    /* Import mesh and print statistics. */
    GmlImportMesh(GmlIdx, "../sample_meshes/square.meshb", GmfVertices, GmfTriangles);
    GetMeshInfo(GmlIdx, GmlVertices, &NbrVer, &VerIdx);
    // GetMeshInfo(GmlIdx, GmlEdges, &NbrEdg, &EdgIdx);
    GetMeshInfo(GmlIdx, GmlTriangles, &NbrTri, &TriIdx);
    printf("+++ Imported %d vertices and %d triangles\n", NbrVer, NbrTri);
    GmlExtractEdges(GmlIdx);
    GetMeshInfo(GmlIdx, GmlEdges, &NbrEdg, &EdgIdx);
    printf("+++ %d edges extracted from the surface\n", NbrEdg);

    GmlStop(GmlIdx);
    return 0;
}
