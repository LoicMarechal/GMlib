

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.42                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Compute the direct and indirect memory bandwidth      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     mar 24 2025                                           */
/*   Last modification: mar 31 2025                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmeshb8.h>
#include <gmlib3.h>
#include "parameters.h"
#include "direct_vertices.h"
#include "direct_tets.h"
#include "indirect_tets.h"
#include "vertices_ball.h"


/*----------------------------------------------------------------------------*/
/* This structure definition must be exactly the same as the OpenCL one       */
/*----------------------------------------------------------------------------*/

typedef struct {
   int   foo;
   float res;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements quality and get minimal and mean values                       */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         i, NmbVer, NmbTet, NmbItr, VerRet, TetRet, IndRet, BalRet;
   int         VerDatIdx, VerKrn, TetDatIdx, TetKrn, IndKrn, VerIdx, TetIdx;
   int         GpuIdx = 0, BalKrn;
   size_t      GmlIdx;
   double      tim = 0;
   char        *MshNam;
   GmlParSct   *GmlPar;


   // If no arguments are given, print the help
   if(ArgCnt != 4)
   {
      puts("\ngpu_bandwidth   GpuIndex   NmbIter   MeshFile");
      puts("In order to fully evaluate all kinds of memory access performances you should generate");
      puts("two different numberings from the same test mesh and feed them to cpu_bandwith:\n");
      puts(" hilbert -in MyTestMesh -out RandomMesh -scheme 2");
      puts(" hilbert -in MyTestMesh -out HilbertMesh -gmlib generic\n");
      puts(" Choose GpuIndex from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
   {
      GpuIdx = atoi(ArgVec[1]);
      NmbItr = atoi(ArgVec[2]);
      MshNam = ArgVec[3];
   }

   // Init the GMLIB and compile the OpenCL source code
   if(!(GmlIdx = GmlInit(GpuIdx)))
      return(1);

   //GmlDebugOn(GmlIdx);

   // Allocate and read the mesh
   GmlImportMesh(GmlIdx, MshNam, GmfVertices, GmfTetrahedra, 0);

   if(!GmlGetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf(  "\nImported %d vertices and %d tets from the mesh file\n\n",
            NmbVer, NmbTet );

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   // Allocate arbitrary data to test bandwidth
   if(!(VerDatIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 1, GmlFlt4, "VerDat")))
      return(1);

   if(!(TetDatIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt4, "TetDat")))
      return(1);

   // Compile a kernel that performs vertices direct read & write
   VerKrn = GmlCompileKernel( GmlIdx, direct_vertices, "direct_vertices",
                              GmlVertices, 2,
                              VerIdx, GmlReadMode, NULL,
                              VerDatIdx, GmlWriteMode, NULL );

   // Compile a kernel that performs tets direct read & write
   TetKrn = GmlCompileKernel( GmlIdx, direct_tets, "direct_tets",
                              GmlTetrahedra, 2,
                              TetIdx, GmlReadMode, NULL,
                              TetDatIdx, GmlWriteMode, NULL );

   // Compile a kernel that performs indirect reads from tets to vertices
   IndKrn = GmlCompileKernel( GmlIdx, indirect_tets, "indirect_tets",
                              GmlTetrahedra, 2,
                              VerIdx, GmlReadMode, NULL,
                              TetDatIdx, GmlWriteMode, NULL );

   // Compile a kernel that performs indirect reads from vertices to tets
   BalKrn = GmlCompileKernel( GmlIdx, vertices_ball, "vertices_ball",
                              GmlVertices, 2,
                              VerDatIdx, GmlWriteMode, NULL,
                              TetDatIdx, GmlReadMode, NULL );

   // Check compilation of the three kernels
   if(!VerKrn || !TetKrn || !IndKrn || !BalKrn)
   {
      printf("Failed to compile kernel: VerKrn = %d, TetKrn = %d, IndKrn = %d, BalKrn = %d\n",
            VerKrn, TetKrn, IndKrn, BalKrn);
      return(1);
   }

   // Perform direct memory access kernels
   tim = GmlGetWallClock();

   for(i=1;i<=NmbItr;i++)
   {
      VerRet = GmlLaunchKernel(GmlIdx, VerKrn);
      TetRet = GmlLaunchKernel(GmlIdx, TetKrn);
   }

   tim = GmlGetWallClock() - tim;

   printf("Direct reads   : %d steps, run time = %7.3fs, bandwidth = %6.1f GB/s\n",
            NmbItr, tim, (NmbItr * (32LL * NmbVer + 32LL * NmbTet)) / (tim * 1E9) );

   // Perform indirect memory access kernel
   tim = GmlGetWallClock();

   for(i=1;i<=NmbItr;i++)
      IndRet = GmlLaunchKernel(GmlIdx, IndKrn);

   tim = GmlGetWallClock() - tim;

   printf("Indirect reads : %d steps, run time = %7.3fs, bandwidth = %6.1f GB/s (unique reads = %6.1f GB/s)\n",
            NmbItr, tim, (NmbItr * 96LL * NmbTet) / (tim * 1E9),
            (NmbItr * (16LL * NmbVer + 32LL * NmbTet)) / (tim * 1E9) );


   // Perform indirect upward memory access kernel
   tim = GmlGetWallClock();

   for(i=1;i<=NmbItr;i++)
      BalRet = GmlLaunchKernel(GmlIdx, BalKrn);

   tim = GmlGetWallClock() - tim;

   printf("Ragged reads   : %d steps, run time = %7.3fs, bandwidth = %6.1f GB/s (unique reads = %6.1f GB/s)\n",
            NmbItr, tim, (NmbItr * 460LL * NmbVer) / (tim * 1E9),
            (NmbItr * (32LL * NmbVer + 16LL * NmbTet)) / (tim * 1E9) );

   // Check execution of the three kernels
   if(VerRet < 0 || TetRet < 0 || IndRet < 0 || BalRet < 0)
   {
      printf("Failed to launch kernel:  VerKrn = %d, TetKrn = %d, IndKrn = %d\n",
      VerRet, TetRet, IndRet);
      return(2);
   }

   // Print GPU memoy allocated and transfered
   printf("\nGPU memory: %.2f GB used, %.2f GB transfered\n\n",
          GmlGetMemoryUsage   (GmlIdx) / 1.E9,
          GmlGetMemoryTransfer(GmlIdx) / 1.E9);


   // Free everything on the GPU
   GmlFreeData(GmlIdx, VerIdx);
   GmlFreeData(GmlIdx, TetIdx);
   GmlFreeData(GmlIdx, VerDatIdx);
   GmlFreeData(GmlIdx, TetDatIdx);
   GmlStop(GmlIdx);

   return(0);
}
