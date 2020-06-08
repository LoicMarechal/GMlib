

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.29                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on tetrahedra                              */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     nov 21 2019                                           */
/*   Last modification: jun 05 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmeshb7.h>
#include <gmlib3.h>

#include "parameters.h"
#include "neighbours.h"
#include "downlink.h"
#include "uplink.h"
#include "double_precision.h"


/*----------------------------------------------------------------------------*/
/* This structure definition must be exactly the same as the OpenCL one       */
/*----------------------------------------------------------------------------*/

typedef struct {
   int   foo;
   float res;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements middle and get back the results.                              */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         i, j, res, NmbVer=0, NmbTri=0, NmbTet=0, CalMid, OptVer, FlxIdx;
   int         VerIdx=0, TriIdx=0, TetIdx=0, BalIdx, MidIdx, SolIdx;
   int         GpuIdx = 0, ResIdx, NgbIdx, NgbKrn, F64Idx, F64Krn, FlxKrn;
   int         n, w, N, W;
   size_t      GmlIdx;
   float       MidTab[4], SolTab[8], TetChk = 0., VerChk = 0.;
   float       IniSol[8] = {.125, .125, .125, .125, .125, .125, .125, .125};
   double      NgbTim, TetTim, VerTim, RedTim, F64Tim;
   double      residual;
   GmlParSct   *GmlPar;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nTetrahedraLoop GPU_index");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
      GpuIdx = atoi(ArgVec[1]);


   // Init the GMLIB and compile the OpenCL source code
   if(!(GmlIdx = GmlInit(GpuIdx)))
      return(1);

   //GmlDebugOn(GmlIdx);

   GmlImportMesh( GmlIdx, "../sample_meshes/tetrahedra.meshb",
                  GmfVertices, GmfTriangles, GmfTetrahedra, 0 );

   if(!GmlGetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTriangles,   &NmbTri, &TriIdx))
      return(1);

   if(!GmlGetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf("Imported %d vertices and %d tets from the mesh file\n", NmbVer, NmbTet);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   // Build neighbours between tets as a user defined topological link
   NgbIdx = GmlSetNeighbours(GmlIdx, GmlTetrahedra);

   // Create a raw datatype to store some value at vertices.
   if(!(SolIdx = GmlNewSolutionData(GmlIdx, GmlVertices, 2, GmlFlt4, "SolAtVer")))
      return(1);

   // Fill the initial field with crap
   for(i=0;i<NmbVer;i++)
      GmlSetDataLine(GmlIdx, SolIdx, i, &IniSol);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt4, "TetMid")))
      return(1);

   if(!(FlxIdx = GmlNewSolutionData(GmlIdx, GmlTriangles, 1, GmlFlt, "Flx")))
      return(1);

   // Allocate a residual vector
   if(!(ResIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "ResVec")))
      return(1);

   // FlxKrn = GmlCompileKernel( GmlIdx, flux, "flux",GmlTriangles, 2,
   //                            FlxIdx, GmlWriteMode, NULL,
   //                            MidIdx, GmlReadMode,  NULL );

   if(GmlCheckFP64(GmlIdx))
   {
      puts("This device has FP64 capability.");

      if(!(F64Idx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlDbl, "F64Dat")))
         return(1);

      // Assemble and compile a neighbours kernel
      F64Krn = GmlCompileKernel( GmlIdx, double_precision, "double_precision",
                                 GmlTetrahedra, 1,
                                 F64Idx, GmlReadMode, NULL );

      if(!F64Krn)
         return(1);
   }
   else
      F64Idx = F64Krn = 0;

   // Assemble and compile a neighbours kernel
   NgbKrn = GmlCompileKernel( GmlIdx, neighbours, "neighbours",
                              GmlTetrahedra, 1,
                              MidIdx, GmlReadMode, NgbIdx );

   if(!NgbKrn)
      return(1);

   // Assemble and compile the scatter kernel
   CalMid = GmlCompileKernel( GmlIdx, downlink, "downlink",
                              GmlTetrahedra, 4,
                              VerIdx, GmlReadMode | GmlRefFlag,  NULL,
                              SolIdx, GmlReadMode,  NULL,
                              ResIdx, GmlWriteMode, NULL,
                              MidIdx, GmlWriteMode, NULL );

   if(!CalMid)
      return(1);

   // Assemble and compile the gather kernel
   OptVer = GmlCompileKernel( GmlIdx, uplink, "uplink",
                              GmlVertices, 2,
                              SolIdx, GmlWriteMode, NULL,
                              MidIdx, GmlReadMode | GmlVoyeurs,  NULL );

   if(!OptVer)
      return(1);

   // Print some information about the uplink:
   // length and width of the base link and the high link
   if(GmlGetLinkInfo(GmlIdx, GmlVertices, GmlTetrahedra, &n, &w, &N, &W))
      printf("GMlib uplink ver -> tet: base: %d vertices x %d tets, extended: %d vertices x %d tets\n", n, w, N, W);

   for(i=1;i<=100;i++)
   {
      GmlPar->res = i;
      GmlUploadParameters(GmlIdx);

      if(F64Krn)
      {

         // Launch the tetrahedra kernel on the GPU
         res  = GmlLaunchKernel(GmlIdx, F64Krn);

         if(res < 0)
         {
            printf("Launch kernel %d failled with error: %d\n", F64Krn, res);
            exit(0);
         }
      }

      // Launch the tetrahedra kernel on the GPU
      res  = GmlLaunchKernel(GmlIdx, NgbKrn);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", NgbKrn, res);
         exit(0);
      }

      // Launch the tetrahedra kernel on the GPU
      res  = GmlLaunchKernel(GmlIdx, CalMid);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", CalMid, res);
         exit(0);
      }

      // Launch the vertex kernel on the GPU
      res = GmlLaunchKernel(GmlIdx, OptVer);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %d\n", OptVer, res);
         exit(0);
      }

      // Launch the reduction kernel on the GPU
      res = GmlReduceVector(GmlIdx, ResIdx, GmlSum, &residual);

      if(res < 0)
      {
         printf("Launch reduction kernel failled with error: %d\n", res);
         exit(0);
      }

      printf("Iteration: %3d, residual: %g\n", i, residual);
   }


   /*-----------------*/
   /* GET THE RESULTS */
   /*-----------------*/

   if(F64Krn)
      F64Tim = GmlGetKernelRunTime(GmlIdx, F64Krn);
   else
      F64Tim = 0.;

   NgbTim = GmlGetKernelRunTime(GmlIdx, NgbKrn);
   TetTim = GmlGetKernelRunTime(GmlIdx, CalMid);
   VerTim = GmlGetKernelRunTime(GmlIdx, OptVer);
   RedTim = GmlGetReduceRunTime(GmlIdx, GmlSum);

   // Get back the MidTet data from the GPU memory and compute a checksum
   for(i=0;i<NmbTet;i++)
   {
      GmlGetDataLine(GmlIdx, MidIdx, i, MidTab);
      TetChk += MidTab[0] + MidTab[1] + MidTab[2];
   }

   // Get back the SolAtVer data from the GPU memory and compute a checksum
   for(i=0;i<NmbVer;i++)
   {
      GmlGetDataLine(GmlIdx, SolIdx, i, SolTab);
      VerChk += SolTab[0] + SolTab[1] + SolTab[2] + SolTab[3];
   }

   printf("%d tets processed in %g seconds, FP64=%g, ngb access=%g, scater=%g, gather=%g, reduction=%g\n",
          NmbTet, F64Tim + NgbTim + TetTim + VerTim + RedTim, F64Tim, NgbTim, TetTim, VerTim, RedTim);

   printf("%zd MB used, %zd MB transfered\n",
          GmlGetMemoryUsage   (GmlIdx) / 1048576,
          GmlGetMemoryTransfer(GmlIdx) / 1048576);

   printf("MidTet checksum = %g, SolAtVer checksum = %g\n",
         TetChk / NmbTet, VerChk / NmbVer);


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(GmlIdx, VerIdx);
   GmlFreeData(GmlIdx, TetIdx);
   GmlFreeData(GmlIdx, MidIdx);
   GmlStop(GmlIdx);

   return(0);
}
