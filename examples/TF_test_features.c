

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.19                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on tetrahedra                              */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     nov 21 2019                                           */
/*   Last modification: mar 30 2020                                           */
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

#include "Parameters.h"
#include "TF_neighbours.h"
#include "TF_downlink.h"
#include "TF_uplink.h"
#include "TF_double_precision.h"


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
   int         i, j, NmbVer=0, NmbTet=0;
   int         ParIdx, VerIdx=0, TetIdx=0, BalIdx, MidIdx, SolIdx, CalMid, OptVer;
   int         GpuIdx = 0, ResIdx, NgbIdx, NgbKrn, F64Idx, F64Krn;
   size_t      GmlIdx;
   float       MidTab[4], SolTab[8], TetChk = 0., VerChk = 0.;
   float       IniSol[8] = {.125, .125, .125, .125, .125, .125, .125, .125};
   double      NgbTim = 0, TetTim = 0, VerTim = 0, RedTim = 0, F64Tim = 0;
   double      res, residual;
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

   GmlImportMesh(GmlIdx, "../sample_meshes/tetrahedra.meshb", GmfVertices, GmfTetrahedra, 0);

   if(!GetMeshInfo(GmlIdx, GmlVertices,   &NmbVer, &VerIdx))
      return(1);

   if(!GetMeshInfo(GmlIdx, GmlTetrahedra, &NmbTet, &TetIdx))
      return(1);

   printf("Imported %d vertices and %d tets from the mesh file\n", NmbVer, NmbTet);

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), Parameters)))
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

   // Allocate a residual vector
   if(!(ResIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "ResVec")))
      return(1);

   if(GmlCheckFP64(GmlIdx))
   {
      puts("This device has FP64 capability.");

      if(!(F64Idx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlDbl, "F64Dat")))
         return(1);

      // Assemble and compile a neighbours kernel
      F64Krn = GmlCompileKernel( GmlIdx, TF_double_precision, "TF_double_precision",
                                 GmlTetrahedra, 1,
                                 F64Idx, GmlReadMode, NULL );

      if(!F64Krn)
         return(1);
   }
   else
      F64Idx = F64Krn = 0;

   // Assemble and compile a neighbours kernel
   NgbKrn = GmlCompileKernel( GmlIdx, TF_neighbours, "TF_neighbours",
                              GmlTetrahedra, 1,
                              MidIdx, GmlReadMode, NgbIdx );

   if(!NgbKrn)
      return(1);

   // Assemble and compile the scatter kernel
   CalMid = GmlCompileKernel( GmlIdx, TF_downlink, "TF_downlink",
                              GmlTetrahedra, 4,
                              VerIdx, GmlReadMode | GmlRefFlag,  NULL,
                              SolIdx, GmlReadMode,  NULL,
                              ResIdx, GmlWriteMode, NULL,
                              MidIdx, GmlWriteMode, NULL );

   if(!CalMid)
      return(1);

   // Assemble and compile the gather kernel
   OptVer = GmlCompileKernel( GmlIdx, TF_uplink, "TF_uplink",
                              GmlVertices, 2,
                              SolIdx, GmlWriteMode, NULL,
                              MidIdx, GmlReadMode | GmlVoyeurs,  NULL );

   if(!OptVer)
      return(1);

   for(i=1;i<=100;i++)
   {
      GmlPar->res = i;

      if(F64Krn)
      {

         // Launch the tetrahedra kernel on the GPU
         res  = GmlLaunchKernel(GmlIdx, F64Krn);

         if(res < 0)
         {
            printf("Launch kernel %d failled with error: %g\n", F64Krn, res);
            exit(0);
         }

         F64Tim += res;
      }

      // Launch the tetrahedra kernel on the GPU
      res  = GmlLaunchKernel(GmlIdx, NgbKrn);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %g\n", NgbKrn, res);
         exit(0);
      }

      NgbTim += res;

      // Launch the tetrahedra kernel on the GPU
      res  = GmlLaunchKernel(GmlIdx, CalMid);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %g\n", CalMid, res);
         exit(0);
      }

      TetTim += res;

      // Launch the vertex kernel on the GPU
      res = GmlLaunchKernel(GmlIdx, OptVer);

      if(res < 0)
      {
         printf("Launch kernel %d failled with error: %g\n", OptVer, res);
         exit(0);
      }

      VerTim += res;

      // Launch the reduction kernel on the GPU
      res = GmlReduceVector(GmlIdx, ResIdx, GmlSum, &residual);

      if(res < 0)
      {
         printf("Launch reduction kernel failled with error: %g\n", res);
         exit(0);
      }

      RedTim += res;
      printf("Iteration: %3d, residual: %g\n", i, residual);
   }


   /*-----------------*/
   /* GET THE RESULTS */
   /*-----------------*/

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

   printf("%ld MB used, %ld MB transfered\n",
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
   GmlFreeData(GmlIdx, ParIdx);
   GmlStop(GmlIdx);

   return(0);
}
