

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.36                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Propagate the surface curvaure to inner P2 edges      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     feb 02 2022                                           */
/*   Last modification: feb 21 2022                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <libmeshb7.h>
#include <gmlib3.h>
#include "parameters.h"
#include "quality.h"


/*----------------------------------------------------------------------------*/
/* This structure definition must be exactly the same as the OpenCL one       */
/*----------------------------------------------------------------------------*/

typedef struct {
   int   foo;
   float res;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* Some very basic kernel launch error checking                               */
/*----------------------------------------------------------------------------*/

int CheckLaunch(int KrnIdx, int res)
{
   if(res < 0)
   {
      printf("Launch kernel %d failled with error: %d\n", KrnIdx, res);
      exit(0);
   }

   return(res);
}


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, perform several smoothing  */
/* steps, compute the resulting mesh quality and save it                      */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int         res, ref, NmbVer, NmbEdg = 0, NmbTet, ver, dim, NmbItr, *EdgNum;
   int         QalIdx, QalKrn, VerIdx, MidIdx, EdgIdx, TetIdx, idx, GpuIdx = 0;
   int         i, j, DbgFlg = 0, (*EdgTab)[3], (*TetTab)[10];
   int         TetEdg[6][2] = { {0,1}, {1,2}, {2,0}, {0,3}, {1,3}, {2,3} };
   size_t      GmlIdx;
   int64_t     InpMsh;
   float       (*CrdTab)[3];
   double      QalTim, RedTim, WalTim, AvgQal, MinQal;
   char        *InpNam;
   GmlParSct   *GmlPar;


   //---------------------------------
   // PARSE THE COMMAND LINE ARGUMENTS
   //---------------------------------

   // If not enough arguments are given, print the help
   if(ArgCnt < 4)
   {
      puts("\np2quality");
      puts(" Usage      : p2quality   p2_mesh_name   GPU_index   NB_loops\n");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
   {
      InpNam = ArgVec[1];
      GpuIdx = atoi(ArgVec[2]);
      NmbItr = atoi(ArgVec[3]);

      // If any fourth argument is given, set on the debug mode
      if(ArgCnt > 4)
         DbgFlg = 1;
   }

   if(!strlen(InpNam))
   {
      puts("No input mesh provided");
      exit(1);
   }

   // Init the GMLIB and compile the OpenCL source code
   if(!(GmlIdx = GmlInit(GpuIdx)))
   {
      printf("Cannot open the GPU %d\n", GpuIdx);
      exit(1);
   }

   if(DbgFlg)
      GmlDebugOn(GmlIdx);


   //---------------------------------
   // ALLOCATE AND READ THE INPUT MESH
   //---------------------------------

   // Check mesh format
   if(!(InpMsh = GmfOpenMesh(InpNam, GmfRead, &ver, &dim)))
   {
      printf("Cannot open mesh %s\n", InpNam);
      exit(1);
   }

   // Get stats and allocate tables
   NmbVer = (int)GmfStatKwd(InpMsh, GmfVertices);
   NmbTet = (int)GmfStatKwd(InpMsh, GmfTetrahedraP2);

   if(!NmbVer || !NmbTet)
   {
      puts("Unsupported kind of mesh");
      exit(1);
   }

   // Allocate mesh memory on the CPU
   CrdTab = malloc( (NmbVer + 1) * 3 * sizeof(float) );
   assert(CrdTab);

   EdgNum = calloc( NmbVer, sizeof(int) );
   assert(EdgNum);

   TetTab = malloc( (NmbTet + 1) * 10 * sizeof(int) );
   assert(TetTab);

   // Read the vertices
   GmfGetBlock(InpMsh, GmfVertices, 1, NmbVer, 0, NULL, NULL,
               GmfFloatVec, 3, CrdTab[1], CrdTab[ NmbVer ],
               GmfInt,         &ref,      &ref);

   // Read the P2 tets
   GmfGetBlock(InpMsh, GmfTetrahedraP2, 1, NmbTet, 0, NULL, NULL,
               GmfIntVec, 10, TetTab[1], TetTab[ NmbTet ],
               GmfInt,       &ref,      &ref);

   GmfCloseMesh(InpMsh);

   printf("\nImported %d vertices and %d tets from the mesh file\n", NmbVer, NmbTet);

   // Make the tet's nodes range from 0 to N - 1 and count the number of edges
   for(i=1;i<=NmbTet;i++)
   {
      for(j=0;j<10;j++)
         TetTab[i][j]--;

      for(j=4;j<10;j++)
         if(!EdgNum[ TetTab[i][j] ])
            EdgNum[ TetTab[i][j] ] = ++NmbEdg;
   }

   // Allocate and fill the table of unique edges
   EdgTab = malloc( (NmbEdg + 1) * 3 * sizeof(int) );
   assert(EdgTab);

   for(i=1;i<=NmbTet;i++)
   {
      for(j=4;j<10;j++)
      {
         if(!(idx = EdgNum[ TetTab[i][j] ]))
            continue;

         EdgTab[ idx ][0] = TetTab[i][ TetEdg[ j-4 ][0] ];
         EdgTab[ idx ][1] = TetTab[i][ TetEdg[ j-4 ][1] ];
         EdgTab[ idx ][2] = TetTab[i][j];
         EdgNum[ TetTab[i][j] ] = 0;
      }
   }

   printf("Extracted %d edges\n", NmbEdg);


   //-------------------------------
   // ALLOCATE AND FILL THE GPU DATA
   //-------------------------------

   // Allocate a common parameters structure to pass along to every kernels
   if(!(GmlPar = GmlNewParameters(GmlIdx, sizeof(GmlParSct), parameters)))
      return(1);

   // Allocate a vertex table on the GPU and fill it with the data from the CPU
   if(!(VerIdx = GmlNewMeshData(GmlIdx, GmlVertices, NmbVer)))
      return(1);

   GmlSetDataBlock(  GmlIdx, GmlVertices, 0, NmbVer-1,
                     CrdTab[1], CrdTab[ NmbVer ], &ref, &ref);

   // Allocate a tets table on the GPU and fill it with the data from the CPU
   if(!(TetIdx = GmlNewMeshData(GmlIdx, GmlTetrahedra, NmbTet)))
      return(1);

   GmlSetDataBlock(  GmlIdx, GmlTetrahedra, 0, NmbTet-1,
                     TetTab[1], TetTab[ NmbTet ], &ref, &ref);

   // Allocate an edges table on the GPU and fill it with the data from the CPU
   if(!(EdgIdx = GmlNewMeshData(GmlIdx, GmlEdges, NmbEdg)))
      return(1);

   GmlSetDataBlock(  GmlIdx, GmlEdges, 0, NmbEdg-1,
                     EdgTab[1], EdgTab[ NmbEdg ], &ref, &ref);

   // Create a raw datatype to store the edges mid point
   if(!(MidIdx = GmlNewSolutionData(GmlIdx, GmlEdges, 1, GmlFlt4, "MidCrd")))
      return(1);

   // Transfer mid edge coordinates
   for(i=1;i<=NmbEdg;i++)
      GmlSetDataLine(GmlIdx, MidIdx, i-1, CrdTab[ EdgTab[i][2] + 1 ]);

   // A vector to store the tets' quality
   if(!(QalIdx = GmlNewSolutionData(GmlIdx, GmlTetrahedra, 1, GmlFlt, "qal")))
      return(1);


   //----------------------------------
   // COMPILE AND LAUNCH THE GPU KERNEL
   //----------------------------------

   // Assemble and compile the tet quality kernel
   QalKrn = GmlCompileKernel( GmlIdx, quality, "quality",
                              GmlTetrahedra, 3,
                              VerIdx, GmlReadMode,  NULL,
                              MidIdx, GmlReadMode,  NULL,
                              QalIdx, GmlWriteMode, NULL );

   if(!QalKrn)
      return(1);

   WalTim = GmlGetWallClock();
   printf("Running %d passes of P2 qualities\n", NmbItr);

   // Launch the tetrahedra quality kernel on the GPU
   for(i=1;i<=NmbItr;i++)
   {
      res = GmlLaunchKernel(GmlIdx, QalKrn);
      CheckLaunch(QalKrn, res);
   }

   // Launch the reduction kernel on the GPU
   res = GmlReduceVector(GmlIdx, QalIdx, GmlSum, &AvgQal);
   CheckLaunch(0, res);

   res = GmlReduceVector(GmlIdx, QalIdx, GmlMin, &MinQal);
   CheckLaunch(0, res);

   printf("P2 tet quality, mean = %g, min = %g\n", AvgQal / NmbTet, MinQal);
   WalTim = GmlGetWallClock() - WalTim;


   //----------------
   // GET THE RESULTS
   //----------------

   QalTim = GmlGetKernelRunTime(GmlIdx, QalKrn);
   RedTim = GmlGetReduceRunTime(GmlIdx, GmlSum);

   printf(  "%lld P2 tets qualities computed in %gs, wall clock: %gs\n",
            (int64_t)NmbItr * (int64_t)NmbTet, RedTim + QalTim, WalTim );

   printf(  "Reduce kernels: %gs\n", RedTim);

   printf("%zd MB used, %zd MB transfered\n\n",
          GmlGetMemoryUsage   (GmlIdx) / 1048576,
          GmlGetMemoryTransfer(GmlIdx) / 1048576);


   //----
   // END
   //----

   GmlFreeData(GmlIdx, QalIdx);
   GmlFreeData(GmlIdx, MidIdx);
   GmlFreeData(GmlIdx, EdgIdx);
   GmlStop(GmlIdx);
   free(CrdTab);
   free(TetTab);

   return(0);
}
