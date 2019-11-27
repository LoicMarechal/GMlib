

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.00                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Basic loop on tetrahedra                              */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     nov 21 2019                                           */
/*   Last modification: nov 27 2019                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libmeshb7.h"
#include "gmlib3.h"

char *Arguments = "\
   typedef struct {\n\
      int   wei;\n\
      float MinQal, rlx;\n\
   }GmlParSct;\n";

char *TetrahedraBasicLoop = "\
   float4 left, right;\n\
   left  = Sol[0][0] + Sol[1][0] + Sol[2][0] + Sol[3][0];\n\
   right = Sol[0][1] + Sol[1][1] + Sol[2][1] + Sol[3][1];\n\
   Mid   = (Ver[0] + Ver[1] + Ver[2] + Ver[3]) * (left + right) * (float4)(.25, .25, .25, 0.);\n";


/*----------------------------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute                    */
/* the elements middle and get back the results.                              */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   int i, NmbVer, NmbTet, ver=0, dim=0, ref, (*TetTab)[4]=NULL;
   int VerIdx, TetIdx, MidIdx, SolIdx, CalMid, GpuIdx=0;
   int64_t InpMsh;
   float (*VerTab)[3]=NULL, (*MidTab)[4], dummy, chk=0.;
   double GpuTim;


   // If no arguments are give, print the help
   if(ArgCnt == 1)
   {
      puts("\nTetrahedraBasicLoop GPU_index");
      puts(" Choose GPU_index from the following list:");
      GmlListGPU();
      exit(0);
   }
   else
      GpuIdx = atoi(ArgVec[1]);


   /*--------------*/
   /* MESH READING */
   /*--------------*/

   // Open the mesh
   if( !(InpMsh = GmfOpenMesh("tetrahedra.meshb", GmfRead, &ver, &dim))
   || (ver != GmfFloat) || (dim != 3) )
   {
      puts("Could not open tetrahedra.meshb");
      puts("Please run the command in the same directory this mesh is located");
      return(1);
   }

   // Read the number of vertices and elements and allocate the memory
   if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) \
   || !(VerTab = malloc((NmbVer+1) * 3 * sizeof(float))) )
   {
      return(1);
   }

   if( !(NmbTet = GmfStatKwd(InpMsh, GmfTetrahedra)) \
   || !(TetTab = malloc((NmbTet+1) * 4 * sizeof(int))) )
   {
      return(1);
   }

   // Read the vertices
   GmfGotoKwd(InpMsh, GmfVertices);
   for(i=1;i<=NmbVer;i++)
      GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

   // Read the elements
   GmfGotoKwd(InpMsh, GmfTetrahedra);
   for(i=1;i<=NmbTet;i++)
      GmfGetLin(  InpMsh, GmfTetrahedra, &TetTab[i][0], &TetTab[i][1], \
                  &TetTab[i][2], &TetTab[i][3], &ref );

   // And close the mesh
   GmfCloseMesh(InpMsh);


   /*---------------*/
   /* GPU COMPUTING */
   /*---------------*/

   // Init the GMLIB and compile the OpenCL source code
   if(!GmlInit(GpuIdx))
      return(1);

   // Create a vertices data type and transfer the data to the GPU
   if(!(VerIdx = GmlNewMeshData(GmlVertices, NmbVer)))
      return(1);

   for(i=1;i<=NmbVer;i++)
      GmlSetDataLine(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2], 0);

   // Do the same with the elements
   if(!(TetIdx = GmlNewMeshData(GmlTetrahedra, NmbTet)))
      return(1);

   for(i=1;i<=NmbTet;i++)
      GmlSetDataLine(TetIdx, i-1,
                     TetTab[i][0]-1, TetTab[i][1]-1,
                     TetTab[i][2]-1, TetTab[i][3]-1, 0);

   // Create a raw datatype to store some value at vertices.
   if(!(SolIdx = GmlNewSolutionData(GmlVertices, 2, GmlFlt4, "Sol")))
      return(1);

   // Create a raw datatype to store the element middles.
   // It does not need to be tranfered to the GPU
   if(!(MidIdx = GmlNewSolutionData(GmlTetrahedra, 1, GmlFlt4, "Mid")))
      return(1);

   if(!(MidTab = malloc((NmbTet+1)*4*sizeof(float))))
      return(1);

   CalMid = GmlCompileKernel( TetrahedraBasicLoop, "TetrahedraBasic",
                              Arguments, GmlTetrahedra, 4,
                              TetIdx, GmlReadMode,  NULL,
                              VerIdx, GmlReadMode,  NULL,
                              SolIdx, GmlReadMode,  NULL,
                              MidIdx, GmlWriteMode, NULL );

   if(!CalMid)
      return(1);

   // Launch the kernel on the GPU
   GpuTim = GmlLaunchKernel(CalMid, NmbTet);

   if(GpuTim < 0)
      return(1);

   for(i=1;i<=NmbTet;i++)
   {
      GmlGetDataLine(MidIdx, i-1, MidTab[i]);
      chk += sqrt(MidTab[i][0] * MidTab[i][0] \
               +  MidTab[i][1] * MidTab[i][1] \
               +  MidTab[i][2] * MidTab[i][2]);
   }

   printf("%d tet centers computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
      NmbTet, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbTet );


   /*-----*/
   /* END */
   /*-----*/

   GmlFreeData(VerIdx);
   GmlFreeData(TetIdx);
   GmlFreeData(MidIdx);
   GmlStop();

   free(MidTab);
   free(TetTab);
   free(VerTab);

   return(0);
}