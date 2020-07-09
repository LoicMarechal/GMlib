

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         MESHB TO METIS V 1.0                               */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Description:         convert a .meshb file into a metis mesh               */
/* Author:              Loic MARECHAL                                         */
/* Creation date:       jun 19 2020                                           */
/* Last modification:   jun 19 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libmeshb7.h"


/*----------------------------------------------------------------------------*/
/* Read the meshb with libMeshb and write the metis ASCII mesh                */
/*----------------------------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
   char     *PtrArg, *TmpStr, InpNam[1000], OutNam[1000];
   int      i, NmbTet, MshVer, dim, v1, v2, v3, v4, ref;
   int64_t  InpMsh;
   FILE     *OutMsh;

   // Command line parsing
   if(ArgCnt == 1)
   {
      puts("\nMESHB2METIS v1.0 june 19 2020   Loic MARECHAL / INRIA");
      puts(" Usage       : meshb2metis -in input_meshb -out metis_mesh");
      puts(" -in name    : name of the input Gamma mesh(b) format");
      puts(" -out name   : name of the output Metis mesh format\n");
      exit(0);
   }

   for(i=2;i<=ArgCnt;i++)
   {
      PtrArg = *++ArgVec;

      if(!strcmp(PtrArg,"-in"))
      {
         TmpStr = *++ArgVec;
         ArgCnt--;
         strcpy(InpNam, TmpStr);

         if(!strstr(InpNam, ".mesh"))
            strcat(InpNam, ".meshb");

         continue;
      }

      if(!strcmp(PtrArg,"-out"))
      {
         TmpStr = *++ArgVec;
         ArgCnt--;
         strcpy(OutNam, TmpStr);

         if(!strstr(OutNam, ".mesh"))
            strcat(OutNam, "_metis.mesh");

         continue;
      }
   }

   if(!strlen(InpNam))
   {
      puts("No input mesh provided");
      exit(1);
   }

   if(!strlen(OutNam))
   {
      puts("No output name provided");
      exit(1);
   }

   if(!(InpMsh = GmfOpenMesh(InpNam, GmfRead, &MshVer, &dim)))
   {
      printf("Cannot open mesh %s\n", InpNam);
      exit(1);
   }

   if(!(OutMsh = fopen(OutNam, "w")))
   {
      printf("Cannot create output mesh %s\n", OutNam);
      exit(1);
   }

   NmbTet = GmfStatKwd(InpMsh, GmfTetrahedra);

   if(NmbTet)
   {
      GmfGotoKwd(InpMsh, GmfTetrahedra);
      fprintf(OutMsh, "%d\n", NmbTet);

      for(i=1;i<=NmbTet;i++)
      {
         GmfGetLin(InpMsh, GmfTetrahedra, &v1, &v2, &v3, &v4, &ref);
         fprintf(OutMsh, "%d %d %d %d\n", v1, v2, v3, v4);
      }
   }

   GmfCloseMesh(InpMsh);
   fclose(OutMsh);
}
