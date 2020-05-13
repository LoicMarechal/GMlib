
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int ArgCnt, char **ArgVec)
{
   char c, *InpNam, *OutNam, *PrcNam;
   int i;
   FILE *InpHdl, *OutHdl;

   if(ArgCnt != 4)
   {
      puts("Usage : cl2h source.cl destination.h procedure");
      exit(1);
   }

   InpNam = *++ArgVec;
   OutNam = *++ArgVec;
   PrcNam = *++ArgVec;

   if(!(InpHdl = fopen(InpNam, "r")))
   {
      printf("Cannot open %s for reading\n", InpNam);
      exit(1);
   }

   if(!(OutHdl = fopen(OutNam, "w")))
   {
      printf("Cannot open %s for writing\n", OutNam);
      exit(1);
   }

   fprintf(OutHdl, "char *%s = \"\\n\" \\\n", PrcNam);

   do
   {
      fprintf(OutHdl, "\"");

      do
      {
         c = fgetc(InpHdl);

         if(c == '\\')
         do
         {
            c = fgetc(InpHdl);
         }while(c == '\n');
         
         if((c == EOF) || (c == '\n'))
            break;

         fputc(c, OutHdl);
      }while((c != EOF) && (c != '\n'));
      fprintf(OutHdl, "\\n\" \\\n");
   }while(c != EOF);

   fprintf(OutHdl, "\n;\n");

   fclose(InpHdl);
   fclose(OutHdl);

   exit(0);
}
