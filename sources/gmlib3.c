

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.01                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: nov 28 2019                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>
#include "gmlib3.h"
#include "reduce.h"


/*----------------------------------------------------------------------------*/
/* Macro instructions                                                         */
/*----------------------------------------------------------------------------*/

#define MIN(a,b)  ((a) < (b) ? (a) : (b))
#define MAX(a,b)  ((a) > (b) ? (a) : (b))
#define MB        1048576


/*----------------------------------------------------------------------------*/
/* Library internal data structures                                           */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int            MshTyp, MemTyp, BasTyp, LnkTyp, DatTyp, RedVecIdx;
   int            NmbCol, ColSiz, VecSiz, NmbLin, LinSiz;
   char           *nam;
   size_t         siz;
   cl_mem         GpuMem;
   void           *CpuMem;
}GmlDatSct;

typedef struct
{
   int            NmbPri, NmbSec, SecSiz, SecPadSiz, VecSiz, NmbExtPri, NmbExtDat;
   int            PriVecIdx, PriExtIdx, ExtDatIdx, PriDegIdx;
}GmlBalSct;

typedef struct
{
   int            idx, siz, DatTab[ GmlMaxDat ];
   cl_kernel      kernel;
   cl_program     program; 
}GmlKrnSct;

typedef struct
{
   int            NmbKrn, CurDev, RedKrnIdx[10], ParIdx;
   int            EleCnt[ GmlMaxTyp ];
   int            TypIdx[ GmlMaxTyp ], LnkMat[ GmlMaxTyp ][ GmlMaxTyp ];
   cl_uint        NmbDev;
   size_t         MemSiz, CurLocSiz, MovSiz;
   GmlDatSct      dat[ GmlMaxDat + 1 ];
   GmlBalSct      bal[ GmlMaxBal + 1 ];
   GmlKrnSct      krn[ GmlMaxKrn + 1 ];
   cl_device_id   device_id[ MaxGpu ];
   cl_context     context;
   cl_command_queue queue;
}GmlSct;


/*----------------------------------------------------------------------------*/
/* Prototypes of local procedures                                             */
/*----------------------------------------------------------------------------*/

static int  GmlNewData              (int, int, int, int, char *);
static int  GmlUploadData           (int);
static int  GmlDownloadData         (int);
static int  GmlNewBall              (int, int);
static int  GmlFreeBall             (int);
static int  GmlUploadBall           (int);
static int  GmlNewKernel            (char *, char *);
static void WriteUserTypedef        (char *, char *);
static void WriteProcedureHeader    (char *, char *, int, int, int *);
static void WriteKernelVariables    (char *, int, int, int *);
static void WriteKernelMemoryReads  (char *, int, int, int *, int *);
static void WriteKernelMemoryWrites (char *, int, int, int *, int *);
static void WriteUserKernel         (char *, char *);
static double OldGmlLaunchKernel    (int, int ,int, ...);


/*----------------------------------------------------------------------------*/
/* Global library variables                                                   */
/*----------------------------------------------------------------------------*/

GmlSct gml;


/*----------------------------------------------------------------------------*/
/* Global tables                                                              */
/*----------------------------------------------------------------------------*/

int  TypSiz[10] = {
   sizeof(cl_int),
   sizeof(cl_int2),
   sizeof(cl_int4),
   sizeof(cl_int8),
   sizeof(cl_int16),
   sizeof(cl_float),
   sizeof(cl_float2),
   sizeof(cl_float4),
   sizeof(cl_float8),
   sizeof(cl_float16) };

char *TypStr[10]  = {
   "int      ",
   "int2     ",
   "int4     ",
   "int8     ",
   "int16    ",
   "float    ",
   "float2   ",
   "float4   ",
   "float8   ",
   "float16  " };

int TypVecSiz[10] = {1,2,4,8,16,1,2,4,8,16};

int MshDatTyp[ GmlMaxTyp ] = {0, 0, 0, GmlFlt4, GmlInt2, GmlInt4,
                              GmlInt4, GmlInt4, GmlInt8, GmlInt8, GmlInt8};
int MshDatVecSiz[ GmlMaxTyp ] = {0, 0, 0, 4, 2, 4, 4, 4, 8, 8, 8};
int MshDatNmbCol[ GmlMaxTyp ] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
char *MshTypStr[ GmlMaxTyp ] = { "", "", "", "Ver", "Edg",
                                 "Tri", "Qad", "Tet", "Pyr", "Pri", "Hex" };

int SizMat[8][8] = {
   {0,16,8,4,32,16,16,8},
   {2,0,2,2,8,8,8,4},
   {4,4,0,0,2,2,2,0},
   {4,4,0,0,0,2,2,2},
   {4,8,4,0,0,0,0,0},
   {8,8,4,1,0,0,0,0},
   {8,16,2,4,0,0,0,0},
   {8,16,0,8,0,0,0,0} };


/*----------------------------------------------------------------------------*/
/* Init device, context and queue                                             */
/*----------------------------------------------------------------------------*/

GmlParSct *GmlInit(int mod)
{
   int            err;
   cl_platform_id platforms[10];
   cl_uint        num_platforms;
   GmlParSct      *par;

   // Select which device to run on
   memset(&gml, 0, sizeof(GmlSct));
   gml.CurDev = mod;

   // Init the GPU
   if(clGetPlatformIDs(10, platforms, &num_platforms) != CL_SUCCESS)
      return(NULL);

   if(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MaxGpu,
                     gml.device_id, &gml.NmbDev) != CL_SUCCESS)
   {
      return(NULL);
   }

   if( (mod < 0) || (mod >= gml.NmbDev) )
      return(NULL);

   if(!(gml.context = clCreateContext( 0, 1, &gml.device_id[ gml.CurDev ],
                                       NULL, NULL, &err)))
   {
      return(NULL);
   }

   if(!(gml.queue = clCreateCommandQueue( gml.context, gml.device_id[ gml.CurDev ],
                                          CL_QUEUE_PROFILING_ENABLE, &err)))
   {
      return(NULL);
   }

   // Allocate and return a public user customizable parameter structure
   if(!(gml.ParIdx = GmlNewParameters(sizeof(GmlParSct), "par")))
      return(NULL);

   return(gml.dat[ gml.ParIdx ].CpuMem);
}


/*----------------------------------------------------------------------------*/
/* Free OpenCL buffers and close the library                                  */
/*----------------------------------------------------------------------------*/

void GmlStop()
{
   int i;

   // Free GPU memories, kernels and queue
   for(i=1;i<=GmlMaxDat;i++)
      if(gml.dat[i].GpuMem)
         clReleaseMemObject(gml.dat[i].GpuMem);

   for(i=1;i<=gml.NmbKrn;i++)
   {
      clReleaseKernel(gml.krn[i].kernel);
      clReleaseProgram(gml.krn[i].program);
   }

   clReleaseCommandQueue(gml.queue); 
   clReleaseContext(gml.context);
}


/*----------------------------------------------------------------------------*/
/* Print all available OpenCL capable GPUs                                    */
/*----------------------------------------------------------------------------*/

void GmlListGPU()
{
   int            i;
   size_t         GpuNamSiz;
   char           GpuNam[100];
   cl_platform_id platforms[10];
   cl_device_id   device_id[ MaxGpu ];
   cl_uint        num_platforms, num_devices;

   if(clGetPlatformIDs(10, platforms, &num_platforms) != CL_SUCCESS)
      return;

   if(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MaxGpu,
                     device_id, &num_devices) != CL_SUCCESS)
   {
      return;
   }

   for(i=0;i<num_devices;i++)
      if(clGetDeviceInfo(  device_id[i], CL_DEVICE_NAME, 100, GpuNam,
                           &GpuNamSiz) == CL_SUCCESS )
      {
         printf("      %d      : %s\n", i, GpuNam);
      }
}


/*----------------------------------------------------------------------------*/
/* Allocate one of the 8 mesh data types                                      */
/*----------------------------------------------------------------------------*/

int GmlNewParameters(int siz, char *nam)
{
   return(GmlNewData(GmlParameters, 0, 0, siz, nam));
}


/*----------------------------------------------------------------------------*/
/* Allocate one of the 8 mesh data types                                      */
/*----------------------------------------------------------------------------*/

int GmlNewMeshData(int MshTyp, int NmbLin)
{
   if( (MshTyp < GmlVertices) || (MshTyp > GmlHexahedra) )
      return(0);

   return(GmlNewData(MshTyp, NmbLin, 0, 0, MshTypStr[ MshTyp ]));
}


/*----------------------------------------------------------------------------*/
/* Allocate a free solution/raw data associated with a mesh data type         */
/*----------------------------------------------------------------------------*/

int GmlNewSolutionData(int MshTyp, int NmbDat, int DatTyp, char *nam)
{
   if( (MshTyp < GmlVertices) || (MshTyp > GmlHexahedra) )
      return(0);

   if( (DatTyp < GmlInt) || (DatTyp > GmlFlt16) )
      return(0);

   return(GmlNewData(GmlRawData, MshTyp, NmbDat, DatTyp, nam));
}


/*----------------------------------------------------------------------------*/
/* Create an arbitrary link table between two kinds of mesh data types        */
/*----------------------------------------------------------------------------*/

int GmlNewLinkData(int BasTyp, int LnkTyp, int NmbDat, char *nam)
{
   if( (BasTyp < GmlVertices) || (BasTyp > GmlHexahedra) )
      return(0);

   if( (LnkTyp < GmlVertices) || (LnkTyp > GmlHexahedra) )
      return(0);

   return(GmlNewData(GmlLnkData, BasTyp, LnkTyp, NmbDat, nam));
}


/*----------------------------------------------------------------------------*/
/* Allocate an OpenCL buffer plus 10% more for resizing                       */
/*----------------------------------------------------------------------------*/

static int GmlNewData(int typ, int par1, int par2, int par3, char *nam)
{
   int         idx;
   GmlDatSct   *dat;

   if( (typ < GmlParameters)  || (typ > GmlHexahedra) )
      return(0);

   // Look for a free data socket
   for(idx=1;idx<=GmlMaxDat;idx++)
      if(!gml.dat[ idx ].GpuMem)
         break;

   if(idx > GmlMaxDat)
      return(0);

   dat = &gml.dat[ idx ];

   if(typ == GmlParameters)
   {
      dat->MshTyp = GmlParameters;
      dat->BasTyp = 0;
      dat->LnkTyp = 0;
      dat->MemTyp = GmlInout;
      dat->DatTyp = 0;
      dat->NmbCol = 1;
      dat->ColSiz = par3;
      dat->VecSiz = 0;
      dat->NmbLin = 1;
      dat->LinSiz = dat->NmbCol * dat->ColSiz;
      dat->nam    = nam;
   }
   else if(typ == GmlRawData)
   {
      dat->MshTyp = GmlRawData;
      dat->BasTyp = par1;
      dat->LnkTyp = 0;
      dat->MemTyp = GmlInout;
      dat->DatTyp = par3;
      dat->NmbCol = par2;
      dat->ColSiz = TypSiz[ par3 ];
      dat->VecSiz = TypVecSiz[ par3 ];
      dat->NmbLin = gml.EleCnt[ par1 ];
      dat->LinSiz = dat->NmbCol * dat->ColSiz;
      dat->nam    = nam;
   }
   else if(typ == GmlLnkData)
   {
      dat->MshTyp = GmlLnkData;
      dat->BasTyp = par1;
      dat->LnkTyp = par2;
      dat->MemTyp = GmlInout;
      dat->DatTyp = GmlInt;
      dat->NmbCol = par3;
      dat->ColSiz = TypSiz[ GmlInt ];
      dat->VecSiz = 1;
      dat->NmbLin = gml.EleCnt[ par1 ];
      dat->LinSiz = dat->NmbCol * dat->ColSiz;
      dat->nam    = nam;
   }
   else
   {
      dat->MshTyp = typ;
      dat->BasTyp = typ;
      dat->LnkTyp = 0;
      dat->MemTyp = GmlInout;
      dat->DatTyp = MshDatTyp[ typ ];
      dat->NmbCol = 1;
      dat->ColSiz = TypSiz[ typ ];
      dat->VecSiz = TypVecSiz[ typ ];
      dat->NmbLin = par1;
      dat->LinSiz = dat->ColSiz;
      dat->nam    = nam;
      gml.EleCnt[ typ ] = par1;
      gml.TypIdx[ typ ] = idx;
      gml.LnkMat[ typ ][ GmlVertices ] = idx;
   }

   dat->siz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;

   // Allocate the requested memory size on the GPU
   if(dat->MemTyp == GmlInput)
   {
      dat->GpuMem = clCreateBuffer( gml.context, CL_MEM_READ_ONLY,
                                    dat->siz, NULL, NULL );
   }
   else if(dat->MemTyp == GmlOutput)
   {
      dat->GpuMem = clCreateBuffer( gml.context, CL_MEM_WRITE_ONLY,
                                    dat->siz, NULL, NULL );
   }
   else if((dat->MemTyp == GmlInout) || (dat->MemTyp == GmlInternal))
   {
      dat->GpuMem = clCreateBuffer( gml.context, CL_MEM_READ_WRITE,
                                    dat->siz, NULL, NULL );
   }

   if(!dat->GpuMem)
   {
      printf("Cannot allocate %ld MB on the GPU (%ld MB already used)\n",
               dat->siz/MB, GmlGetMemoryUsage()/MB);
      return(0);
   }

   // Allocate the requested memory size on the CPU side
   if( (dat->MemTyp != GmlInternal) && !(dat->CpuMem = calloc(1, dat->siz)) )
      return(0);

   // Keep track of allocated memory
   gml.MemSiz += dat->siz;

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Release an OpenCL buffer                                                   */
/*----------------------------------------------------------------------------*/

int GmlFreeData(int idx)
{
   GmlDatSct *dat = &gml.dat[ idx ];

   // Free both GPU and CPU memory buffers
   if( (idx >= 1) && (idx <= GmlMaxDat) && dat->GpuMem )
   {
      if(clReleaseMemObject(dat->GpuMem) != CL_SUCCESS)
         return(0);

      dat->GpuMem = NULL;
      gml.MemSiz -= dat->siz;

      if(dat->CpuMem)
         free(dat->CpuMem);

      if(dat->RedVecIdx)
      {
         GmlFreeData(dat->RedVecIdx);
         dat->RedVecIdx = 0;
      }

      return(1);
   }
   else
      return(0);
}


/*----------------------------------------------------------------------------*/
/* Set a line of mesh, link or solution data                                  */
/*----------------------------------------------------------------------------*/

int GmlSetDataLine(int idx, int lin, ...)
{
   GmlDatSct   *dat = &gml.dat[ idx ];
   char        *adr = (void *)dat->CpuMem;
   int         i, *tab, dim = 3, siz, ref;
   int         EleNmbInt[ GmlMaxTyp ] = {0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 8};
   float       *crd;
   va_list     VarArg;

   va_start(VarArg, lin);

   if(dat->MshTyp == GmlRawData)
   {
      memcpy(&adr[ lin * dat->LinSiz ], va_arg(VarArg, void *), dat->LinSiz);
   }
   else if(dat->MshTyp == GmlLnkData)
   {
      tab = (int *)dat->CpuMem;

      for(i=0;i<dat->NmbCol;i++)
         tab[ lin * dat->NmbCol + i ] = va_arg(VarArg, int);
   }
   else if(dat->MshTyp == GmlVertices)
   {
      crd = (float *)dat->CpuMem;

      if(dim == 2)
         siz = 2;
      else
         siz = 4;

      for(i=0;i<dim;i++)
         crd[ lin * siz + i ] = va_arg(VarArg, double);

      ref = va_arg(VarArg, int);
      //crd[ lin*dat->LinSiz + dim ] = va_arg(VarArg, int);
   }
   else
   {
      tab = (int *)dat->CpuMem;
      siz = TypVecSiz[ dat->MshTyp ];

      for(i=0;i<EleNmbInt[ dat->MshTyp ];i++)
         tab[ lin * siz + i ] = va_arg(VarArg, int);
   }

   va_end(VarArg);

   if(lin == dat->NmbLin -1)
      gml.MovSiz += GmlUploadData(idx);

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Get a line of solution or vertex corrdinates from the librarie's buffers   */
/*----------------------------------------------------------------------------*/

int GmlGetDataLine(int idx, int lin, ...)
{
   GmlDatSct   *dat = &gml.dat[ idx ];
   char        *adr = (void *)dat->CpuMem;
   int         i, *tab, dim = 2;
   float       *GpuCrd;
   double      *UsrCrd;
   va_list     VarArg;

   if(lin == 0)
      gml.MovSiz += GmlDownloadData(idx);

   va_start(VarArg, lin);

   if(dat->MshTyp == GmlRawData)
   {
      memcpy(va_arg(VarArg, void *), &adr[ lin * dat->LinSiz ], dat->LinSiz);
   }
   else if(dat->MshTyp == GmlVertices)
   {
      GpuCrd = (float *)dat->CpuMem;

      for(i=0;i<dim;i++)
      {
         UsrCrd = va_arg(VarArg, double *);
         *UsrCrd = (double)GpuCrd[ lin*dat->LinSiz + i ];
      }
   }

   va_end(VarArg);

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Copy user's data into an OpenCL buffer                                     */
/*----------------------------------------------------------------------------*/

static int GmlUploadData(int idx)
{
   GmlDatSct *dat = &gml.dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem
   || !dat->CpuMem || (dat->MemTyp == GmlOutput) )
   {
      return(0);
   }

   // Upload buffer from CPU ram to GPU ram
   // and keep track of the amount of uploaded data
   if(clEnqueueWriteBuffer(gml.queue, dat->GpuMem, CL_FALSE, 0, dat->siz,
                           dat->CpuMem, 0, NULL,NULL) != CL_SUCCESS)
   {
      return(0);
   }
   else
   {
      gml.MovSiz += dat->siz;
      return(dat->siz);
   }
}


/*----------------------------------------------------------------------------*/
/* Copy an OpenCL buffer into user's data                                     */
/*----------------------------------------------------------------------------*/

static int GmlDownloadData(int idx)
{
   GmlDatSct *dat = &gml.dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || !dat->CpuMem )
      return(0);

   // Download buffer from GPU ram to CPU ram
   // and keep track of the amount of downloaded data
   if(clEnqueueReadBuffer( gml.queue, dat->GpuMem, CL_TRUE, 0,
                           dat->siz, dat->CpuMem, 0, NULL, NULL ) != CL_SUCCESS)
   {
      return(0);
   }
   else
   {
      gml.MovSiz += dat->siz;
      return(dat->siz);
   }
}


/*----------------------------------------------------------------------------*/
/* Generate the kernel from user's source and data and compile it             */
/*----------------------------------------------------------------------------*/

int GmlCompileKernel(char *KrnSrc, char *PrcNam, char *DefSrc,
                     int BasTyp, int NmbTyp, ...)
{
   int      KrnIdx, i, IdxTab[ GmlMaxDat ], TypTab[ GmlMaxDat ];
   int      FlgTab[ GmlMaxDat ], LnkTab[ GmlMaxDat ];
   char     src[10000] = "\0";
   va_list  VarArg;
   GmlDatSct *dat;

   // Decode datatypes arguments
   va_start(VarArg, NmbTyp);

   for(i=0;i<NmbTyp;i++)
   {
      IdxTab[i] = va_arg(VarArg, int);
      FlgTab[i] = va_arg(VarArg, int);
      LnkTab[i] = va_arg(VarArg, int);
   }

   va_end(VarArg);

   src[0] = 0;

   //Parse arguments and add the required topological links
   for(i=0;i<NmbTyp;i++)
   {
      dat = &gml.dat[ IdxTab[i] ];

      if( (dat->MshTyp >= GmlVertices) && (FlgTab[i] & GmlRefFlag) )
      {
         // Add a ref reading information
      }

      if(!LnkTab[i])
      {
         if(!gml.LnkMat[ BasTyp ][ dat->BasTyp ])
         {
            // Generate the default link between the two kinds of entities
         }
      }
   }

   // Generate the kernel source code
   WriteUserTypedef        (src, DefSrc);
   WriteProcedureHeader    (src, PrcNam, BasTyp, NmbTyp, IdxTab);
   WriteKernelVariables    (src, BasTyp, NmbTyp, IdxTab);
   WriteKernelMemoryReads  (src, BasTyp, NmbTyp, IdxTab, FlgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, BasTyp, NmbTyp, IdxTab, FlgTab);

   // And Compile it
   KrnIdx = GmlNewKernel   (src, PrcNam);

   puts(src);
   exit(0);

   return(KrnIdx);
}


/*----------------------------------------------------------------------------*/
/* Add the user's parameters structure definition to the source code          */
/*----------------------------------------------------------------------------*/

static void WriteUserTypedef(char *src, char *TypSct)
{
   strcat(src, "// USER'S ARGUMENTS STRUCTURE\n");
   strcat(src, TypSct);
   strcat(src, "\n");
}


/*----------------------------------------------------------------------------*/
/* Write the procedure name and typedef with all arguments types and names    */
/*----------------------------------------------------------------------------*/

static void WriteProcedureHeader(char *src, char *PrcNam,
                                 int BasTyp, int NmbTyp, int *IdxTab)
{
   int   i;
   char  str[100];

   strcat(src, "// KERNEL HEADER\n");
   sprintf(str, "__kernel void %s(", PrcNam);
   strcat(src, str);

   for(i=0;i<NmbTyp;i++)
   {
      GmlDatSct *dat = &gml.dat[ IdxTab[i] ];

      sprintf(str,  "\n   __global %s ", TypStr[ dat->DatTyp ]);
      strcat(src, str);

      if(dat->NmbCol <= 1)
         sprintf(str, "*%sTab,", dat->nam);
      else
         sprintf(str, "(*%sTab)[%d],", dat->nam, dat->NmbCol);

      strcat(src, str);
   }

   strcat(src, "\n   __global GmlParSct *par,");
   strcat(src, "\n   const    int        count )\n{\n");
}


/*----------------------------------------------------------------------------*/
/* Write definition of automatic local variables                              */
/*----------------------------------------------------------------------------*/

static void WriteKernelVariables(char *src, int BasTyp, int NmbTyp, int *IdxTab)
{
   int   i, siz;
   char  str[100];

   strcat(src, "// KERNEL VARIABLES DEFINITION\n");

   for(i=0;i<NmbTyp;i++)
   {
      GmlDatSct *dat = &gml.dat[ IdxTab[i] ];
      siz = SizMat[ BasTyp - 3 ][ dat->BasTyp - 3 ];

      if(siz > 1)
      {
         if(dat->NmbCol > 1)
            sprintf(str,  "   %s %s[%d][%d];\n", TypStr[ dat->DatTyp ], dat->nam, siz, dat->NmbCol);
         else
            sprintf(str,  "   %s %s[%d];\n", TypStr[ dat->DatTyp ], dat->nam, siz);
      }
      else
      {
         if(dat->NmbCol > 1)
            sprintf(str,  "   %s %s[%d];\n", TypStr[ dat->DatTyp ], dat->nam, dat->NmbCol);
         else
            sprintf(str,  "   %s %s;\n", TypStr[ dat->DatTyp ], dat->nam);
      }

      strcat(src, str);
   }
}


/*----------------------------------------------------------------------------*/
/* Write the memory reading from the global structure to the local variables  */
/*----------------------------------------------------------------------------*/

static void WriteKernelMemoryReads( char *src, int BasTyp, int NmbTyp,
                                    int *IdxTab, int *FlgTab )
{
   int   i, j, c, siz;
   char  str[100];

   strcat(src, "   int cnt = get_global_id(0);\n\n");
   strcat(src, "   if(cnt >= count)\n      return;\n\n");
   strcat(src, "// KERNEL MEMORY READINGS\n");

   for(i=0;i<NmbTyp;i++)
   {
      if(!(FlgTab[i] & GmlReadMode))
         continue;

      GmlDatSct *dat = &gml.dat[ IdxTab[i] ];
      siz = SizMat[ BasTyp - 3 ][ dat->BasTyp - 3 ];

      if(dat->NmbCol == 1)
      {
         if(siz <= 1)
         {
            sprintf( str, "   %s = %sTab[ cnt ];\n",
                     dat->nam, MshTypStr[ BasTyp ]);
            strcat(src, str);
         }
         else
         {
            for(j=0;j<siz;j++)
            {
               sprintf( str, "   %s[%d] = %sTab[ %s.s%d ];\n",
                        dat->nam, j, dat->nam, MshTypStr[ BasTyp ], j);
               strcat(src, str);
            }
         }
      }
      else
      {
         for(c=0;c<dat->NmbCol;c++)
         {
            if(siz <= 1)
            {
               sprintf( str, "   %s[%d] = %sTab[ cnt ][%d];\n",
                        dat->nam, c, MshTypStr[ BasTyp ], c);
               strcat(src, str);
            }
            else
            {
               for(j=0;j<siz;j++)
               {
                  sprintf( str, "   %s[%d][%d] = %sTab[ %s.s%d ][%d];\n",
                           dat->nam, j, c, dat->nam, MshTypStr[ BasTyp ], j, c);
                  strcat(src, str);
               }
            }
         }
      }

      strcat(src, "\n");
   }
}


/*----------------------------------------------------------------------------*/
/* Pour the user's kernel code as it is                                       */
/*----------------------------------------------------------------------------*/

static void WriteUserKernel(char *src, char *KrnSrc)
{
   strcat(src, "// USER'S KERNEL CODE\n");
   strcat                  (src, KrnSrc);
}


/*----------------------------------------------------------------------------*/
/* Write the final storage of requested variables from local to global struct */
/*----------------------------------------------------------------------------*/

static void WriteKernelMemoryWrites(char *src, int BasTyp, int NmbTyp,
                                    int *IdxTab, int *FlgTab)
{
   int   i, j, c, siz;
   char  str[100];

   strcat(src, "\n");
   strcat(src, "// KERNEL MEMORY WRITINGS\n");

   for(i=0;i<NmbTyp;i++)
   {
      if(!(FlgTab[i] & GmlWriteMode))
         continue;

      GmlDatSct *dat = &gml.dat[ IdxTab[i] ];
      siz = SizMat[ BasTyp - 3 ][ dat->BasTyp - 3 ];

      if(dat->NmbCol == 1)
      {
         sprintf( str, "   %sTab[ cnt ] = %s;\n",
                  dat->nam, dat->nam );
         strcat(src, str);
      }
      else
      {
         for(c=0;c<dat->NmbCol;c++)
         {
            sprintf( str, "   %sTab[ cnt ][%d] = %s[%d];\n",
                     dat->nam, c, dat->nam, c );
            strcat(src, str);
         }
      }
   }

   strcat(src, "}\n");
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

double GmlLaunchKernel(int KrnIdx, int cnt)
{
   return(0.);
}


/*----------------------------------------------------------------------------*/
/* Read and compile an OpenCL source code                                     */
/*----------------------------------------------------------------------------*/

static int GmlNewKernel(char *KernelSource, char *PrcNam)
{
   char        *buffer, *StrTab[1];
   int         err, idx = ++gml.NmbKrn;
   GmlKrnSct   *krn = &gml.krn[ idx ];
   size_t      len, LenTab[1];

   if(idx > GmlMaxKrn)
      return(0);

   StrTab[0] = KernelSource;
   LenTab[0] = strlen(KernelSource)-1;

   // Compile source code
   if(!(krn->program = clCreateProgramWithSource(gml.context, 1, (const char **)StrTab,
                                                (const size_t *)LenTab, &err)))
   {
      return(0);
   }

   if(clBuildProgram(krn->program, 0, NULL,
      "-cl-single-precision-constant -cl-mad-enable", NULL, NULL) != CL_SUCCESS)
   {
      clGetProgramBuildInfo(  krn->program, gml.device_id[ gml.CurDev ],
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

      if(!(buffer = malloc(len)))
         return(0);

      clGetProgramBuildInfo(  krn->program, gml.device_id[ gml.CurDev ],
                              CL_PROGRAM_BUILD_LOG, len, buffer, &len);
      printf("%s\n", buffer);
      free(buffer);
      return(0);
   }

   if(!(krn->kernel = clCreateKernel(krn->program, PrcNam, &err))
   || (err != CL_SUCCESS))
   {
      return(0);
   }

//   clGetProgramInfo(krn->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &len, NULL);
//      printf("\ncompilation of %s : %ld bytes\n",PrcNam,len);

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

double OldGmlLaunchKernel(int idx, int TruSiz, int NmbDat, ...)
{
   int i,      DatTab[ GmlMaxDat ];
   size_t      GloSiz, LocSiz, RetSiz = 0;
   va_list     VarArg;
   GmlDatSct   *dat;
   GmlKrnSct   *krn = &gml.krn[ idx ];
   cl_event    event;
   cl_ulong    start, end;

   if( (idx < 1) || (idx > gml.NmbKrn) || !krn->kernel )
      return(-1);

   // First send the parameters to the GPU memmory
   if(!GmlUploadData(gml.ParIdx))
      return(-2);

   // Build arguments list
   va_start(VarArg, NmbDat);

   for(i=0;i<NmbDat;i++)
      DatTab[i] = va_arg(VarArg, int);

   va_end(VarArg);

   for(i=0;i<NmbDat;i++)
   {
      dat = &gml.dat[ DatTab[i] ];

      if( (DatTab[i] < 1) || (DatTab[i] > GmlMaxDat) || !dat->GpuMem
      || (clSetKernelArg(krn->kernel, i, sizeof(cl_mem), &dat->GpuMem) != CL_SUCCESS) )
      {
         printf("i=%d, DatTab[i]=%d, GpuMem=%p\n",i,DatTab[i],dat->GpuMem);
         return(-3);
      }
   }

   if(clSetKernelArg(krn->kernel, NmbDat, sizeof(cl_mem),
                     &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
   {
      return(-4);
   }

   if(clSetKernelArg(krn->kernel, NmbDat+1, sizeof(int), &TruSiz) != CL_SUCCESS)
      return(-5);

   // Fit data loop size to the GPU kernel size
   if(clGetKernelWorkGroupInfo(  krn->kernel, gml.device_id[ gml.CurDev ],
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                 &LocSiz, &RetSiz) != CL_SUCCESS )
   {
      return(-6);
   }

   gml.CurLocSiz = LocSiz;
   GloSiz = TruSiz / LocSiz;
   GloSiz *= LocSiz;

   if(GloSiz < TruSiz)
      GloSiz += LocSiz;

   // Launch GPU code
   clFinish(gml.queue);

   if(clEnqueueNDRangeKernel( gml.queue, krn->kernel, 1, NULL,
                              &GloSiz, &LocSiz, 0, NULL, &event) )
   {
      return(-7);
   }

   clFinish(gml.queue);

   if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL) != CL_SUCCESS)
   {
      return(-8);
   }

   if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, NULL) != CL_SUCCESS)
   {
      return(-9);
   }

   // Finaly, get back the parameters from the GPU memmory
   if(!GmlDownloadData(gml.ParIdx))
      return(-10);

   return((double)(end - start) / 1e9);
}


/*----------------------------------------------------------------------------*/
/* Return memory currently allocated on the GPU                               */
/*----------------------------------------------------------------------------*/

size_t GmlGetMemoryUsage()
{
   return(gml.MemSiz);
}


/*----------------------------------------------------------------------------*/
/* Return memory currently transfered to or from the GPU                      */
/*----------------------------------------------------------------------------*/

size_t GmlGetMemoryTransfer()
{
   return(gml.MovSiz);
}


/*----------------------------------------------------------------------------*/
/* Return public parameter structure                                          */
/*----------------------------------------------------------------------------*/

GmlParSct *GmlGetParameters()
{
   return(gml.dat[ gml.ParIdx ].CpuMem);
}
