

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.00                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCl                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: jul 12 2018                                           */
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

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MB 1048576


/*----------------------------------------------------------------------------*/
/* Library internal data structures                                           */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int      MshTyp, RefTyp, NmbLin, NmbCol, LinSiz, RedVecIdx, UplFlg, DwlFlg;
   char     *NamStr;
   size_t   siz;
   cl_mem   GpuMem;
   void     *CpuMem;
   char     *TypStr;
}GmlDatSct;

typedef struct
{
   int   NmbPri, NmbSec, SecSiz, SecPadSiz, VecSiz, NmbExtPri, NmbExtDat;
   int   PriVecIdx, PriExtIdx, ExtDatIdx, PriDegIdx;
}GmlBalSct;

typedef struct
{
   int         idx, siz, DatTab[ GmlMaxDat ];
   char        *KrnSrc, PrcNam[10];
   cl_kernel   kernel;
   cl_program  program; 
}GmlKrnSct;

typedef struct
{
   int            NmbKrn, CurDev, RedKrnIdx[10], ParIdx;
   int            IdxTab[ GmlEnd ];
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

static int     GmlUploadData     (int);
static int     GmlDownloadData   (int);
static int     GmlCompileKernel  (int, char *, char *);
static double  GmlOpenClLaunch   (cl_kernel, int, int, int *, int *);


/*----------------------------------------------------------------------------*/
/* Global library variables                                                   */
/*----------------------------------------------------------------------------*/

GmlSct gml;


/*----------------------------------------------------------------------------*/
/* Global tables                                                              */
/*----------------------------------------------------------------------------*/

char *MshStr[7]   = { "", "float4", "int2", "int4", "int4", "int4", "int8"};

char *TypStr[11]   = {  "int", "int2", "int4", "int8", "int16", "long16",
                        "float", "float2", "float4", "float8", "float16" };

int  EleDeg[7]    = {0,0,2,3,4,4,8};

int  EleSiz[7]    = {1, sizeof(cl_float4), sizeof(cl_int2), sizeof(cl_int4),
                        sizeof(cl_int4),   sizeof(cl_int4), sizeof(cl_int8)};

int  DegMat[7][7] = {   {0,0,0,0,0,0,0},
                        {0,1,16,8,4,32,8},
                        {0,2,1,2,2,8,4},
                        {0,3,3,1,0,2,0},
                        {0,4,4,0,1,0,2},
                        {0,4,6,4,0,1,0},
                        {0,8,12,0,6,0,1} };

int  TypMat[7][7] = { {0,0,0,0,0,0,0},
                      {0,GmlFloat4,GmlInt16,GmlInt8,GmlInt4,GmlInt32,GmlInt8},
                      {0,  GmlInt2, GmlInt2,GmlInt2,GmlInt2, GmlInt8,GmlInt4},
                      {0,  GmlInt4, GmlInt4,GmlInt4,      0, GmlInt2,      0},
                      {0,  GmlInt4, GmlInt4,      0,GmlInt4,       0,GmlInt2},
                      {0,  GmlInt4, GmlInt8,GmlInt4,      0, GmlInt4,      0},
                      {0,  GmlInt8,GmlInt16,      0,GmlInt8,       0,GmlInt8} };


/*----------------------------------------------------------------------------*/
/* Init device, context and queue                                             */
/*----------------------------------------------------------------------------*/

GmlParSct *GmlInit(int mod)
{
   int err;
   cl_platform_id platforms[10];
   cl_uint num_platforms;
   GmlParSct *par;

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

   // Load all internal reduction kernels
/*   if(!(gml.RedKrnIdx[ GmlMin ] = GmlNewKernel(reduce, "reduce_min")))
      return(NULL);

   if(!(gml.RedKrnIdx[ GmlSum ] = GmlNewKernel(reduce, "reduce_sum")))
      return(NULL);

   if(!(gml.RedKrnIdx[ GmlMax ] = GmlNewKernel(reduce, "reduce_max")))
      return(NULL);
*/
   // Allocate and return a public user customizable parameter structure
   if(!(gml.ParIdx = GmlNewData(GmlRawData, NULL, 1, 0, 1, "GmlParSct", sizeof(GmlParSct))))
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
   int i;
   size_t GpuNamSiz;
   char GpuNam[100];
   cl_platform_id platforms[10];
   cl_device_id device_id[ MaxGpu ];
   cl_uint num_platforms, num_devices;

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
/* Allocate an OpenCL buffer plus 10% more for resizing                       */
/*----------------------------------------------------------------------------*/

int GmlNewData(int MshTyp, char *NamStr, int NmbLin, ...)
{
   int idx, RefTyp, NmbCol;
   char *RawStr;
   GmlDatSct *dat;
   size_t LinSiz;
   va_list VarArg;

   if( (MshTyp < GmlRawData) || (MshTyp > GmlHexahedra) || (NmbLin <= 0) )
      return(0);

   // Look for a free data socket
   for(idx=1;idx<=GmlMaxDat;idx++)
      if(!gml.dat[ idx ].GpuMem)
         break;

   if(idx > GmlMaxDat)
      return(0);

   dat = &gml.dat[ idx ];
   dat->MshTyp = MshTyp;
   dat->NmbLin = NmbLin;
   dat->NamStr = NamStr;

   if(MshTyp == GmlRawData)
   {
      va_start(VarArg, NmbLin);
      RefTyp = va_arg(VarArg, int);
      NmbCol = va_arg(VarArg, int);
      RawStr = va_arg(VarArg, char *);
      LinSiz = va_arg(VarArg, size_t);
      va_end(VarArg);

      if( (RefTyp < GmlRawData) || (RefTyp > GmlHexahedra)
      ||  !LinSiz || !RawStr )
      {
         return(0);
      }

      dat->RefTyp = RefTyp;
      dat->NmbCol = NmbCol;
      dat->LinSiz = NmbCol * LinSiz;
      dat->TypStr = RawStr;
   }
   else
   {
      dat->RefTyp = MshTyp;
      dat->NmbCol = 1;
      dat->LinSiz = EleSiz[ MshTyp ];
      dat->TypStr = MshStr[ MshTyp ];
      gml.IdxTab[ MshTyp ] = idx;
   }

   dat->siz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;

   // Allocate the requested memory size on the GPU
   dat->GpuMem = clCreateBuffer( gml.context, CL_MEM_READ_WRITE,
                                 dat->siz, NULL, NULL );

   if(!dat->GpuMem)
   {
      printf("Cannot allocate %ld MB on the GPU (%ld MB already used)\n",
               dat->siz/MB, GmlGetMemoryUsage()/MB);
      return(0);
   }

   // Allocate the requested memory size on the CPU side
   if(!(dat->CpuMem = calloc(1, dat->siz)))
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
/* Set a procedures to set or get lines of data                               */
/*----------------------------------------------------------------------------*/

int GmlSetDataLine(int idx, int lin, void *UsrDat)
{
   char *adr;
   GmlDatSct *dat;

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) )
      return(0);

   dat = &gml.dat[ idx ];

   if( (lin < 0) || (lin >= dat->NmbLin) )
      return(0);

   dat->UplFlg = 0;
   adr = (char *)dat->CpuMem + lin * dat->LinSiz;
   memcpy(adr, UsrDat, dat->LinSiz);

   return(1);
}

int GmlGetDataLine(int idx, int lin, void *UsrDat)
{
   char *adr;
   GmlDatSct *dat;

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) )
      return(0);

   dat = &gml.dat[ idx ];

   if( (lin < 0) || (lin >= dat->NmbLin) )
      return(0);

   if(!dat->DwlFlg)
   {
      GmlDownloadData(idx);
      dat->DwlFlg = 1;
   }

   adr = (char *)dat->CpuMem + lin * dat->LinSiz;
   memcpy(UsrDat, adr, dat->LinSiz);

   return(1);
}

int GmlSetDataBlock(int idx, void *UsrBeg, void *UsrEnd)
{
   char *DstAdr, *SrcAdr;
   size_t UsrSiz;
   GmlDatSct *dat;

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) )
      return(0);

   dat = &gml.dat[ idx ];
   dat->UplFlg = 0;
   DstAdr = (char *)dat->CpuMem;
   SrcAdr = (char *)UsrBeg;
   UsrSiz = ((char *)UsrEnd - (char *)UsrBeg) / (dat->NmbLin - 1);

   while(SrcAdr < (char *)UsrEnd)
   {
      memcpy(DstAdr, SrcAdr, dat->LinSiz);
      DstAdr += dat->LinSiz;
      SrcAdr += UsrSiz;
   }

   return(1);
}

int GmlGetDataBlock(int idx, void *UsrBeg, void *UsrEnd)
{
   char *DstAdr, *SrcAdr;
   size_t UsrSiz;
   GmlDatSct *dat;

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) )
      return(0);

   dat = &gml.dat[ idx ];

   if(!dat->DwlFlg)
   {
      GmlDownloadData(idx);
      dat->DwlFlg = 1;
   }

   DstAdr = (char *)dat->CpuMem;
   SrcAdr = (char *)UsrBeg;
   UsrSiz = ((char *)UsrEnd - (char *)UsrBeg) / (dat->NmbLin - 1);

   while(SrcAdr < (char *)UsrEnd)
   {
      memcpy(SrcAdr, DstAdr, dat->LinSiz);
      DstAdr += dat->LinSiz;
      SrcAdr += UsrSiz;
   }

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Copy user's data into an OpenCL buffer                                     */
/*----------------------------------------------------------------------------*/

static int GmlUploadData(int idx)
{
   GmlDatSct *dat = &gml.dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || !dat->CpuMem )
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
/* Vectorized arbitrary sized data type                                       */
/*----------------------------------------------------------------------------*/

int GmlNewBall(int typ1, int typ2)
{
   int i, j, PriIdx, SecIdx, ExtIdx, (*DegTab)[2], BalIdx;
   int *SecTab, *PriVec, (*PriExt)[3], *ExtDat;
   unsigned char *PriDeg;
   GmlBalSct *bal;
   GmlDatSct *PriDat = &gml.dat[ typ1 ], *SecDat = &gml.dat[ typ2 ];

   // Check input indices
   if( (typ1 < 0) || (typ1 > GmlMaxDat) || (PriDat->MshTyp != GmlVertices) )
      return(0);

   if( (typ2 < 0) || (typ2 > GmlMaxDat) || (SecDat->MshTyp < GmlEdges)
   || (SecDat->MshTyp > GmlHexahedra) )
   {
      return(0);
   }

   // Find a free ball structure
   for(BalIdx=1; BalIdx<=GmlMaxBal; BalIdx++)
      if(!gml.bal[ BalIdx ].NmbPri)
         break;

   if(BalIdx > GmlMaxBal)
      return(0);

   // Init the structure header and allocate primary tables
   bal = &gml.bal[ BalIdx ];
   bal->NmbPri = PriDat->NmbLin;
   bal->NmbSec = SecDat->NmbLin;
   bal->SecSiz = EleDeg[ SecDat->MshTyp ];
   bal->VecSiz = (bal->SecSiz * bal->NmbSec) / bal->NmbPri;
   bal->NmbExtPri = bal->NmbExtDat = 0;
   SecTab = (int *)SecDat->CpuMem;

   if(bal->VecSiz < 2)
      bal->VecSiz = 2;
   else if(bal->VecSiz < 4)
      bal->VecSiz = 4;
   else if(bal->VecSiz < 8)
      bal->VecSiz = 8;
   else if(bal->VecSiz < 16)
      bal->VecSiz = 16;
   else
      bal->VecSiz = 32;

   if(bal->SecSiz <= 2)
      bal->SecPadSiz = 2;
   else if(bal->SecSiz <= 4)
      bal->SecPadSiz = 4;
   else if(bal->SecSiz <= 8)
      bal->SecPadSiz = 8;
   else if(bal->SecSiz <= 16)
      bal->SecPadSiz = 16;
   else
      bal->SecPadSiz = 32;

   bal->PriDegIdx = GmlNewData(  GmlRawData, NULL, bal->NmbPri, GmlVertices,
                                 1, "char", sizeof(cl_char) );
   bal->PriVecIdx = GmlNewData(  GmlRawData, NULL, bal->NmbPri, GmlVertices,
                                 1, "int", bal->VecSiz * sizeof(cl_int) );

   PriDeg = gml.dat[ bal->PriDegIdx ].CpuMem;
   PriVec = gml.dat[ bal->PriVecIdx ].CpuMem;

   // Build temporary vertex degrees table
   DegTab = calloc(bal->NmbPri, 2 * sizeof(int));

   for(i=0;i<bal->NmbSec;i++)
      for(j=0;j<bal->SecSiz;j++)
         DegTab[ SecTab[ i * bal->SecPadSiz + j ] ][0]++;

   // Count the number of primary type elements whose size exceeds
   // that of the vector and the number of extra datas needed
   for(i=0;i<bal->NmbPri;i++)
      if(DegTab[i][0] > bal->VecSiz)
      {
         bal->NmbExtPri++;
         bal->NmbExtDat += DegTab[i][0] - bal->VecSiz;
      }

   // Allocate both tables
   bal->PriExtIdx = GmlNewData(  GmlRawData, NULL, bal->NmbExtPri,
                                 0, 1, "int", 3 * sizeof(cl_int) );
   bal->ExtDatIdx = GmlNewData(  GmlRawData, NULL, bal->NmbExtDat,
                                 0, 1, "int",     sizeof(cl_int) );

   PriExt = gml.dat[ bal->PriExtIdx ].CpuMem;
   ExtDat = gml.dat[ bal->ExtDatIdx ].CpuMem;
   bal->NmbExtPri = bal->NmbExtDat = 0;

   // Fill the primary type extension table
   for(i=0;i<bal->NmbPri;i++)
   {
      if(DegTab[i][0] > bal->VecSiz)
      {
         PriExt[ bal->NmbExtPri ][0] = i;
         PriExt[ bal->NmbExtPri ][1] = bal->NmbExtDat;
         DegTab[i][1] = bal->NmbExtPri;
         bal->NmbExtPri++;
         bal->NmbExtDat += DegTab[i][0] - bal->VecSiz;
      }

      DegTab[i][0] = 0;
   }

   // Fill the extension datas
   for(i=0;i<bal->NmbSec;i++)
      for(j=0;j<bal->SecSiz;j++)
      {
         PriIdx = SecTab[ i * bal->SecPadSiz + j ];
         SecIdx = (i << 3) | j;

         if(DegTab[ PriIdx ][0] < bal->VecSiz)
            PriVec[ PriIdx * bal->VecSiz + DegTab[ PriIdx ][0] ] = SecIdx;
         else
         {
            ExtIdx = DegTab[ PriIdx ][1];
            ExtDat[ PriExt[ ExtIdx ][1] + PriExt[ ExtIdx ][2] ] = SecIdx;
            PriExt[ ExtIdx ][2]++;
         }

         DegTab[ PriIdx ][0]++;
      }

   // Set the base degree for each vertices and set to -1 useless vector data
   for(i=0;i<bal->NmbPri;i++)
   {
      PriDeg[i] = (DegTab[i][0] <= bal->VecSiz) ? DegTab[i][0] : bal->VecSiz;

      for(j=DegTab[i][0]; j<bal->VecSiz; j++)
         PriVec[ i * bal->VecSiz + j ] = -1;
   }

   free(DegTab);

   return(BalIdx);
}


/*----------------------------------------------------------------------------*/
/* Release four OpenCL buffers associated with a ball                         */
/*----------------------------------------------------------------------------*/

int GmlFreeBall(int idx)
{
   GmlBalSct *bal = &gml.bal[ idx ];

   if( (idx < 1) || (idx > GmlMaxBal) || !bal->NmbPri )
      return(0);

   GmlFreeData(bal->PriVecIdx);
   GmlFreeData(bal->PriExtIdx);
   GmlFreeData(bal->ExtDatIdx);
   GmlFreeData(bal->PriDegIdx);

   return(1);

}


/*----------------------------------------------------------------------------*/
/* Send all four data fields to the GPU                                       */
/*----------------------------------------------------------------------------*/

int GmlUploadBall(int idx)
{
   GmlBalSct *bal = &gml.bal[ idx ];

   if( (idx < 1) || (idx > GmlMaxBal) || !bal->NmbPri )
      return(0);

   GmlUploadData(bal->PriDegIdx);
   GmlUploadData(bal->PriVecIdx);
   GmlUploadData(bal->PriExtIdx);
   GmlUploadData(bal->ExtDatIdx);

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Read and compile an OpenCL source code                                     */
/*----------------------------------------------------------------------------*/

int GmlNewKernel(char *KernelSource)
{
   int idx = ++gml.NmbKrn;
   GmlKrnSct *krn = &gml.krn[ idx ];

   if(idx > GmlMaxKrn)
      return(0);

   krn->kernel = NULL;
   krn->KrnSrc = KernelSource;
   sprintf(krn->PrcNam, "kernel%d", idx);

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Read and compile an OpenCL source code                                     */
/*----------------------------------------------------------------------------*/

static int GmlCompileKernel(int idx, char *KernelSource, char *PrcNam)
{
   char *buffer, *StrTab[1];
   int err;
   GmlKrnSct *krn = &gml.krn[ idx ];
   size_t len, LenTab[1];

   StrTab[0] = krn->KrnSrc;
   LenTab[0] = strlen(krn->KrnSrc)-1;

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

   if(!(krn->kernel = clCreateKernel(krn->program, krn->PrcNam, &err))
   || (err != CL_SUCCESS))
   {
      return(0);
   }

//   clGetProgramInfo(krn->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &len, NULL);
//      printf("\ncompilation of %s : %ld bytes\n",krn->PrcNam,len);

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

double GmlLaunchKernel(int idx, int DatIdx, ...)
{
   int i, j, typ, cpt, OldNmbDat, NmbDat;
   int DatTab[ GmlMaxDat ], MemTyp[ GmlMaxDat ];
   int TruSiz = gml.dat[ DatIdx ].NmbLin, DatFlg[ GmlEnd ];
   double GpuTim;
   va_list VarArg;
   GmlDatSct *dat, *RefDat = &gml.dat[ DatIdx ];
   GmlKrnSct *krn = &gml.krn[ idx ];
   char KrnSrc[10000], TmpStr[100];

   if( (idx < 1) || (idx > gml.NmbKrn) )
      return(-1);

   // Get arguments list
   va_start(VarArg, DatIdx);
   i = 0;

   do
   {
      typ = va_arg(VarArg, int);

      if(typ == GmlEnd)
         break;

      MemTyp[i] = typ;
      DatTab[i] = va_arg(VarArg, int);

      dat = &gml.dat[ DatTab[i] ];

      if( ((MemTyp[i] == GmlRead) || (MemTyp[i] == GmlReadWrite)) && !dat->UplFlg)
      {
         GmlUploadData(DatTab[i]);
         dat->UplFlg = 1;
      }
   }while(i++ < GmlMaxDat);

   NmbDat = i;
   va_end(VarArg);

   if(!krn->kernel)
   {
      for(i=0;i<GmlEnd;i++)
         DatFlg[i] = 0;

      for(i=0;i<NmbDat;i++)
         DatFlg[ gml.dat[ DatTab[i] ].MshTyp ] = 1;

      OldNmbDat = NmbDat;

      for(i=0;i<OldNmbDat;i++)
      {
         dat = &gml.dat[ DatTab[i] ];
         printf("dat %d, type %d, ref type %d\n", i, dat->MshTyp, dat->RefTyp);

         if( (dat->RefTyp > dat->MshTyp) && !DatFlg[ dat->RefTyp ] )
         {
            DatFlg[ dat->RefTyp ] = 1;
            DatTab[ NmbDat++ ] = gml.IdxTab[ dat->RefTyp ];
            printf("type %d adds type %d (idx = %d)\n", dat->MshTyp, dat->RefTyp, gml.IdxTab[ dat->RefTyp ]);
         }
      }

      KrnSrc[0] = '\0';
      sprintf(TmpStr, "typedef struct\n{\n   int empty;\n}GmlParSct;\n");
      strcat(KrnSrc, TmpStr);

      sprintf(TmpStr, "__kernel void %s(", krn->PrcNam);
      strcat(KrnSrc, TmpStr);

      for(i=0;i<NmbDat;i++)
      {
         dat = &gml.dat[ DatTab[i] ];
         //cpt = DegMat[ RefDat->MshTyp ][ dat->MshTyp ];
         //typ = TypMat[ RefDat->MshTyp ][ dat->MshTyp ];

         if( (dat->MshTyp == GmlRawData) && (dat->NmbCol > 1) )
            sprintf(TmpStr, "__global %s (*%sTab)[%d], ", dat->TypStr, dat->NamStr, dat->NmbCol);
         else
            sprintf(TmpStr, "__global %s *%sTab, ", dat->TypStr, dat->NamStr);

         strcat(KrnSrc, TmpStr);
      }

      strcat(KrnSrc, "__global GmlParSct *par, const int count )\n");
      strcat(KrnSrc, "{\n   int idx = get_global_id(0);\n\n   if(idx >= count)\n      return;\n\n");

      for(i=0;i<NmbDat;i++)
      {
         dat = &gml.dat[ DatTab[i] ];
         cpt = DegMat[ RefDat->MshTyp ][ dat->RefTyp ];
         typ = TypMat[ RefDat->MshTyp ][ dat->RefTyp ];

         if(cpt > 1)
            sprintf( TmpStr, "   %s %s%s[%d];\n", dat->TypStr, RefDat->NamStr, dat->NamStr, cpt);
         else if( (dat->MshTyp == GmlRawData) && (dat->NmbCol > 1) )
            sprintf( TmpStr, "   %s %s[%d];\n", dat->TypStr, dat->NamStr, dat->NmbCol);
         else
            sprintf( TmpStr, "   %s %s;\n", dat->TypStr, dat->NamStr);

         strcat(KrnSrc, TmpStr);
      }

      strcat(KrnSrc, "\n");

      for(i=0;i<NmbDat;i++)
      {
         if( (MemTyp[i] != GmlRead) && (MemTyp[i] != GmlReadWrite) )
            continue;

         dat = &gml.dat[ DatTab[i] ];
         cpt = DegMat[ RefDat->MshTyp ][ dat->RefTyp ];
         typ = TypMat[ RefDat->MshTyp ][ dat->RefTyp ];

         if(cpt == 1)
         {
            sprintf( TmpStr, "   %s = %sTab[idx];\n", dat->NamStr, dat->NamStr);
            strcat(KrnSrc, TmpStr);
         }
         else
         {
            for(j=0;j<cpt;j++)
            {
               sprintf( TmpStr, "   %s%s[%d] = %sTab[ %s.s%d ];\n",
                        RefDat->NamStr, dat->NamStr, j, dat->NamStr, RefDat->NamStr, j);
               strcat(KrnSrc, TmpStr);
            }
         }
      }

      strcat(KrnSrc, krn->KrnSrc);

      for(i=0;i<NmbDat;i++)
      {
         if( (MemTyp[i] != GmlWrite) && (MemTyp[i] != GmlReadWrite) )
            continue;

         dat = &gml.dat[ DatTab[i] ];

         if( (dat->MshTyp == GmlRawData) && (dat->NmbCol > 1) )
         {
            for(j=0;j<cpt;j++)
            {
               sprintf( TmpStr, "   %sTab[idx][%d] = %s[%d];\n",
                        dat->NamStr, j, dat->NamStr, j );
               strcat(KrnSrc, TmpStr);
            }
         }
         else
         {
            sprintf(TmpStr, "   %sTab[idx] = %s;\n",dat->NamStr, dat->NamStr);
            strcat(KrnSrc, TmpStr);
         }
      }

      sprintf(TmpStr, "}\n");
      strcat(KrnSrc, TmpStr);

      puts(KrnSrc);
      krn->KrnSrc = KrnSrc;
      GmlCompileKernel(idx, NULL, NULL);
      //return(-1);
   }

   // First send the parameters to the GPU memmory
   if(!GmlUploadData(gml.ParIdx))
      return(-2);

   GpuTim = GmlOpenClLaunch(krn->kernel, TruSiz, NmbDat, DatTab, MemTyp);
   printf("RUNTIME = %g\n", GpuTim);

   for(i=0;i<NmbDat;i++)
      if( (MemTyp[i] == GmlWrite) || (MemTyp[i] == GmlReadWrite) )
         gml.dat[ DatTab[i] ].DwlFlg = 0;

   // Finaly, get back the parameters from the GPU memmory
   if(!GmlDownloadData(gml.ParIdx))
      return(-10);

   return(GpuTim);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

double GmlOpenClLaunch( cl_kernel kernel, int TruSiz,
                        int NmbDat, int *DatTab, int *MemTyp )
{
   int i;
   size_t GloSiz, LocSiz, RetSiz = 0;
   GmlDatSct *dat;
   cl_event event;
   cl_ulong start, end;

   // Build arguments list

   for(i=0;i<NmbDat;i++)
   {
      dat = &gml.dat[ DatTab[i] ];

      if( (DatTab[i] < 1) || (DatTab[i] > GmlMaxDat) || !dat->GpuMem
      || (clSetKernelArg(kernel, i, sizeof(cl_mem), &dat->GpuMem) != CL_SUCCESS) )
      {
         printf("i=%d, DatTab[i]=%d, GpuMem=%p\n",i,DatTab[i],dat->GpuMem);
         return(-3);
      }
   }

   if(clSetKernelArg(kernel, NmbDat, sizeof(cl_mem),
                     &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
   {
      return(-4);
   }

   if(clSetKernelArg(kernel, NmbDat+1, sizeof(int), &TruSiz) != CL_SUCCESS)
      return(-5);

   // Fit data loop size to the GPU kernel size
   if(clGetKernelWorkGroupInfo(  kernel, gml.device_id[ gml.CurDev ],
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

   if(clEnqueueNDRangeKernel( gml.queue, kernel, 1, NULL,
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

   return((double)(end - start) / 1e9);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

double GmlLaunchBallKernel(int KrnIdx1, int KrnIdx2, int BalIdx, int NmbDat, ...)
{
   int i, DatIdxTab[ GmlMaxDat ];
   double tim1, tim2;
   size_t GloSiz, LocSiz, RetSiz = 0;
   va_list VarArg;
   GmlDatSct *dat, *DatTab[ GmlMaxDat ];
   GmlBalSct *bal = &gml.bal[ BalIdx ];
   GmlKrnSct *krn1 = &gml.krn[ KrnIdx1 ], *krn2 = &gml.krn[ KrnIdx2 ];
   cl_event event;
   cl_ulong start, end;


   /*-----------------------*/
   /* Check inputs validity */
   /*-----------------------*/

   va_start(VarArg, NmbDat);

   for(i=0;i<NmbDat;i++)
   {
      DatIdxTab[i] = va_arg(VarArg, int);
      DatTab[i] = &gml.dat[ DatIdxTab[i] ];
   }

   va_end(VarArg);

   if( (KrnIdx1 < 1) || (KrnIdx1 > gml.NmbKrn) || !krn1->kernel
   ||   (KrnIdx2 < 1) || (KrnIdx2 > gml.NmbKrn) || !krn2->kernel )
   {
      return(-1);
   }

   if( (BalIdx < 1) || (BalIdx > GmlMaxBal) || !bal->NmbPri )
      return(-2);

   for(i=0;i<NmbDat;i++)
      if( (DatIdxTab[i] < 1) || (DatIdxTab[i] > GmlMaxDat) || !DatTab[i]->GpuMem )
         return(-3);

   // First send the parameters to the GPU memmory
   if(!GmlUploadData(gml.ParIdx))
      return(-4);


   /*-----------------------------------*/
   /* Build first kernel arguments list */
   /*-----------------------------------*/

   if((clSetKernelArg(  krn1->kernel, 0, sizeof(cl_mem),
                        &gml.dat[ bal->PriDegIdx ].GpuMem) != CL_SUCCESS)
   || (clSetKernelArg(  krn1->kernel, 1, sizeof(cl_mem),
                        &gml.dat[ bal->PriVecIdx ].GpuMem) != CL_SUCCESS) )
   {
      return(-5);
   }

   for(i=0;i<NmbDat;i++)
      if(clSetKernelArg(krn1->kernel, i+2, sizeof(cl_mem),
                        &DatTab[i]->GpuMem) != CL_SUCCESS)
      {
         return(-6);
      }

   if(clSetKernelArg(krn1->kernel, NmbDat+2, sizeof(cl_mem),
                     &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
   {
      return(-7);
   }

   if(clSetKernelArg(krn1->kernel, NmbDat+3, sizeof(int),
      &bal->NmbPri) != CL_SUCCESS)
   {
      return(-8);
   }

   // Fit data loop size to the GPU kernel size
   if(clGetKernelWorkGroupInfo(  krn1->kernel, gml.device_id[ gml.CurDev ],
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                 &LocSiz, &RetSiz) != CL_SUCCESS )
   {
      return(-9);
   }

   GloSiz = bal->NmbPri / LocSiz;
   GloSiz *= LocSiz;

   if(GloSiz < bal->NmbPri)
      GloSiz += LocSiz;

   // Launch GPU code
   clFinish(gml.queue);

   if(clEnqueueNDRangeKernel( gml.queue, krn1->kernel, 1, NULL,
                              &GloSiz, &LocSiz, 0, NULL, &event) )
   {
      return(-10);
   }

   clFinish(gml.queue);

   if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL) != CL_SUCCESS)
   {
      return(-11);
   }

   if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, NULL) != CL_SUCCESS)
   {
      return(-12);
   }

   tim1 = end - start;

   if(!bal->NmbExtPri)
      return(tim1 / 1e9);


   /*------------------------------------*/
   /* Build second kernel arguments list */
   /*------------------------------------*/

   if((clSetKernelArg(  krn2->kernel, 0, sizeof(cl_mem),
                        &gml.dat[ bal->PriExtIdx ].GpuMem) != CL_SUCCESS)
   || (clSetKernelArg(  krn2->kernel, 1, sizeof(cl_mem),
                        &gml.dat[ bal->ExtDatIdx ].GpuMem) != CL_SUCCESS) )
   {
      return(-13);
   }

   for(i=0;i<NmbDat;i++)
      if(clSetKernelArg(krn2->kernel, i+2, sizeof(cl_mem),
                        &DatTab[i]->GpuMem) != CL_SUCCESS)
      {
         return(-14);
      }

   if(clSetKernelArg(krn2->kernel, NmbDat+2, sizeof(cl_mem),
                     &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
   {
      return(-15);
   }

   if(clSetKernelArg(krn2->kernel, NmbDat+3, sizeof(int),
                     &bal->NmbExtPri) != CL_SUCCESS)
   {
      return(-16);
   }

   // Fit data loop size to the GPU kernel size
   if(clGetKernelWorkGroupInfo(  krn2->kernel, gml.device_id[ gml.CurDev ],
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                 &LocSiz, &RetSiz) != CL_SUCCESS )
   {
      return(-17);
   }

   GloSiz = bal->NmbExtPri / LocSiz;
   GloSiz *= LocSiz;

   if(GloSiz < bal->NmbExtPri)
      GloSiz += LocSiz;

   // Launch GPU code
   clFinish(gml.queue);

   if(clEnqueueNDRangeKernel( gml.queue, krn2->kernel, 1, NULL,
                              &GloSiz, &LocSiz, 0, NULL, &event) )
   {
      return(-18);
   }

   clFinish(gml.queue);

   if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL) != CL_SUCCESS)
   {
      return(-19);
   }

   if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, NULL) != CL_SUCCESS)
   {
      return(-20);
   }

   tim2 = end - start;

   // Finaly, get back the parameters from the GPU memmory
   if(!GmlDownloadData(gml.ParIdx))
      return(-21);

   return((tim1 + tim2) / 1e9);
}


/*----------------------------------------------------------------------------*/
/* Compute various reduction functions: min,max,L1,L2 norms                   */
/*----------------------------------------------------------------------------*/

double GmlReduceVector(int DatIdx, int opp, double *res)
{
   int i;
   float *vec;
   double tim;
   GmlDatSct *dat = &gml.dat[ DatIdx ], *dat2;

   // Check indices
   if( (DatIdx < 1) || (DatIdx > GmlMaxDat) || !dat->GpuMem
   || (opp < GmlMin) || (opp > GmlMax) )
   {
      return(-1);
   }

   // Allocate an output vector the size of the input vector
   if(!dat->RedVecIdx)
       if(!(dat->RedVecIdx = GmlNewData(GmlRawData, NULL, 1, 0, 1, "float", dat->siz)))
         return(-2);
         
   // Launch the right reduction kernel according to the requested opperation
   tim = GmlLaunchKernel(  gml.RedKrnIdx[ opp ], dat->siz/sizeof(float),
                           GmlRead, DatIdx, GmlWrite, dat->RedVecIdx , GmlEnd);

   if(tim < 0)   
      return(tim);

   // Trim the size of the output vector down to the number of OpenCL groups
   // used by the kernel and download this amount of data
   dat2 = &gml.dat[ dat->RedVecIdx ];
   dat2->siz = dat->siz / gml.CurLocSiz;
   GmlDownloadData(dat->RedVecIdx);
   vec = (float *)dat2->CpuMem;

   // Perform the last reduction step with the CPU
   switch(opp)
   {
      case GmlMin :
      {
         *res = 1e37;
         for(i=0;i<dat2->siz/sizeof(float);i++)
            *res = MIN(*res, vec[i]);
      }break;

      case GmlSum :
      {
         *res = 0.;
         for(i=0;i<dat2->siz/sizeof(float);i++)
            *res += vec[i];
      }break;

      case GmlMax :
      {
         *res = -1e37;
         for(i=0;i<dat2->siz/sizeof(float);i++)
            *res = MAX(*res, vec[i]);
      }break;
   }

   return(tim);
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
