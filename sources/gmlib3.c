

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.14                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: mar 11 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gmlib3.h"
#include "reduce.h"


/*----------------------------------------------------------------------------*/
/* Defines                                                                    */
/*----------------------------------------------------------------------------*/

#define MB           1048576
#define VECPOWOCL    4
#define VECPOWMAX    7
enum data_type       {GmlArgDat, GmlRawDat, GmlLnkDat, GmlEleDat, GmlRefDat};
enum memory_type     {GmlInternal, GmlInput, GmlOutput, GmlInout};


/*----------------------------------------------------------------------------*/
/* Library's internal data structures                                         */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int            EleIdx, ItmIdx, NxtDat, nod[4];
}BucSct;

typedef struct
{
   int            HshTyp, DatLen, KeyLen, *HshTab;
   int64_t        NmbMis, NmbHit, TabSiz, NmbDat, NxtDat;
   BucSct         *DatTab;
}HshTabSct;

typedef struct
{
   int            ArgIdx, MshTyp, DatIdx, LnkTyp, LnkIdx, LnkDir, CntIdx;
   int            LnkDeg, MaxDeg, NmbItm, ItmLen, ItmTyp, FlgTab;
   const char     *nam;
}ArgSct;

typedef struct
{
   int            AloTyp, MemAcs, MshTyp, LnkTyp, ItmTyp, RedIdx;
   int            NmbItm, ItmLen, ItmSiz, NmbLin, LinSiz;
   char           *src, use;
   const char     *nam;
   size_t         MemSiz;
   cl_mem         GpuMem;
   void           *CpuMem;
}DatSct;

typedef struct
{
   int            idx, HghIdx, NmbLin[2], NmbDat, DatTab[ GmlMaxDat ];
   cl_kernel      kernel;
   cl_program     program; 
}KrnSct;

typedef struct
{
   int            NmbKrn, ParIdx, CurDev, DbgFlg, DblExt, FpnSiz;
   int            TypIdx[ GmlMaxEleTyp ];
   int            RefIdx[ GmlMaxEleTyp ];
   int            NmbEle[ GmlMaxEleTyp ];
   int            LnkMat[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   int            LnkHgh[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   int            CntMat[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   int            RedKrn[ GmlMaxRed ];
   cl_uint        NmbDev;
   size_t         MemSiz, CurGrpSiz, MovSiz;
   DatSct         dat[ GmlMaxDat + 1 ];
   KrnSct         krn[ GmlMaxKrn + 1 ];
   cl_device_id   device_id[ MaxGpu ];
   cl_context     context;
   cl_command_queue queue;
}GmlSct;


/*----------------------------------------------------------------------------*/
/* Macro instructions                                                         */
/*----------------------------------------------------------------------------*/

#define MIN(a,b)        ((a) < (b) ? (a) : (b))
#define MAX(a,b)        ((a) > (b) ? (a) : (b))
#define CHKELETYP(t)    if( ((t) < 0) || ((t) >= GmlMaxEleTyp)) return(0)
#define CHKOCLTYP(t)    if( ((t) < GmlInt) || ((t) >= GmlMaxOclTyp)) return(0)
#define GETGMLPTR(p,i)  GmlSct *p = (GmlSct *)(i)


/*----------------------------------------------------------------------------*/
/* Prototypes of local procedures                                             */
/*----------------------------------------------------------------------------*/

static int     NewData                 (GmlSct *, DatSct *);
static int     NewBallData             (GmlSct *, int, int, char *, char *);
static int     UploadData              (GmlSct *, int);
static int     DownloadData            (GmlSct *, int);
static int     NewOclKrn               (GmlSct *, char *, char *);
static int     GetNewDatIdx            (GmlSct *);
static double  RunOclKrn               (GmlSct *, KrnSct *);
static void    WriteUserTypedef        (char *, char *);
static void    WriteProcedureHeader    (char *, char *, int, int, ArgSct *);
static void    WriteKernelVariables    (char *, int, int, ArgSct *);
static void    WriteKernelMemoryReads  (char *, int, int, ArgSct *);
static void    WriteKernelMemoryWrites (char *, int, int, ArgSct *);
static void    WriteUserKernel         (char *, char *);
static void    GetCntVec               (int , int *, int *, int *);
static int     BldLnk                  (GmlSct *, int, int);
static void    GetItmNod               (int *, int, int, int, int *);
static int     CalHshKey               (HshTabSct *, int *);
static void    AddHsh                  (HshTabSct *, int, int, int, int *);
static int     GetHsh                  (HshTabSct *, int, int, int, int *, int *);


/*----------------------------------------------------------------------------*/
/* Global tables                                                              */
/*----------------------------------------------------------------------------*/

static const char OclHexNmb[16] = {
   '0','1','2','3','4','5','6','7', '8','9','a','b','c','d','e','f' };

static const int  OclTypSiz[ GmlMaxOclTyp ] = {
   sizeof(cl_int),
   sizeof(cl_int2),
   sizeof(cl_int4),
   sizeof(cl_int8),
   sizeof(cl_int16),
   sizeof(cl_float),
   sizeof(cl_float2),
   sizeof(cl_float4),
   sizeof(cl_float8),
   sizeof(cl_float16),
   sizeof(cl_double),
   sizeof(cl_double2),
   sizeof(cl_double4),
   sizeof(cl_double8),
   sizeof(cl_double16) };

static const char *OclTypStr[ GmlMaxOclTyp ]  = {
   "int      ",
   "int2     ",
   "int4     ",
   "int8     ",
   "int16    ",
   "float    ",
   "float2   ",
   "float4   ",
   "float8   ",
   "float16  ",
   "double   ",
   "double2  ",
   "double4  ",
   "double8  ",
   "double16 " };

static const char *OclNulVec[ GmlMaxOclTyp ]  = {
   "(int){0}",
   "(int2){0,0}",
   "(int4){0,0,0,0}",
   "(int8){0,0,0,0,0,0,0,0}",
   "(int16){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}",
   "(float){0.}",
   "(float2){0.,0.}",
   "(float4){0.,0.,0.,0.}",
   "(float8){0.,0.,0.,0.,0.,0.,0.,0.}",
   "(float16){0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}",
   "(double){0.}",
   "(double2){0.,0.}",
   "(double4){0.,0.,0.,0.}",
   "(double8){0.,0.,0.,0.,0.,0.,0.,0.}",
   "(double16){0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}" };

static const int  TypVecSiz[ GmlMaxOclTyp ]  = {1,2,4,8,16,1,2,4,8,16,1,2,4,8,16};

static const int  OclVecPow[ VECPOWOCL +1 ]  = {
   GmlInt, GmlInt2, GmlInt4, GmlInt8, GmlInt16};

static const int  MshItmTyp[ GmlMaxEleTyp ]  = {
   GmlFlt4, GmlInt2, GmlInt4, GmlInt4, GmlInt4, GmlInt8, GmlInt8, GmlInt8};

static const int  EleNmbNod[ GmlMaxEleTyp ]  = {0, 2, 3, 4, 4, 5, 6, 8};
static const int  EleItmLen[ GmlMaxEleTyp ]  = {4, 2, 4, 4, 4, 8, 8, 8};
static const int  EleNmbItm[ GmlMaxEleTyp ]  = {1, 1, 1, 1, 1, 1, 1, 1};
static const int  MshTypDim[ GmlMaxEleTyp ]  = {0,1,2,2,3,3,3,3};

static const char *BalTypStr[ GmlMaxEleTyp ]  = {
   "Ver", "Edg", "Tri", "Qad", "Tet", "Pyr", "Pri", "Hex"};

static const char *MshTypStr[ GmlMaxEleTyp ]  = {
   "VerCrd", "EdgVer", "TriVer", "QadVer",
   "TetVer", "PyrVer", "PriVer", "HexVer" };

static const char *MshRefStr[ GmlMaxEleTyp ]  = {
   "VerRef", "EdgRef", "TriRef", "QadRef",
   "TetRef", "PyrRef", "PriRef", "HexRef" };

static const int LenMatBas[ GmlMaxEleTyp ][ GmlMaxEleTyp ] = {
   {0,16, 8, 4,32,16,16, 8},
   {2, 0, 2, 2, 8, 8, 8, 4},
   {4, 4, 0, 0, 2, 2, 2, 0},
   {4, 4, 0, 0, 0, 2, 2, 2},
   {4, 8, 4, 0, 4, 4, 4, 0},
   {8, 8, 4, 1, 4, 8, 8, 1},
   {8,16, 2, 4, 2, 8, 8, 4},
   {8,16, 0, 8, 0, 8, 8, 8} };

static int SizMatHgh[ GmlMaxEleTyp ][ GmlMaxEleTyp ];

static const int LenMatMax[ GmlMaxEleTyp ][ GmlMaxEleTyp ] = {
   {0,64,16, 8,64,32,32,16},
   {2, 0, 2, 2,16,16,16, 8},
   {4, 4, 0, 0, 2, 2, 2, 0},
   {4, 4, 0, 0, 0, 2, 2, 2},
   {4, 8, 4, 0, 4, 4, 4, 0},
   {8, 8, 4, 1, 4, 8, 8, 1},
   {8,16, 2, 4, 2, 8, 8, 4},
   {8,16, 0, 8, 0, 8, 8, 8} };

static const int NmbTpoLnk[ GmlMaxEleTyp ][ GmlMaxEleTyp ] = {
   {1, 1, 1, 1, 1, 1, 1, 1},
   {2, 2, 1, 1, 1, 1, 1, 1},
   {3, 3, 3, 0, 2, 2, 2, 0},
   {4, 4, 0, 4, 0, 2, 2, 2},
   {4, 6, 4, 0, 4, 4, 4, 0},
   {5, 8, 4, 1, 4, 5, 5, 1},
   {6, 9, 2, 3, 2, 5, 5, 3},
   {8,12, 0, 6, 0, 6, 6, 6} };

static const int NgbTyp[8]    = {-1,0,1,1,2,3,3,3};
static const int HshLenTab[5] = {0,1,2,3,2};
static const int ItmNmbVer[8] = {1,2,3,4,4,5,6,8};
static const int ItmNmbEdg[8] = {0,1,3,4,6,8,9,12};

static const int ItmEdgNod[8][12][2] = {
{ {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {1,2}, {2,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {1,2}, {2,3}, {3,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {1,2}, {2,0}, {3,0}, {3,1}, {3,2}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {3,2}, {0,3}, {1,2}, {0,4}, {1,4}, {2,4}, {3,4}, {0,0}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {1,2}, {2,0}, {3,4}, {4,5}, {5,3}, {0,3}, {1,4}, {2,5}, {0,0}, {0,0}, {0,0} },
{ {0,1}, {3,2}, {7,6}, {4,5}, {0,3}, {4,7}, {5,6}, {1,2}, {0,4}, {1,5}, {2,6}, {3,7} } };

static const int ItmNmbFac[8] = {0,0,1,1,4,5,5,6};
static const int ItmNmbTri[8] = {0,0,1,0,4,4,2,0};
static const int ItmNmbQad[8] = {0,0,0,1,0,1,3,6};

static const int ItmFacDeg[8][6] = { 
{0,0,0,0,0,0}, {0,0,0,0,0,0}, {3,0,0,0,0,0}, {4,0,0,0,0,0},
{3,3,3,3,0,0}, {3,3,3,3,4,0}, {3,3,4,4,4,0}, {4,4,4,4,4,4} };

static const int ItmFacNod[8][6][4] = {
{ {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
{ {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
{ {0,1,2,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
{ {0,1,2,3}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
{ {1,2,3,0}, {2,0,3,0}, {3,0,1,0}, {0,2,1,0}, {0,0,0,0}, {0,0,0,0} },
{ {0,1,4,0}, {1,2,4,0}, {2,3,4,0}, {3,0,4,0}, {3,2,1,0}, {0,0,0,0} },
{ {0,2,1,0}, {3,4,5,0}, {0,1,4,3}, {1,2,5,4}, {3,5,2,0}, {0,0,0,0} },
{ {0,4,7,3}, {1,2,6,5}, {0,1,5,4}, {3,7,6,2}, {0,3,2,1}, {4,5,6,7} } };

char *sep="\n\n############################################################\n";


/*----------------------------------------------------------------------------*/
/* Init device, context and queue                                             */
/*----------------------------------------------------------------------------*/

size_t GmlInit(int DevIdx)
{
   char           str[1024];
   int            err, res;
   cl_platform_id PlfTab[ GmlMaxOclTyp ];
   cl_uint        NmbPlf;
   GmlSct         *gml;
   size_t         GmlIdx, retSiz;

   // Select which device to run on
   gml = calloc(1, sizeof(GmlSct));
   assert(gml);
   GmlIdx = (size_t)gml;
   gml->CurDev = DevIdx;

   // Init the OpenCL software platform
   res = clGetPlatformIDs(10, PlfTab, &NmbPlf);

   if(res != CL_SUCCESS)
   {
      puts("Could not find a valid platform to compile and run OpenCL sources.");
      return(0);
   }

   // Get the list of available OpenCL devices
   res = clGetDeviceIDs(PlfTab[0], CL_DEVICE_TYPE_ALL,
                        MaxGpu, gml->device_id, &gml->NmbDev);

   if(res != CL_SUCCESS)
   {
      puts("Could not get the OpenCL devices list.");
      return(0);
   }

   // Check the user choosen device index against the bounds
   if( (DevIdx < 0) || (DevIdx >= gml->NmbDev) )
   {
      printf("Selected device Id is out of bounds (1 -> %d)\n", gml->NmbDev - 1);
      return(0);
   }

   // Create the context based on the platform and the selected device
   gml->context = clCreateContext(0, 1, &gml->device_id[ gml->CurDev ],
                                 NULL, NULL, &err);

   if(!gml->context)
   {
      printf("OpenCL context creation failed with error: %d\n", err);
      return(0);
   }

   // Create a command queue for this platform and device
   gml->queue = clCreateCommandQueue(gml->context, gml->device_id[ gml->CurDev ],
                                    CL_QUEUE_PROFILING_ENABLE, &err);

   if(!gml->queue)
   {
      printf("OpenCL command queue creation failed with error: %d\n", err);
      return(0);
   }

   err = clGetDeviceInfo(gml->device_id[ gml->CurDev ], CL_DEVICE_EXTENSIONS, 1024, str, &retSiz);

   if(strstr(str, "cl_khr_fp64"))
      gml->DblExt = 1;


   // Return a pointer on the allocated and initialize GMlib structure
   return(GmlIdx);
}


/*----------------------------------------------------------------------------*/
/* Free OpenCL buffers and close the library                                  */
/*----------------------------------------------------------------------------*/

void GmlStop(size_t GmlIdx)
{
   int i;
   GETGMLPTR(gml, GmlIdx);

   // Free GPU memories, kernels and queue
   for(i=1;i<=GmlMaxDat;i++)
      if(gml->dat[i].GpuMem)
         clReleaseMemObject(gml->dat[i].GpuMem);

   for(i=1;i<=gml->NmbKrn;i++)
   {
      clReleaseKernel(gml->krn[i].kernel);
      clReleaseProgram(gml->krn[i].program);
   }

   clReleaseCommandQueue(gml->queue); 
   clReleaseContext(gml->context);
}


/*----------------------------------------------------------------------------*/
/* Print all available OpenCL capable GPUs                                    */
/*----------------------------------------------------------------------------*/

void GmlListGPU()
{
   int            i, res;
   size_t         GpuNamSiz;
   char           GpuNam[ GmlMaxStrSiz ];
   cl_platform_id PlfTab[ GmlMaxOclTyp ];
   cl_device_id   device_id[ MaxGpu ];
   cl_uint        NmbPlf, num_devices;

   res = clGetPlatformIDs(10, PlfTab, &NmbPlf);

   if(res != CL_SUCCESS)
   {
      printf("Getting the platform ID failed with error %d\n", res);
      return;
   }

   res = clGetDeviceIDs(PlfTab[0], CL_DEVICE_TYPE_ALL, MaxGpu,
                        device_id, &num_devices);

   if(res != CL_SUCCESS)
   {
      printf("Getting the list of GPU failed with error %d\n", res);
      return;
   }

   for(i=0;i<num_devices;i++)
      if(clGetDeviceInfo(  device_id[i], CL_DEVICE_NAME,  GmlMaxStrSiz , GpuNam,
                           &GpuNamSiz) == CL_SUCCESS )
      {
         printf("      %d      : %s\n", i, GpuNam);
      }
}


/*----------------------------------------------------------------------------*/
/* Allocate one of the 8 mesh data types                                      */
/*----------------------------------------------------------------------------*/

void *GmlNewParameters(size_t GmlIdx, int siz, char *src)
{
   int      idx;
   DatSct   *dat;
   GETGMLPTR(gml, GmlIdx);

   if(!(idx = GetNewDatIdx(gml)))
      return(0);

   dat = &gml->dat[ idx ];

   dat->AloTyp = GmlArgDat;
   dat->MshTyp = 0;
   dat->LnkTyp = 0;
   dat->MemAcs = GmlInout;
   dat->ItmTyp = 0;
   dat->NmbItm = 1;
   dat->ItmSiz = siz;
   dat->ItmLen = 0;
   dat->NmbLin = 1;
   dat->LinSiz = dat->NmbItm * dat->ItmSiz;
   dat->MemSiz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = "GmlParSct";
   dat->src    = src;

   if(!NewData(gml, dat))
      return(0);

   gml->ParIdx = idx;

   if(gml->DbgFlg)
   {
      puts(sep);
      printf(  "Allocated a parameters data: index=%2d, size=%zu bytes\n",
               idx, dat->MemSiz );
   }

   return((void *)dat->CpuMem);
}


/*----------------------------------------------------------------------------*/
/* Allocate one of the 8 mesh data types                                      */
/*----------------------------------------------------------------------------*/

int GmlNewMeshData(size_t GmlIdx, int MshTyp, int NmbLin)
{
   int      EleIdx, RefIdx;
   DatSct   *EleDat, *RefDat;

   GETGMLPTR(gml, GmlIdx);
   CHKELETYP(MshTyp);

   if(!(EleIdx = GetNewDatIdx(gml)))
      return(0);

   EleDat = &gml->dat[ EleIdx ];

   EleDat->AloTyp = GmlEleDat;
   EleDat->MshTyp = MshTyp;
   EleDat->LnkTyp = 0;
   EleDat->MemAcs = GmlInout;
   EleDat->ItmTyp = MshItmTyp[ MshTyp ];
   EleDat->NmbItm = 1;
   EleDat->ItmSiz = OclTypSiz[ EleDat->ItmTyp ];
   EleDat->ItmLen = TypVecSiz[ EleDat->ItmTyp ];
   EleDat->NmbLin = NmbLin;
   EleDat->LinSiz = EleDat->NmbItm * EleDat->ItmSiz;
   EleDat->MemSiz = (size_t)EleDat->NmbLin * (size_t)EleDat->LinSiz;
   EleDat->GpuMem = EleDat->CpuMem = NULL;
   EleDat->nam    = MshTypStr[ MshTyp ];

   if(!NewData(gml, EleDat))
      return(0);

   if(!(RefIdx = GetNewDatIdx(gml)))
      return(0);

   RefDat = &gml->dat[ RefIdx ];

   RefDat->AloTyp = GmlRefDat;
   RefDat->MshTyp = MshTyp;
   RefDat->LnkTyp = 0;
   RefDat->MemAcs = GmlInout;
   RefDat->ItmTyp = GmlInt;
   RefDat->NmbItm = 1;
   RefDat->ItmSiz = OclTypSiz[ GmlInt ];
   RefDat->ItmLen = 0;
   RefDat->NmbLin = NmbLin;
   RefDat->LinSiz = RefDat->NmbItm * RefDat->ItmSiz;
   RefDat->MemSiz = (size_t)RefDat->NmbLin * (size_t)RefDat->LinSiz;
   RefDat->GpuMem = RefDat->CpuMem = NULL;
   RefDat->nam    = MshRefStr[ MshTyp ];

   if(!NewData(gml, RefDat))
      return(0);

   gml->NmbEle[ MshTyp ] = NmbLin;
   gml->TypIdx[ MshTyp ] = EleIdx;
   gml->LnkMat[ MshTyp ][ GmlVertices ] = EleIdx;
   gml->CntMat[ MshTyp ][ MshTyp ] = RefIdx;
   gml->RefIdx[ MshTyp ] = RefIdx;

   if(gml->DbgFlg)
   {
      puts(sep);
      printf(  "Allocated mesh data: index=%2d, size=%zu bytes\n",
               EleIdx, EleDat->MemSiz );
      printf(  "Allocated ref  data: index=%2d, size=%zu bytes\n",
               RefIdx, RefDat->MemSiz );
   }

   return(EleIdx);
}


/*----------------------------------------------------------------------------*/
/* Allocate a free solution/raw data associated with a mesh data type         */
/*----------------------------------------------------------------------------*/

int GmlNewSolutionData( size_t GmlIdx, int MshTyp, int NmbDat,
                                       int ItmTyp, char *nam )
{
   int      idx;
   DatSct   *dat;

   GETGMLPTR(gml, GmlIdx);
   CHKELETYP(MshTyp);
   CHKOCLTYP(ItmTyp);

   if(!(idx = GetNewDatIdx(gml)))
      return(0);

   dat = &gml->dat[ idx ];

   dat->AloTyp = GmlRawDat;
   dat->MshTyp = MshTyp;
   dat->LnkTyp = 0;
   dat->MemAcs = GmlInout;
   dat->ItmTyp = ItmTyp;
   dat->NmbItm = NmbDat;
   dat->ItmSiz = OclTypSiz[ ItmTyp ];
   dat->ItmLen = TypVecSiz[ ItmTyp ];
   dat->NmbLin = gml->NmbEle[ MshTyp ];
   dat->LinSiz = dat->NmbItm * dat->ItmSiz;
   dat->MemSiz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(gml, dat))
      return(0);

   if(gml->DbgFlg)
   {
      puts(sep);
      printf(  "Allocated solutions data: index=%2d, size=%zu bytes\n",
               idx, dat->MemSiz );
   }

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Create an arbitrary link table between two kinds of mesh data types        */
/*----------------------------------------------------------------------------*/

int GmlNewLinkData(size_t GmlIdx, int MshTyp, int LnkTyp, int NmbDat, char *nam)
{
   int      LnkIdx, VecCnt, VecSiz, ItmTyp;
   DatSct   *dat;

   GETGMLPTR(gml, GmlIdx);
   CHKELETYP(MshTyp);
   CHKELETYP(LnkTyp);

   if(!(LnkIdx = GetNewDatIdx(gml)))
      return(0);

   GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

   dat = &gml->dat[ LnkIdx ];

   dat->AloTyp = GmlLnkDat;
   dat->MshTyp = MshTyp;
   dat->LnkTyp = LnkTyp;
   dat->MemAcs = GmlInout;
   dat->ItmTyp = ItmTyp;
   dat->NmbItm = VecCnt;
   dat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
   dat->ItmLen = 1;
   dat->NmbLin = gml->NmbEle[ MshTyp ];
   dat->LinSiz = dat->NmbItm * dat->ItmSiz;
   dat->MemSiz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(gml, dat))
      return(0);

   gml->LnkMat[ MshTyp ][ LnkTyp ] = LnkIdx;

   if(gml->DbgFlg)
   {
      puts(sep);
      printf(  "Allocated link data: index=%2d, size=%zu bytes\n",
               LnkIdx, dat->MemSiz );
   }

   return(LnkIdx);
}


/*----------------------------------------------------------------------------*/
/* Build an arbitray element kind and dimension link table                    */
/*----------------------------------------------------------------------------*/

int NewBallData(GmlSct *gml, int SrcTyp, int DstTyp, char *BalNam, char *DegNam)
{
   int         i, j, k, idx, cod, cpt, tmp, dir, HshKey, ItmTab[3];
   int         BalIdx, HghIdx, DegIdx, VerIdx, SrcIdx, DstIdx;
   int         EleSiz, VecSiz, BalSiz, MaxSiz, HghSiz = 0;
   int         *EleTab, *BalTab, *DegTab, *HghTab;
   int         MaxDeg = 0, MaxPos = 0, DegTot = 0, VecCnt, ItmTyp, NmbDat;
   int         SrcNmbItm, SrcLen, DstNmbItm, DstLen, *SrcNod, *DstNod, *EleNod;
   const char  *SrcNam, *DstNam;
   DatSct      *src, *dst, *bal, *hgh, *deg, *BalDat, *HghDat, *DegDat;
   BucSct      *buc;
   HshTabSct   lnk;

   // Get and check the source and destination mesh datatypes
   CHKELETYP(SrcTyp);
   CHKELETYP(DstTyp);
   memset(&lnk, 0, sizeof(HshTabSct));

   src = &gml->dat[ gml->TypIdx[ SrcTyp ] ];
   dst = &gml->dat[ gml->TypIdx[ DstTyp ] ];

   SrcNod = (int *)src->CpuMem;
   DstNod = (int *)dst->CpuMem;

   if(gml->DbgFlg)
   {
      SrcNam = BalTypStr[ SrcTyp ];
      DstNam = BalTypStr[ DstTyp ];
   }

   // Select the link direction:
   // +1 = uplink (ball or shell)
   //  0 = intra link (neighbours)
   // -1 = down link (list of edges or faces)
   if(SrcTyp < DstTyp)
   {
      dir = 1;
      lnk.HshTyp = SrcTyp;
      lnk.TabSiz = src->NmbLin;

      if(gml->DbgFlg)
         printf("Building up link %s -> %s\n", SrcNam, DstNam);
   }
   else if(SrcTyp > DstTyp)
   {
      dir = -1;
      lnk.HshTyp = DstTyp;
      lnk.TabSiz = dst->NmbLin;

      if(gml->DbgFlg)
         printf("Building down link %s -> %s\n", SrcNam, DstNam);
   }
   else
   {
      dir = 0;
      lnk.HshTyp = NgbTyp[ SrcTyp ];
      lnk.TabSiz = src->NmbLin * NmbTpoLnk[ SrcTyp ][ SrcTyp ] / 2;

      if(gml->DbgFlg)
         printf(  "Building %s neighbours through %s\n",
                  SrcNam, BalTypStr[ NgbTyp[ SrcTyp ] ] );
   }

   // Setup a hash table
   lnk.DatLen = ItmNmbVer[ lnk.HshTyp ];
   lnk.KeyLen = HshLenTab[ lnk.DatLen ];
   lnk.NmbDat = lnk.TabSiz;
   lnk.NxtDat = 1;
   lnk.HshTab = calloc(lnk.TabSiz, sizeof(int));
   lnk.DatTab = malloc(lnk.TabSiz * sizeof(BucSct));

   if(gml->DbgFlg)
      printf(  "Hash table: lines=%lld, stored items=%d, hash keys=%d\n",
               lnk.TabSiz, lnk.DatLen, lnk.KeyLen);

   SrcNmbItm = NmbTpoLnk[ SrcTyp ][ lnk.HshTyp ];
   DstNmbItm = NmbTpoLnk[ DstTyp ][ lnk.HshTyp ];

   SrcLen = ItmNmbVer[ SrcTyp ];
   DstLen = ItmNmbVer[ DstTyp ];

   // Add destination entities to the hash table
   for(i=0;i<dst->NmbLin;i++)
   {
      EleNod = &DstNod[ i * DstLen ];

      for(j=0;j<DstNmbItm;j++)
      {
         GetItmNod(EleNod, DstTyp, lnk.HshTyp, j, ItmTab);
         HshKey = CalHshKey(&lnk, ItmTab);
         AddHsh(&lnk, HshKey, i, j, ItmTab);
      }
   }

   if(gml->DbgFlg)
      printf(  "Hashed %lld entities: occupency=%lld%%, collisions=%g\n",
               lnk.NmbDat, (100LL * lnk.NmbHit) / lnk.TabSiz,
               (double)lnk.NmbMis / (double)lnk.TabSiz );

   // Allocate and fill a GPU data type to store the degrees in case uf uplink
   if(dir == 1)
   {
      // Firsdt allocate the GPU datatype
      if(!(DegIdx = GetNewDatIdx(gml)))
         return(0);

      DegDat = &gml->dat[ DegIdx ];

      DegDat->AloTyp = GmlLnkDat;
      DegDat->MshTyp = SrcTyp;
      DegDat->LnkTyp = DstTyp;
      DegDat->MemAcs = GmlInout;
      DegDat->ItmTyp = GmlInt;
      DegDat->NmbItm = 1;
      DegDat->ItmSiz = OclTypSiz[ GmlInt ];
      DegDat->ItmLen = 0;
      DegDat->NmbLin = gml->NmbEle[ SrcTyp ];
      DegDat->LinSiz = DegDat->NmbItm * DegDat->ItmSiz;
      DegDat->MemSiz = (size_t)DegDat->NmbLin * (size_t)DegDat->LinSiz;
      DegDat->GpuMem = DegDat->CpuMem = NULL;
      DegDat->nam    = DegNam;

      if(!NewData(gml, DegDat))
         return(0);

      if(gml->DbgFlg)
         printf("Allocate a degree table with %d lines\n", DegDat->NmbLin);

      // Then fetch the degrees from the hash table
      gml->CntMat[ SrcTyp ][ DstTyp ] = DegIdx;
      deg = &gml->dat[ gml->CntMat[ src->MshTyp ][ dst->MshTyp ] ];
      DegTab = deg->CpuMem;

      for(i=0;i<src->NmbLin;i++)
      {
         if(SrcTyp == GmlVertices)
            EleNod = &i;
         else
            EleNod = &SrcNod[ i * SrcLen ];

         for(j=0;j<SrcNmbItm;j++)
         {
            idx = i * SrcNmbItm + j;
            GetItmNod(EleNod, SrcTyp, lnk.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&lnk, ItmTab);
            DegTab[ idx ] = GetHsh(&lnk, HshKey, i, j, ItmTab, NULL);
         }
      }

      EleTab = dst->CpuMem;
      BalSiz = LenMatBas[ src->MshTyp ][ dst->MshTyp ];
      MaxSiz = LenMatMax[ src->MshTyp ][ dst->MshTyp ];

      for(i=0;i<src->NmbLin;i++)
      {
         if(!MaxPos && (DegTab[i] > BalSiz))
            MaxPos = i;

         DegTot += DegTab[i];
         MaxDeg = MAX(MaxDeg, DegTab[i]);
      }

      MaxDeg = pow(2., ceil(log2(MaxDeg)));
      HghSiz = MIN(MaxDeg, LenMatMax[ src->MshTyp ][ dst->MshTyp ]);
      SizMatHgh[ src->MshTyp ][ dst->MshTyp ] = HghSiz;

      if(gml->DbgFlg)
         printf(  "Width for lines 1..%d: %d, for lines %d..%d:%d\n",
                  MaxPos, BalSiz, MaxPos+1, src->NmbLin, MaxDeg);
   }

   // Build downlinks and neighbours
   if(dir == -1 || dir == 0)
   {
      // Allocate the downlink table
      if(!(BalIdx = GetNewDatIdx(gml)))
         return(0);

      NmbDat = LenMatBas[ SrcTyp ][ DstTyp ];
      GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

      BalDat = &gml->dat[ BalIdx ];

      BalDat->AloTyp = GmlLnkDat;
      BalDat->MshTyp = SrcTyp;
      BalDat->LnkTyp = DstTyp;
      BalDat->MemAcs = GmlInout;
      BalDat->ItmTyp = ItmTyp;
      BalDat->NmbItm = VecCnt;
      BalDat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
      BalDat->ItmLen = 1;
      BalDat->NmbLin = src->NmbLin;
      BalDat->LinSiz = BalDat->NmbItm * BalDat->ItmSiz;
      BalDat->MemSiz = (size_t)BalDat->NmbLin * (size_t)BalDat->LinSiz;
      BalDat->GpuMem = BalDat->CpuMem = NULL;
      BalDat->nam    = BalNam;

      if(!NewData(gml, BalDat))
         return(0);

      // fetch the pointed items from the hash table and store them as downlinks
      gml->LnkMat[ SrcTyp ][ DstTyp ] = BalIdx;
      BalTab = bal->CpuMem;

      for(i=0;i<src->NmbLin;i++)
      {
         EleNod = &SrcNod[ i * SrcLen ];

         for(j=0;j<SrcNmbItm;j++)
         {
            idx = i * SrcNmbItm + j;
            GetItmNod(EleNod, SrcTyp, lnk.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&lnk, ItmTab);

            if((cpt = GetHsh(&lnk, HshKey, i, j, ItmTab, &cod)))
            {
               if(dir == 1)
                  BalTab[ i * BalDat->ItmSiz + j ] = cod >> 4;
               else
               {
                  if(cpt != 2)
                     BalTab[ i * BalDat->ItmSiz + j ] = 0;
                  else if( ((cod >> 4) != i) || ((cod & 7) != j) )
                     BalTab[ i * BalDat->ItmSiz + j ] = cod;
               }
            }
         }
      }

      // Upload the ball data to th GPU memory
      gml->MovSiz += UploadData(gml, BalIdx);

      printf("Stored %d uniq entries in the link table\n", src->NmbLin * SrcNmbItm);
   }
   else if(dir == 1) // More complex case: balls and shells
   {
      // Allocate the base vector ball table
      if(!(BalIdx = GetNewDatIdx(gml)))
         return(0);

      NmbDat = LenMatBas[ SrcTyp ][ DstTyp ];
      GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

      BalDat = &gml->dat[ BalIdx ];

      BalDat->AloTyp = GmlLnkDat;
      BalDat->MshTyp = SrcTyp;
      BalDat->LnkTyp = DstTyp;
      BalDat->MemAcs = GmlInout;
      BalDat->ItmTyp = ItmTyp;
      BalDat->NmbItm = VecCnt;
      BalDat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
      BalDat->ItmLen = 1;
      BalDat->NmbLin = MaxPos+1;
      BalDat->LinSiz = BalDat->NmbItm * BalDat->ItmSiz;
      BalDat->MemSiz = (size_t)BalDat->NmbLin * (size_t)BalDat->LinSiz;
      BalDat->GpuMem = BalDat->CpuMem = NULL;
      BalDat->nam    = BalNam;

      if(!NewData(gml, BalDat))
         return(0);

      if(gml->DbgFlg)
         printf(  "Allocate a base table with %d lines of %d width vectors\n",
                  BalDat->NmbLin, NmbDat);

      gml->LnkMat[ SrcTyp ][ DstTyp ] = BalIdx;
      bal = &gml->dat[ gml->LnkMat[ src->MshTyp ][ dst->MshTyp ] ];

      // Allocate the high vector ball table
      if(!(HghIdx = GetNewDatIdx(gml)))
         return(0);

      NmbDat = SizMatHgh[ SrcTyp ][ DstTyp ];
      GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

      HghDat = &gml->dat[ HghIdx ];

      HghDat->AloTyp = GmlLnkDat;
      HghDat->MshTyp = SrcTyp;
      HghDat->LnkTyp = DstTyp;
      HghDat->MemAcs = GmlInout;
      HghDat->ItmTyp = ItmTyp;
      HghDat->NmbItm = VecCnt;
      HghDat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
      HghDat->ItmLen = 1;
      HghDat->NmbLin = gml->NmbEle[ SrcTyp ] - MaxPos;
      HghDat->LinSiz = HghDat->NmbItm * HghDat->ItmSiz;
      HghDat->MemSiz = (size_t)HghDat->NmbLin * (size_t)HghDat->LinSiz;
      HghDat->GpuMem = HghDat->CpuMem = NULL;
      HghDat->nam    = BalNam;

      if(!NewData(gml, HghDat))
         return(0);

      if(gml->DbgFlg)
         printf(  "Allocate a hash table with %d lines of %d width vectors\n",
                  HghDat->NmbLin, NmbDat);

      gml->LnkHgh[ SrcTyp ][ DstTyp ] = HghIdx;
      hgh = &gml->dat[ gml->LnkHgh[ src->MshTyp ][ dst->MshTyp ] ];

      // Fill both ball tables at the same time
      BalTab = bal->CpuMem;
      HghTab = hgh->CpuMem;

      if(gml->DbgFlg)
         puts("Fetching balls from the hash table and filling both low and high degree tables");

      for(i=0;i<src->NmbLin;i++)
      {
         if(SrcTyp == GmlVertices)
            EleNod = &i;
         else
            EleNod = &SrcNod[ i * SrcLen ];

         GetItmNod(EleNod, SrcTyp, lnk.HshTyp, 0, ItmTab);
         HshKey = CalHshKey(&lnk, ItmTab);

         if(i <= MaxPos)
            GetHsh(&lnk, HshKey, i, 0, ItmTab, &BalTab[ i * BalSiz ]);
         else
            GetHsh(&lnk, HshKey, i, 0, ItmTab, &HghTab[ (i - MaxPos) * HghSiz ]);
      }

      if(gml->DbgFlg)
         puts("Uploading the three tables");

      // Upload the ball data to th GPU memory
      gml->MovSiz += UploadData(gml, BalIdx);
      gml->MovSiz += UploadData(gml, HghIdx);
      gml->MovSiz += UploadData(gml, DegIdx);

      if(gml->DbgFlg)
      {
         puts(sep);
         printf(  "Ball generation: type %s -> %s\n",
                  BalTypStr[ SrcTyp ], BalTypStr[ DstTyp ] );
         printf(  "low degree ranging from 1 to %d, occupency = %g%%\n",
                  MaxPos, (float)(100 * DegTot) / (float)(MaxPos * BalSiz) );
         printf(  "high degree entities = %d\n", src->NmbLin - MaxPos);
         printf(  "Allocated degree     data: index=%2d, size=%zu bytes\n",
                  DegIdx, DegDat->MemSiz );
         printf(  "Allocated short ball data: index=%2d, size=%zu bytes\n",
                  BalIdx, BalDat->MemSiz );
         printf(  "Allocated long ball  data: index=%2d, size=%zu bytes\n",
                  HghIdx, HghDat->MemSiz );
      }
   }

   free(lnk.HshTab);
   free(lnk.DatTab);

   return(0);
}


/*----------------------------------------------------------------------------*/
/* Extract an element's entity node table                                     */
/*----------------------------------------------------------------------------*/

static void GetItmNod(  int *EleNod, int EleTyp,
                        int ItmTyp, int ItmIdx, int *ItmTab )
{
   int i, j, tmp, NmbVer = ItmNmbVer[ ItmTyp ];

   // Copy the hashed entity's vertices to the return table
   switch(ItmTyp)
   {
      case GmlVertices :
      {
         ItmTab[0] = EleNod[ ItmIdx ];
      }return;

      case GmlEdges :
      {
         i = EleNod[ ItmEdgNod[ EleTyp ][ ItmIdx ][0] ];
         j = EleNod[ ItmEdgNod[ EleTyp ][ ItmIdx ][1] ];
         ItmTab[0] = MIN(i,j);
         ItmTab[1] = MAX(i,j);
      }return;

      case GmlTriangles :
      {
         for(i=0;i<3;i++)
            ItmTab[i] = EleNod[ ItmFacNod[ EleTyp ][ ItmIdx ][i] ];

         // Sort the vertices to speed-up further comparison
         for(i=0;i<3;i++)
            for(j=i+1;j<3;j++)
               if(ItmTab[i] > ItmTab[j])
               {
                  tmp       = ItmTab[i];
                  ItmTab[i] = ItmTab[j];
                  ItmTab[j] = tmp;
               }
      }return;

      case GmlQuadrilaterals :
      {
         for(i=0;i<4;i++)
            ItmTab[i] = EleNod[ ItmFacNod[ EleTyp ][ ItmIdx ][i] ];

         // Sort the vertices to speed-up further comparison
         for(i=0;i<4;i++)
            for(j=i+1;j<4;j++)
               if(ItmTab[i] > ItmTab[j])
               {
                  tmp       = ItmTab[i];
                  ItmTab[i] = ItmTab[j];
                  ItmTab[j] = tmp;
               }
      }return;
   }
}


/*----------------------------------------------------------------------------*/
/* Compute a hash key based on a node table and a hash table size             */
/*----------------------------------------------------------------------------*/

static int CalHshKey(HshTabSct *lnk, int *ItmTab)
{
   int i, k = 0, wei[4] = {3,5,7,11};

   if(lnk->DatLen == 1)
      return(ItmTab[0]);

   for(i=0;i<lnk->DatLen;i++)
      k += wei[i] * ItmTab[i];

   return(k % lnk->TabSiz);
}


/*----------------------------------------------------------------------------*/
/* Add a pair element/entity to the hash table and handle any collision       */
/*----------------------------------------------------------------------------*/

static void AddHsh(  HshTabSct *lnk, int HshKey, int EleIdx,
                     int ItmIdx, int *ItmTab )
{
   int i, nxt;
   BucSct *buc;
   
   if(lnk->NxtDat == lnk->NmbDat)
   {
      lnk->NmbDat *= 2;
      lnk->DatTab = realloc(lnk->DatTab, lnk->NmbDat * sizeof(BucSct));
   }

   nxt = lnk->HshTab[ HshKey ];
   lnk->HshTab[ HshKey ] = lnk->NxtDat;
   buc = &lnk->DatTab[ lnk->NxtDat ];
   buc->EleIdx = EleIdx;
   buc->ItmIdx = ItmIdx;
   buc->NxtDat = nxt;
   lnk->NxtDat++;

   if(nxt)
      lnk->NmbMis++;
   else
      lnk->NmbHit++;

   for(i=0;i<lnk->DatLen;i++)
      buc->nod[i] = ItmTab[i];
}


/*----------------------------------------------------------------------------*/
/* Fetch an element/entity pair from the hash table                           */
/*----------------------------------------------------------------------------*/

static int GetHsh(   HshTabSct *lnk, int HshKey, int EleIdx,
                     int ItmIdx, int *ItmTab, int *UsrTab )
{
   int i, flg, deg = 0;
   BucSct *buc;

   if(!lnk->HshTab[ HshKey ])
      return(0);

   buc = &lnk->DatTab[ lnk->HshTab[ HshKey ] ];

   while(buc)
   {
      flg = 1;

      for(i=0;i<lnk->DatLen;i++)
         if(buc->nod[i] != ItmTab[i])
         {
            flg = 0;
            break;
         }

      if(flg)
      {
         if(UsrTab)
            UsrTab[ deg ] = (buc->EleIdx << 4) | buc->ItmIdx;

         deg++;
      }

      if(!buc->NxtDat)
         return(deg);

      buc = &lnk->DatTab[ buc->NxtDat ];
   }

   return(0);
}


/*----------------------------------------------------------------------------*/
/* Find and return a free data slot int the GML structure                     */
/*----------------------------------------------------------------------------*/

static int GetNewDatIdx(GmlSct *gml)
{
   for(int i=1;i<=GmlMaxDat;i++)
      if(!gml->dat[i].use)
      {
         gml->dat[i].use = 1;
         return(i);
      }

   return(0);
}


/*----------------------------------------------------------------------------*/
/* Allocate an OpenCL buffer plus 10% more for resizing                       */
/*----------------------------------------------------------------------------*/

static int NewData(GmlSct *gml, DatSct *dat)
{
   int MemAcs[4] = {0, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE};

   // Allocate the requested memory size on the GPU
   dat->GpuMem = clCreateBuffer( gml->context, MemAcs[ dat->MemAcs ],
                                 dat->MemSiz, NULL, NULL );

   if(!dat->GpuMem)
   {
      printf(  "Cannot allocate %ld MB on the GPU (%ld MB already used)\n",
               dat->MemSiz / MB, GmlGetMemoryUsage((size_t)gml) / MB);
      return(0);
   }

   // Allocate the requested memory size on the CPU side
   if( (dat->MemAcs != GmlInternal) && !(dat->CpuMem = calloc(1, dat->MemSiz)) )
   {
      printf("Cannot allocate %ld MB on the CPU\n", dat->MemSiz/MB);
      return(0);
   }

   // Keep track of allocated memory
   gml->MemSiz += dat->MemSiz;

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Release an OpenCL buffer                                                   */
/*----------------------------------------------------------------------------*/

int GmlFreeData(size_t GmlIdx, int idx)
{
   GETGMLPTR(gml, GmlIdx);
   DatSct *dat = &gml->dat[ idx ];

   // Free both GPU and CPU memory buffers
   if( (idx >= 1) && (idx <= GmlMaxDat) && dat->GpuMem )
   {
      if(clReleaseMemObject(dat->GpuMem) != CL_SUCCESS)
         return(0);

      dat->GpuMem = NULL;
      dat->use = 0;
      gml->MemSiz -= dat->MemSiz;

      if(dat->CpuMem)
         free(dat->CpuMem);

      return(1);
   }
   else
      return(0);
}


/*----------------------------------------------------------------------------*/
/* Set a line of mesh, link or solution data                                  */
/*----------------------------------------------------------------------------*/

int GmlSetDataLine(size_t GmlIdx, int idx, int lin, ...)
{
   GETGMLPTR(gml, GmlIdx);
   DatSct   *dat = &gml->dat[ idx ], *RefDat;
   char     *adr = (void *)dat->CpuMem;
   int      i, *EleTab, siz, *RefTab, *tab;
   float    *CrdTab;
   va_list  VarArg;

   va_start(VarArg, lin);

   if(dat->AloTyp == GmlRawDat)
   {
      memcpy(&adr[ lin * dat->LinSiz ], va_arg(VarArg, void *), dat->LinSiz);
   }
   else if(dat->AloTyp == GmlLnkDat)
   {
      tab = (int *)dat->CpuMem;

      for(i=0;i<dat->NmbItm;i++)
         tab[ lin * dat->NmbItm + i ] = va_arg(VarArg, int);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp == GmlVertices) )
   {
      CrdTab = (float *)dat->CpuMem;
      RefDat = &gml->dat[ gml->CntMat[ dat->MshTyp ][ dat->MshTyp ] ];
      RefTab = (int *)RefDat->CpuMem;
      siz = 4;

      for(i=0;i<3;i++)
         CrdTab[ lin * siz + i ] = va_arg(VarArg, double);

      CrdTab[ lin * siz + 3 ] = 0.;
      RefTab[ lin ] = va_arg(VarArg, int);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp > GmlVertices) )
   {
      EleTab = (int *)dat->CpuMem;
      siz = TypVecSiz[ MshItmTyp[ dat->MshTyp ] ];
      RefDat = &gml->dat[ gml->CntMat[ dat->MshTyp ][ dat->MshTyp ] ];
      RefTab = (int *)RefDat->CpuMem;

      for(i=0;i<EleNmbNod[ dat->MshTyp ];i++)
         EleTab[ lin * siz + i ] = va_arg(VarArg, int);

      RefTab[ lin ] = va_arg(VarArg, int);
   }

   va_end(VarArg);

   if(lin == dat->NmbLin - 1)
      gml->MovSiz += UploadData(gml, idx);

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Get a line of solution or vertex corrdinates from the librarie's buffers   */
/*----------------------------------------------------------------------------*/

int GmlGetDataLine(size_t GmlIdx, int idx, int lin, ...)
{
   GETGMLPTR(gml, GmlIdx);
   DatSct   *dat = &gml->dat[ idx ];
   char     *adr = (void *)dat->CpuMem;
   int      i;
   float    *GpuCrd;
   double   *UsrCrd;
   va_list  VarArg;

   if(lin == 0)
      gml->MovSiz += DownloadData(gml, idx);

   va_start(VarArg, lin);

   if(dat->AloTyp == GmlRawDat)
   {
      memcpy(va_arg(VarArg, void *), &adr[ lin * dat->LinSiz ], dat->LinSiz);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp == GmlVertices) )
   {
      GpuCrd = (float *)dat->CpuMem;

      for(i=0;i<3;i++)
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

static int UploadData(GmlSct *gml, int idx)
{
   int      res;
   DatSct   *dat = &gml->dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem
   || !dat->CpuMem || (dat->MemAcs == GmlOutput) )
   {
      return(0);
   }

   // Upload buffer from CPU ram to GPU ram
   // and keep track of the amount of uploaded data
   res = clEnqueueWriteBuffer(gml->queue, dat->GpuMem, CL_FALSE, 0,
                              dat->MemSiz, dat->CpuMem, 0, NULL,NULL);

   if(res != CL_SUCCESS)
   {
      printf("Uploading the data to the GPu failed with error %d\n", res);
      return(0);
   }
   else
   {
      gml->MovSiz += dat->MemSiz;
      return(dat->MemSiz);
   }
}


/*----------------------------------------------------------------------------*/
/* Copy an OpenCL buffer into user's data                                     */
/*----------------------------------------------------------------------------*/

static int DownloadData(GmlSct *gml, int idx)
{
   int      res;
   DatSct   *dat = &gml->dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || !dat->CpuMem )
      return(0);

   // Download buffer from GPU ram to CPU ram
   // and keep track of the amount of downloaded data
   res = clEnqueueReadBuffer( gml->queue, dat->GpuMem, CL_TRUE, 0,
                              dat->MemSiz, dat->CpuMem, 0, NULL, NULL );

   if(res != CL_SUCCESS)
   {
      printf("Downloading the data from the GPu failed with error %d\n", res);
      return(0);
   }
   else
   {
      gml->MovSiz += dat->MemSiz;
      return(dat->MemSiz);
   }
}


/*----------------------------------------------------------------------------*/
/* Generate the kernel from user's source and data and compile it             */
/*----------------------------------------------------------------------------*/

int GmlCompileKernel(size_t GmlIdx, char *KrnSrc, char *PrcNam,
                     int MshTyp, int NmbTyp, ...)
{
   GETGMLPTR(gml, GmlIdx);
   int      i, j, flg, KrnIdx, KrnHghIdx, SrcTyp, DstTyp, NmbArg = 0, RefIdx;
   int      FlgTab[ GmlMaxDat ], IdxTab[ GmlMaxDat ];
   int      LnkTab[ GmlMaxDat ], CntTab[ GmlMaxDat ];
   int      LnkItm, NmbItm, ItmTyp, ItmLen, LnkPos, CptPos, ArgHghPos;
   int      NmbHgh, HghVec, HghSiz, HghTyp, HghArg = -1, HghIdx = -1;
   char     *ParSrc, src[ GmlMaxSrcSiz ] = "\0";
   char     BalNam[ GmlMaxStrSiz ], DegNam[ GmlMaxStrSiz ];
   va_list  VarArg;
   DatSct   *dat, *RefDat;
   ArgSct   *arg, ArgTab[ GmlMaxOclTyp ];
   KrnSct   *krn;

   // Read user's datatypes arguments
   ParSrc = gml->ParIdx ? gml->dat[ gml->ParIdx ].src : NULL;

   va_start(VarArg, NmbTyp);

   for(i=0;i<NmbTyp;i++)
   {
      IdxTab[i] = va_arg(VarArg, int);
      FlgTab[i] = va_arg(VarArg, int);
      LnkTab[i] = va_arg(VarArg, int);
      CntTab[i] = 0;
   }

   va_end(VarArg);

   // Check or build datatype indirect access tables
   for(i=0;i<NmbTyp;i++)
   {
      dat = &gml->dat[ IdxTab[i] ];

      if( LnkTab[i] || (dat->MshTyp == MshTyp) )
         continue;

      // Source and destination mesh datatype to access the connectivity matrix
      SrcTyp = MshTyp;
      DstTyp = dat->MshTyp;

      if(MshTypDim[ SrcTyp ] > MshTypDim[ DstTyp ])
      {
         // Downlink
         LnkTab[i] = gml->LnkMat[ MshTyp ][ DstTyp ];
         CntTab[i] = 0;
      }
      else if(MshTypDim[ SrcTyp ] < MshTypDim[ DstTyp ])
      {
         // Uplink
         if(!gml->LnkMat[ MshTyp ][ dat->MshTyp ])
         {
            // Generate the default link between the two kinds of entities
            sprintf(BalNam, "%s%sBal", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            sprintf(DegNam, "%s%sDeg", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            NewBallData(gml, MshTyp, DstTyp, BalNam, DegNam);
         }

         LnkTab[i] = gml->LnkMat[ MshTyp ][ DstTyp ];
         CntTab[i] = gml->CntMat[ MshTyp ][ DstTyp ];
      }
   }

   // Build the context arguments to store links
   for(i=0;i<NmbTyp;i++)
   {
      // Process each datatype that needs a link
      if(!LnkTab[i])
         continue;

      // Try to find if this link already exists in the previously defined arguments
      dat = &gml->dat[ IdxTab[i] ];
      flg = 0;

      for(j=0;j<NmbArg;j++)
         if(ArgTab[j].DatIdx == LnkTab[i])
            flg = 1;

      if(flg)
         continue;

      // If not, get the link data index from the conectivity matrix
      DstTyp = dat->MshTyp;
      LnkItm = LenMatBas[ MshTyp ][ DstTyp ];
      GetCntVec(LnkItm, &NmbItm, &ItmLen, &ItmTyp);

      // Create a new contextual arguments
      arg = &ArgTab[ NmbArg ];
      arg->ArgIdx = NmbArg;
      NmbArg++;

      if(MshTypDim[ SrcTyp ] > MshTypDim[ DstTyp ])
      {
         // Downlink access arguments
         arg->MshTyp = DstTyp;
         arg->DatIdx = LnkTab[i];
         arg->LnkDir = -1;
         arg->LnkTyp = -1;
         arg->LnkIdx = -1;
         arg->CntIdx = -1;
         arg->LnkDeg = LnkItm;
         arg->MaxDeg = LnkItm;
         arg->NmbItm = NmbItm;
         arg->ItmLen = ItmLen;
         arg->ItmTyp = ItmTyp;
         arg->FlgTab = GmlReadMode;
         arg->nam    = gml->dat[ arg->DatIdx ].nam;
      }
      else if(MshTypDim[ SrcTyp ] == MshTypDim[ DstTyp ])
      {
         // Neighbours access arguments
         arg->MshTyp = DstTyp;
         arg->DatIdx = LnkTab[i];
         arg->LnkDir = 0;
         arg->LnkTyp = -1;
         arg->LnkIdx = -1;
         arg->CntIdx = -1;
         arg->LnkDeg = LnkItm;
         arg->MaxDeg = LnkItm;
         arg->NmbItm = NmbItm;
         arg->ItmLen = ItmLen;
         arg->ItmTyp = ItmTyp;
         arg->FlgTab = GmlReadMode;
         arg->nam    = gml->dat[ arg->DatIdx ].nam;
      }
      else if(MshTypDim[ SrcTyp ] < MshTypDim[ DstTyp ])
      {
         // Uplink access arguments
         arg->MshTyp = DstTyp;
         arg->DatIdx = LnkTab[i];
         arg->LnkDir = 1;
         arg->LnkTyp = -1;
         arg->LnkIdx = -1;
         arg->CntIdx = -1;
         arg->LnkDeg = -1;
         arg->MaxDeg = LnkItm;
         arg->NmbItm = NmbItm;
         arg->ItmLen = ItmLen;
         arg->ItmTyp = ItmTyp;
         arg->FlgTab = GmlReadMode;
         arg->nam    = gml->dat[ arg->DatIdx ].nam;
         HghArg      = arg->ArgIdx;
         HghIdx      = gml->LnkHgh[ MshTyp ][ DstTyp ];

         // Variable counter argument
         arg = &ArgTab[ NmbArg ];
         arg->ArgIdx = NmbArg;
         NmbArg++;

         arg->MshTyp = DstTyp;
         arg->DatIdx = gml->CntMat[ MshTyp ][ DstTyp ];
         arg->LnkDir = 0;
         arg->LnkTyp = -1;
         arg->LnkIdx = -1;
         arg->CntIdx = -1;
         arg->LnkDeg = 1;
         arg->MaxDeg = 1;
         arg->NmbItm = 1;
         arg->ItmLen = 1;
         arg->ItmTyp = GmlInt;
         arg->FlgTab = GmlReadMode;
         arg->nam    = gml->dat[ arg->DatIdx ].nam;
      }
   }

   // Now that the links are defined first, the user's data can be accessed
   for(i=0;i<NmbTyp;i++)
   {
      // For each datatype, try to find the arguments containing their link and counter
      dat = &gml->dat[ IdxTab[i] ];
      DstTyp = dat->MshTyp;
      LnkPos = CptPos = -1;

      if(LnkTab[i])
         for(j=0;j<NmbArg;j++)
            if(ArgTab[j].DatIdx == LnkTab[i])
               LnkPos = j;

      if(CntTab[i])
         for(j=0;j<NmbArg;j++)
            if(ArgTab[j].DatIdx == CntTab[i])
               CptPos = j;

      // Create a new data argument
      arg = &ArgTab[ NmbArg ];
      arg->ArgIdx = NmbArg;
      NmbArg++;

      arg->MshTyp = DstTyp;
      arg->DatIdx = IdxTab[i];
      arg->LnkDir = 0;
      arg->LnkTyp = DstTyp;
      arg->LnkIdx = LnkPos;
      arg->CntIdx = CptPos;

      if(LnkPos != -1 && CptPos != -1)
      {
         // Downlink with variable counter
         arg->LnkDeg = -1;
         arg->MaxDeg = ArgTab[ LnkPos ].MaxDeg;
         ArgHghPos   = arg->ArgIdx;
      }
      else if(LnkPos != -1 && CptPos == -1)
      {
         // Link with constant counter
         arg->LnkDeg = ArgTab[ LnkPos ].LnkDeg;
         arg->MaxDeg = ArgTab[ LnkPos ].MaxDeg;
      }
      else
      {
         // No link: straight access
         arg->LnkDeg = -1;
         arg->MaxDeg = -1;
      }

      arg->NmbItm = dat->NmbItm;
      arg->ItmLen = dat->ItmLen;
      arg->ItmTyp = dat->ItmTyp;
      arg->FlgTab = FlgTab[i];
      arg->nam    = dat->nam;

      if(!(FlgTab[i] & GmlRefFlag) || (CptPos != -1))
         continue;

      // Create a new ref argument
      arg = &ArgTab[ NmbArg ];
      arg->ArgIdx = NmbArg;
      NmbArg++;

      RefIdx = gml->RefIdx[ DstTyp ];
      RefDat = &gml->dat[ RefIdx ];
      arg->MshTyp = DstTyp;
      arg->DatIdx = RefIdx;
      arg->LnkDir = 0;
      arg->LnkTyp = DstTyp;
      arg->LnkIdx = LnkPos;
      arg->CntIdx = CptPos;
      arg->LnkDeg = -1;
      arg->MaxDeg = ArgTab[ LnkPos ].MaxDeg;
      arg->NmbItm = RefDat->NmbItm;
      arg->ItmLen = RefDat->ItmLen;
      arg->ItmTyp = RefDat->ItmTyp;
      arg->FlgTab = FlgTab[i];
      arg->nam    = RefDat->nam;
   }

   // Generate the kernel source code
   WriteUserTypedef        (src, ParSrc);
   WriteProcedureHeader    (src, PrcNam, MshTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, MshTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, MshTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, MshTyp, NmbArg, ArgTab);

   // And Compile it
   KrnIdx = NewOclKrn      (gml, src, PrcNam);

   if(gml->DbgFlg)
   {
      puts(sep);
      printf("Generated source for kernel=%s, index=%2d\n", PrcNam, KrnIdx);
      puts(src);
   }

   // Store information usefull to the kernel: loop indices and arguments list
   krn = &gml->krn[ KrnIdx ];
   krn->NmbDat    = NmbArg;
   krn->NmbLin[1] = 0;

   // In case of constant counter, the kernel loops from 0 to NmbLin-1
   // Otherwise, it loops up to the start of high degree entities
   if(HghIdx == -1)
      krn->NmbLin[0] = gml->dat[ gml->TypIdx[ MshTyp ] ].NmbLin;
   else
      krn->NmbLin[0] = gml->dat[ gml->TypIdx[ MshTyp ] ].NmbLin - gml->dat[ HghIdx ].NmbLin;

   for(i=0;i<NmbArg;i++)
      krn->DatTab[i] = ArgTab[i].DatIdx;

   if(HghArg == -1)
      return(KrnIdx);

   // In case of uplink kernel, generate a second high degree kernel
   NmbHgh = SizMatHgh[ MshTyp ][ DstTyp ];
   GetCntVec(NmbHgh, &HghVec, &HghSiz, &HghTyp);

   // Mofify the argument containing the uplink with the high count sizes
   ArgTab[ HghArg ].DatIdx = HghIdx;
   ArgTab[ HghArg ].MaxDeg = NmbHgh;
   ArgTab[ HghArg ].NmbItm = HghVec;
   ArgTab[ HghArg ].ItmLen = HghSiz;
   ArgTab[ HghArg ].ItmTyp = HghTyp;
   src[0] = '\0';

   // Generate the kernel source code
   WriteUserTypedef        (src, ParSrc);
   WriteProcedureHeader    (src, PrcNam, MshTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, MshTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, MshTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, MshTyp, NmbArg, ArgTab);

   // And Compile it
   KrnHghIdx = NewOclKrn   (gml, src, PrcNam);
   gml->krn[ KrnIdx ].HghIdx = KrnHghIdx;

   // Store information usefull to the kernel: loop indices and arguments list
   krn = &gml->krn[ KrnHghIdx ];
   krn->NmbDat    = NmbArg;
   krn->NmbLin[0] = gml->dat[ HghIdx ].NmbLin;
   krn->NmbLin[1] = gml->dat[ gml->TypIdx[ MshTyp ] ].NmbLin - gml->dat[ HghIdx ].NmbLin;

   for(i=0;i<NmbArg;i++)
      krn->DatTab[i] = ArgTab[i].DatIdx;

   if(gml->DbgFlg)
   {
      puts(sep);
      printf("Generated source for kernel=%s, index=%2d\n", PrcNam, KrnIdx);
      puts(src);
   }

   return(KrnIdx);
}


/*----------------------------------------------------------------------------*/
/* Add the user's parameters structure definition to the source code          */
/*----------------------------------------------------------------------------*/

static void WriteUserTypedef(char *src, char *ParSrc)
{
   strcat(src, "// USER'S ARGUMENTS STRUCTURE\n");
   strcat(src, ParSrc);
   strcat(src, "\n");
}


/*----------------------------------------------------------------------------*/
/* Write the procedure name and typedef with all arguments types and names    */
/*----------------------------------------------------------------------------*/

static void WriteProcedureHeader(char *src, char *PrcNam, int MshTyp,
                                 int NmbArg, ArgSct *ArgTab)
{
   int      i;
   char     str[ GmlMaxStrSiz ];
   ArgSct   *arg;

   strcat(src, "// KERNEL HEADER\n");
   sprintf(str, "__kernel void %s(", PrcNam);
   strcat(src, str);

   for(i=0;i<NmbArg;i++)
   {
      arg = &ArgTab[i];

      sprintf(str,  "\n   __global %s ", OclTypStr[ arg->ItmTyp ]);
      strcat(src, str);

      if(arg->NmbItm <= 1)
         sprintf(str, "*%sTab,", arg->nam);
      else
         sprintf(str, "(*%sTab)[%d],", arg->nam, arg->NmbItm);

      strcat(src, str);
   }

   strcat(src, "\n   __global GmlParSct *GmlPar,");
   strcat(src, "\n   const    int2       count )\n{\n");
}


/*----------------------------------------------------------------------------*/
/* Write definition of automatic local variables                              */
/*----------------------------------------------------------------------------*/

static void WriteKernelVariables(char *src, int MshTyp,
                                 int NmbArg, ArgSct *ArgTab)
{
   int      i;
   char     str[ GmlMaxStrSiz ];
   ArgSct   *arg, *LnkArg, *CptArg;

   strcat(src, "// KERNEL VARIABLES DEFINITION\n");

   for(i=0;i<NmbArg;i++)
   {
      arg = &ArgTab[i];
      LnkArg = (arg->LnkIdx != -1) ? &ArgTab[ arg->LnkIdx ] : NULL;
      CptArg = (arg->CntIdx != -1) ? &ArgTab[ arg->CntIdx ] : NULL;

      if(LnkArg && LnkArg->MaxDeg > 1)
      {
         if(arg->NmbItm > 1)
            sprintf( str,  "   %s %s[%d][%d];\n", OclTypStr[ arg->ItmTyp ],
                     arg->nam, LnkArg->MaxDeg, arg->NmbItm );
         else
            sprintf( str,  "   %s %s[%d];\n", OclTypStr[ arg->ItmTyp ],
                     arg->nam, LnkArg->MaxDeg );
      }
      else
      {
         if(arg->NmbItm > 1)
            sprintf( str,  "   %s %s[%d];\n", OclTypStr[ arg->ItmTyp ],
                     arg->nam, arg->NmbItm );
         else
            sprintf(str,  "   %s %s;\n", OclTypStr[ arg->ItmTyp ], arg->nam);
      }

      strcat(src, str);

      if(CptArg)
      {
         sprintf(str,  "   %s %sDeg;\n", OclTypStr[ CptArg->ItmTyp ], arg->nam);
         strcat(src, str);
         sprintf(str,  "   %s %sNul;\n", OclTypStr[ arg->ItmTyp ], arg->nam);
         strcat(src, str);
         sprintf(str,  "   #define   %sDegMax %d\n", arg->nam, LnkArg->MaxDeg);
         strcat(src, str);
      }
      else if(LnkArg && LnkArg->LnkDir == 0)
      {
         sprintf(str,  "   %s %sNul;\n", OclTypStr[ arg->ItmTyp ], arg->nam);
         strcat(src, str);
      }
   }
}


/*----------------------------------------------------------------------------*/
/* Write the memory reading from the global structure to the local variables  */
/*----------------------------------------------------------------------------*/

static void WriteKernelMemoryReads( char *src, int MshTyp,
                                    int NmbArg, ArgSct *ArgTab)
{
   int      i, j, k, siz;
   char     str   [ GmlMaxStrSiz ], ArgTd1[ GmlMaxStrSiz ], ArgTd2[ GmlMaxStrSiz ];
   char     LnkTd1[ GmlMaxStrSiz ], LnkTd2[ GmlMaxStrSiz ], LnkNam[ GmlMaxStrSiz ];
   char     CptNam[ GmlMaxStrSiz ], DegTst[ GmlMaxStrSiz ], DegNul[ GmlMaxStrSiz ];
   char     BalSft[ GmlMaxStrSiz ], BalMsk[ GmlMaxStrSiz ];
   ArgSct   *arg, *LnkArg, *CptArg;

   strcat(src, "   int       cnt = get_global_id(0);\n\n");
   strcat(src, "   if(cnt >= count.s0)\n      return;\n\n");
   strcat(src, "// KERNEL MEMORY READINGS\n");

   for(i=0;i<NmbArg;i++)
   {
      arg = &ArgTab[i];

      if(!(arg->FlgTab & GmlReadMode))
         continue;

      LnkArg = (arg->LnkIdx != -1) ? &ArgTab[ arg->LnkIdx ] : NULL;
      CptArg = (arg->CntIdx != -1) ? &ArgTab[ arg->CntIdx ] : NULL;

      for(j=0;j<arg->NmbItm;j++)
      {
         if(CptArg)
         {
            sprintf(CptNam,  "%sDeg", arg->nam);
            sprintf(str,  "   %s = %s;\n", CptNam, CptArg->nam);
            strcat(src, str);
            sprintf(str,  "   %sNul = %s;\n", arg->nam, OclNulVec[ arg->ItmTyp ]);
            strcat(src, str);
            strcat(src, "\n");
         }
         else if(LnkArg && LnkArg->LnkDir == 0)
         {
            sprintf(str,  "   %sNul = %s;\n", arg->nam, OclNulVec[ arg->ItmTyp ]);
            strcat(src, str);
         }

         if(arg->NmbItm > 1)
            sprintf(ArgTd1, "[%d]", j);
         else
            ArgTd1[0] = '\0';

         if(LnkArg && LnkArg->NmbItm * LnkArg->ItmLen > 1)
            siz = LnkArg->NmbItm * LnkArg->ItmLen;
         else
            siz = 1;

         for(k=0;k<siz;k++)
         {
            if(siz > 1)
               sprintf(ArgTd2, "[%d]", k);
            else
               ArgTd2[0] = '\0';

            if(LnkArg && LnkArg->NmbItm > 1)
               sprintf(LnkTd1, "[%d]", k/16);
            else
               LnkTd1[0] = '\0';

            if(LnkArg && LnkArg->ItmLen > 1)
               sprintf(LnkTd2, ".s%c", OclHexNmb[ k & 15 ]);
            else
               LnkTd2[0] = '\0';

            if(LnkArg)
               sprintf(LnkNam, "%s", LnkArg->nam);
            else if(arg->LnkDir == 1)
               sprintf(LnkNam, "cnt");
            else
               sprintf(LnkNam, "cnt + count.s1");

            if(CptArg)
            {
               sprintf(DegTst, "(%s <= %d) ?", CptNam, LnkArg->MaxDeg);
               sprintf(DegNul, ": %sNul", arg->nam);
            }
            else
            {
               if(LnkArg && LnkArg->LnkDir == 0)
               {
                  sprintf(DegTst, "%s%s ?", LnkArg->nam, LnkTd2);
                  sprintf(DegNul, ": %sNul", arg->nam);
               }
               else
                  DegNul[0] = DegTst[0] = '\0';
            }

            if(arg->LnkDir == 1)
               sprintf(BalSft, ">> 4");
            else
               BalSft[0] = '\0';

            sprintf( str, "   %s%s%s = %s %sTab[ %s%s%s ]%s %s %s;\n",
                     arg->nam, ArgTd2, ArgTd1,
                     DegTst, arg->nam, LnkNam, LnkTd1, LnkTd2, ArgTd1, BalSft, DegNul);

            strcat(src, str);
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

static void WriteKernelMemoryWrites(char *src, int MshTyp,
                                    int NmbArg, ArgSct *ArgTab)
{
   int      i, c;
   char     str[ GmlMaxStrSiz ];
   ArgSct   *arg;

   strcat(src, "\n");
   strcat(src, "// KERNEL MEMORY WRITINGS\n");

   for(i=0;i<NmbArg;i++)
   {
      arg = &ArgTab[i];

      if(!(arg->FlgTab & GmlWriteMode))
         continue;

      if(arg->NmbItm == 1)
      {
         sprintf( str, "   %sTab[ cnt + count.s1 ] = %s;\n",
                  arg->nam, arg->nam );
         strcat(src, str);
      }
      else
      {
         for(c=0;c<arg->NmbItm;c++)
         {
            sprintf( str, "   %sTab[ cnt + count.s1 ][%d] = %s[%d];\n",
                     arg->nam, c, arg->nam, c );
            strcat(src, str);
         }
      }
   }

   strcat(src, "}\n");
}


/*----------------------------------------------------------------------------*/
/* Spread an arbitrary sized vector across a number of hardware smaller ones  */
/*----------------------------------------------------------------------------*/

static void GetCntVec(int siz, int *cnt, int *vec, int *typ)
{
   int p;

   p = ceil(log2(siz));
   p = MIN(p, VECPOWMAX);

   if(p > VECPOWOCL)
   {
      *cnt = 1 << (p - VECPOWOCL);
      *vec = 1 << VECPOWOCL;
      *typ = OclVecPow[ VECPOWOCL ];
   }
   else
   {
      *cnt = 1;
      *vec = 1 << p;
      *typ = OclVecPow[p];
   }
}


/*----------------------------------------------------------------------------*/
/* Read and compile an OpenCL source code                                     */
/*----------------------------------------------------------------------------*/

static int NewOclKrn(GmlSct *gml, char *KernelSource, char *PrcNam)
{
   char     *buffer, *StrTab[1];
   int      err, res, idx = ++gml->NmbKrn;
   KrnSct   *krn = &gml->krn[ idx ];
   size_t   len, LenTab[1];

   if(idx > GmlMaxKrn)
      return(0);

   StrTab[0] = KernelSource;
   LenTab[0] = strlen(KernelSource) - 1;

   // Compile source code
   krn->program = clCreateProgramWithSource( gml->context, 1, (const char **)StrTab,
                                             (const size_t *)LenTab, &err );
   if(!krn->program)
   {
      printf("Compiling the kernel %s failed at step 1 with error %d\n", PrcNam, err);
      return(0);
   }

   res = clBuildProgram(krn->program, 0, NULL,
                        "-cl-single-precision-constant -cl-mad-enable", NULL, NULL);

   if(res != CL_SUCCESS)
   {
      clGetProgramBuildInfo(  krn->program, gml->device_id[ gml->CurDev ],
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

      if(!(buffer = malloc(len)))
         return(0);

      clGetProgramBuildInfo(  krn->program, gml->device_id[ gml->CurDev ],
                              CL_PROGRAM_BUILD_LOG, len, buffer, &len);

      printf("Compiling the kernel %s failed at step 2 with error %d\n", PrcNam, res);
      printf("%s\n", buffer);
      free(buffer);
      return(0);
   }

   krn->kernel = clCreateKernel(krn->program, PrcNam, &err);

   if( !krn->kernel || (err != CL_SUCCESS) )
   {
      printf("Compiling the kernel %s failed at step 3 with error %d\n", PrcNam, err);
      return(0);
   }

   krn->idx = idx;

   if(gml->DbgFlg)
   {
      puts(sep);
      res = clGetProgramInfo( krn->program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t), &len, NULL );

      if(res == CL_SUCCESS )
         printf("binary executable of kernel %s: %ld bytes\n", PrcNam, len);
      else
         printf("could not get the size of kernel %s executable\n", PrcNam);
   }

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

double GmlLaunchKernel(size_t GmlIdx, int idx)
{
   GETGMLPTR(gml, GmlIdx);
   double   RunTim;
   KrnSct   *hgh, *krn = &gml->krn[ idx ];

   if( (idx < 1) || (idx > gml->NmbKrn) || !krn->kernel )
      return(-1);

   // First send the parameters to the GPU memmory
   if(!UploadData(gml, gml->ParIdx))
      return(-2);

   if(!krn->HghIdx)
      RunTim = RunOclKrn(gml, krn);
   else
   {
      hgh = &gml->krn[ krn->HghIdx ];
      RunTim  = RunOclKrn(gml, krn);
      RunTim += RunOclKrn(gml, hgh);
   }

   // Finaly, get back the parameters from the GPU memmory
   if(!DownloadData(gml, gml->ParIdx))
      return(-10);

   return(RunTim);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

static double RunOclKrn(GmlSct *gml, KrnSct *krn)
{
   int      i, res;
   size_t   NmbGrp, GrpSiz, RetSiz = 0;
   DatSct   *dat;
   cl_event event;
   cl_ulong start, end;

   for(i=0;i<krn->NmbDat;i++)
   {
      dat = &gml->dat[ krn->DatTab[i] ];

      if((krn->DatTab[i] < 1) || (krn->DatTab[i] > GmlMaxDat) || !dat->GpuMem)
      {
         printf(  "Invalid user argument %d, DatTab[i]=%d, GpuMem=%p\n",
                  i, krn->DatTab[i], dat->GpuMem );
         return(-1.);
      }

      res = clSetKernelArg(krn->kernel, i, sizeof(cl_mem), &dat->GpuMem);

      if(res != CL_SUCCESS)
      {
         printf("Adding user argument %d failed with error: %d\n", i, res);
         return(-2.);
      }
   }

   res = clSetKernelArg(krn->kernel, krn->NmbDat, sizeof(cl_mem),
                        &gml->dat[ gml->ParIdx ].GpuMem);

   if(res != CL_SUCCESS)
   {
      printf("Adding the GMlib parameters argument failed with error %d\n", res);
      return(-3.);
   }

   res = clSetKernelArg(krn->kernel, krn->NmbDat+1, 2 * sizeof(int), krn->NmbLin);

   if(res != CL_SUCCESS)
   {
      printf("Adding the kernel loop counter argument failed with error %d\n", res);
      return(-4.);
   }

   // Fit data loop size to the GPU kernel size
   res = clGetKernelWorkGroupInfo(  krn->kernel, gml->device_id[ gml->CurDev ],
                                    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                    &GrpSiz, &RetSiz );

   if(res != CL_SUCCESS )
   {
      printf("Geting the kernel workgroup size failed with error %d\n", res);
      return(-5.);
   }

   // Compute the hyperthreading level
   gml->CurGrpSiz = GrpSiz;
   NmbGrp = krn->NmbLin[0] / GrpSiz;
   NmbGrp *= GrpSiz;

   if(NmbGrp < krn->NmbLin[0])
      NmbGrp += GrpSiz;

   // Launch GPU code
   clFinish(gml->queue);

   if(clEnqueueNDRangeKernel( gml->queue, krn->kernel, 1, NULL,
                              &NmbGrp, &GrpSiz, 0, NULL, &event) )
   {
      return(-6.);
   }

   clFinish(gml->queue);

   res = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                 sizeof(start), &start, NULL);
   if(res != CL_SUCCESS )
   {
      printf("Geting the start time event failed with error %d\n", res);
      return(-7.);
   }

   res = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                 sizeof(end), &end, NULL);

   if(res != CL_SUCCESS )
   {
      printf("Geting the end time event failed with error %d\n", res);
      return(-8.);
   }

   return((double)(end - start) / 1e9);
}


/*----------------------------------------------------------------------------*/
/* Compute various reduction functions: min,max,L1,L2 norms                   */
/*----------------------------------------------------------------------------*/

double GmlReduceVector(size_t GmlIdx, int DatIdx, int RedOpp, double *res)
{
   int      i, NmbLin;
   float    *vec;
   double   tim;
   char     *RedNam[3] = {"reduce_min", "reduce_max", "reduce_sum"};
   GETGMLPTR(gml, GmlIdx);
   DatSct   *dat, *red;
   KrnSct   *krn;

   // Check indices and data conformity
   if( (DatIdx < 1) || (DatIdx > GmlMaxDat) )
   {
      printf("Invalid data index: %d\n", DatIdx);
      return(-1.);
   }

   dat = &gml->dat[ DatIdx ];

   if( (dat->ItmTyp != GmlFlt) || (dat->NmbItm != 1) || (dat->ItmLen != 1) )
   {
      printf(  "Invalid data structure: count %d, type %s, length %d\n",
               dat->NmbItm, OclTypStr[ dat->ItmTyp ], dat->ItmLen );
      return(-2.);
   }

   if( (RedOpp < 0) || (RedOpp > GmlMaxRed) )
   {
      printf("Invalid operation code %d\n", RedOpp);
      return(-3.);
   }

   // Allocate an output vector the size of the input vector
   if(!dat->RedIdx)
      dat->RedIdx = GmlNewSolutionData(GmlIdx, dat->MshTyp, 1, GmlFlt, "reduce");

   if(!dat->RedIdx)
   {
      printf(  "Failed to allocate a reduction vector of %d bytes\n",
               OclTypSiz[ GmlFlt ] * dat->NmbLin );
      return(-4.);
   }

   // Compile a reduction kernel with the required operation if needed
   if(!gml->RedKrn[ RedOpp ])
      gml->RedKrn[ RedOpp ] = NewOclKrn(gml, reduce, RedNam[ RedOpp ]);

   if(!gml->RedKrn[ RedOpp ])
   {
      printf("Failed to compile the %s reduction kernel\n", RedNam[ RedOpp ]);
      return(-5.);
   }

   // Set the kernel with two vectors: an input and a reduced output one
   krn = &gml->krn[ gml->RedKrn[ RedOpp ] ];
   krn->NmbDat    = 2;
   krn->DatTab[0] = DatIdx;
   krn->DatTab[1] = dat->RedIdx;
   krn->NmbLin[0] = dat->NmbLin;

   // Launch the right reduction kernel according to the requested opperation
   tim = RunOclKrn(gml, krn);

   if(tim < 0)   
      return(tim);

   // Trim the size of the output vector down to the number of OpenCL groups
   // used by the kernel and download this amount of data
   red = &gml->dat[ dat->RedIdx ];
   red->MemSiz = dat->MemSiz / gml->CurGrpSiz;
   NmbLin = dat->NmbLin / gml->CurGrpSiz;
   DownloadData(gml, dat->RedIdx);
   red->MemSiz = dat->MemSiz;
   vec = (float *)red->CpuMem;

   // Perform the last reduction step on the CPU
   switch(RedOpp)
   {
      case GmlMin :
      {
         *res = 1e37;
         for(i=0;i<NmbLin;i++)
            *res = MIN(*res, vec[i]);
      }break;

      case GmlSum :
      {
         *res = 0.;
         for(i=0;i<NmbLin;i++)
            *res += vec[i];
      }break;

      case GmlMax :
      {
         *res = -1e37;
         for(i=0;i<NmbLin;i++)
            *res = MAX(*res, vec[i]);
      }break;
   }

   return(tim);
}


/*----------------------------------------------------------------------------*/
/* Return memory currently allocated on the GPU                               */
/*----------------------------------------------------------------------------*/

size_t GmlGetMemoryUsage(size_t GmlIdx)
{
   GETGMLPTR(gml, GmlIdx);
   return(gml->MemSiz);
}


/*----------------------------------------------------------------------------*/
/* Return memory currently transfered to or from the GPU                      */
/*----------------------------------------------------------------------------*/

size_t GmlGetMemoryTransfer(size_t GmlIdx)
{
   GETGMLPTR(gml, GmlIdx);
   return(gml->MovSiz);
}


/*----------------------------------------------------------------------------*/
/* Turning the printing of debugging information on or off                    */
/*----------------------------------------------------------------------------*/

void GmlDebugOn(size_t GmlIdx)
{
   GETGMLPTR(gml, GmlIdx);
   gml->DbgFlg = 1;
}

void GmlDebugOff(size_t GmlIdx)
{
   GETGMLPTR(gml, GmlIdx);
   gml->DbgFlg = 0;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

int GmlCheckFP64(size_t GmlIdx)
{
   GETGMLPTR(gml, GmlIdx);
   return(gml->DblExt);
}


/*----------------------------------------------------------------------------*/
/* Exctract the list of unique volume edges from any kind of elements         */
/*----------------------------------------------------------------------------*/

int GmlExtractEdges(size_t GmlIdx)
{
   int         i, j, typ, idx, cod, cpt = 0, HshKey, ItmTab[3];
   int         EdgIdx, EleSiz, NmbItm, EleLen, *EleNod, *nod;
   DatSct      *dat;
   BucSct      *buc;
   HshTabSct   lnk;

   GETGMLPTR(gml, GmlIdx);

   for(typ=GmlEdges+1; typ<GmlMaxEleTyp; typ++)
      if(gml->TypIdx[ typ ])
         cpt += gml->dat[ gml->TypIdx[ typ ] ].NmbLin;

   // Setup a hash table
   memset(&lnk, 0, sizeof(HshTabSct));
   lnk.HshTyp = GmlEdges;
   lnk.TabSiz = cpt;
   lnk.DatLen = ItmNmbVer[ lnk.HshTyp ];
   lnk.KeyLen = HshLenTab[ lnk.DatLen ];
   lnk.NmbDat = lnk.TabSiz;
   lnk.NxtDat = 1;
   lnk.HshTab = calloc(lnk.TabSiz, sizeof(int));
   lnk.DatTab = malloc(lnk.TabSiz * sizeof(BucSct));
   cpt = 0;

   if(gml->DbgFlg)
      printf(  "Hash table: lines=%lld, stored items=%d, hash keys=%d\n",
               lnk.TabSiz, lnk.DatLen, lnk.KeyLen);

   for(typ=GmlEdges+1; typ<GmlMaxEleTyp; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      EleNod = (int *)dat->CpuMem;
      NmbItm = NmbTpoLnk[ typ ][ lnk.HshTyp ];
      EleLen = ItmNmbVer[ typ ];

      // Add edges to the hash table
      for(i=0;i<dat->NmbLin;i++)
      {
         nod = &EleNod[ i * EleLen ];

         for(j=0;j<NmbItm;j++)
         {
            GetItmNod(nod, typ, lnk.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&lnk, ItmTab);

            if(GetHsh(&lnk, HshKey, i, j, ItmTab, NULL))
               continue;

            AddHsh(&lnk, HshKey, i, j, ItmTab);
            cpt++;
         }
      }
   }

   if(gml->DbgFlg)
      printf(  "Hashed %lld entities: occupency=%lld%%, collisions=%g\n",
               lnk.NxtDat-1, (100LL * lnk.NmbHit) / lnk.TabSiz,
               (double)lnk.NmbMis / (double)lnk.TabSiz );

   EdgIdx = GmlNewMeshData(GmlIdx, GmlEdges, cpt);
   cpt = 0;

   for(typ=GmlEdges+1; typ<GmlMaxEleTyp; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      EleNod = (int *)dat->CpuMem;
      NmbItm = NmbTpoLnk[ typ ][ lnk.HshTyp ];
      EleLen = ItmNmbVer[ typ ];

      // Get edges from the hash table
      for(i=0;i<dat->NmbLin;i++)
      {
         nod = &EleNod[ i * EleLen ];

         for(j=0;j<NmbItm;j++)
         {
            GetItmNod(nod, typ, lnk.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&lnk, ItmTab);
            cod = 0;

            if(!GetHsh(&lnk, HshKey, i, j, ItmTab, &cod))
               continue;

            if(cod >> 4 != i)
               continue;

            GmlSetDataLine(GmlIdx, EdgIdx, cpt, ItmTab[0], ItmTab[1], 0);
            cpt++;
         }
      }
   }

   free(lnk.HshTab);
   free(lnk.DatTab);

   if(gml->DbgFlg)
      printf("Hashed, setup and transfered %d edges to the GMlib.\n", cpt);

   return(EdgIdx);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

int GmlExtractFaces(size_t GmlIdx)
{
   return(0);
}


/*----------------------------------------------------------------------------*/
/* Build, allocate and transfer neighbourhood information                     */
/*----------------------------------------------------------------------------*/

int GmlSetNeighbours(size_t GmlIdx, int typ)
{
   int         i, j, k, cpt, cod[2], HshKey, ItmTab[4], TetNgb[4];
   int         NgbIdx;
   int         NmbItm, EleLen, *EleNod, *nod;
   DatSct      *dat;
   BucSct      *buc;
   HshTabSct   lnk;

   if(typ == GmlPyramids || typ == GmlPrisms)
   {
      puts("Neighbours calculation between prisms or pyramids is not yet implemented.");
      return(0);
   }

   GETGMLPTR(gml, GmlIdx);

   // Get and check the source and destination mesh datatypes
   CHKELETYP(typ);
   dat = &gml->dat[ gml->TypIdx[ typ ] ];
   EleNod = (int *)dat->CpuMem;

   // Setup a hash table
   memset(&lnk, 0, sizeof(HshTabSct));
   lnk.HshTyp = NgbTyp[ typ ];
   lnk.TabSiz = dat->NmbLin * NmbTpoLnk[ typ ][ typ ] / 2;
   lnk.DatLen = ItmNmbVer[ lnk.HshTyp ];
   lnk.KeyLen = HshLenTab[ lnk.DatLen ];
   lnk.NmbDat = lnk.TabSiz;
   lnk.NxtDat = 1;
   lnk.HshTab = calloc(lnk.TabSiz, sizeof(int));
   lnk.DatTab = malloc(lnk.TabSiz * sizeof(BucSct));

   if(gml->DbgFlg)
      printf(  "Hash table: lines=%lld, stored items=%d, hash keys=%d\n",
               lnk.TabSiz, lnk.DatLen, lnk.KeyLen);

   NmbItm = NmbTpoLnk[ typ ][ lnk.HshTyp ];
   EleLen = ItmNmbVer[ typ ];

   // Add destination entities to the hash table
   for(i=0;i<dat->NmbLin;i++)
   {
      nod = &EleNod[ i * EleLen ];

      for(j=0;j<NmbItm;j++)
      {
         GetItmNod(nod, typ, lnk.HshTyp, j, ItmTab);
         HshKey = CalHshKey(&lnk, ItmTab);
         AddHsh(&lnk, HshKey, i, j, ItmTab);
      }
   }

   if(gml->DbgFlg)
      printf(  "Hashed %lld entities: occupency=%lld%%, collisions=%g\n",
               lnk.NmbDat, (100LL * lnk.NmbHit) / lnk.TabSiz,
               (double)lnk.NmbMis / (double)lnk.TabSiz );

   // Build downlinks and neighbours
   NgbIdx = GmlNewLinkData(GmlIdx, typ, typ, ItmNmbFac[ typ ], "ngb");

   for(i=0;i<dat->NmbLin;i++)
   {
      nod = &EleNod[ i * EleLen ];

      for(j=0;j<NmbItm;j++)
      {
         GetItmNod(nod, typ, lnk.HshTyp, j, ItmTab);
         HshKey = CalHshKey(&lnk, ItmTab);
         cpt = GetHsh(&lnk, HshKey, i, j, ItmTab, cod);
         TetNgb[j] = 0;

         for(k=0;k<cpt;k++)
            if(cod[k] >> 4 != i)
               TetNgb[j] = cod[k] >> 4;
      }

      GmlSetDataLine(GmlIdx, NgbIdx, i, &TetNgb);
   }

   if(gml->DbgFlg)
      printf("Stored %d uniq entries in the link table\n", dat->NmbLin * NmbItm);

   free(lnk.HshTab);
   free(lnk.DatTab);

   return(NgbIdx);
}
