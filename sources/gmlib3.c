

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.32                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: feb 07 2022                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#include <sys/timeb.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <wchar.h>
#include <io.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "gmlib3.h"
#include "reduce.h"
#include "toolkit.h"


/*----------------------------------------------------------------------------*/
/* Defines                                                                    */
/*----------------------------------------------------------------------------*/

#define MB           1048576
#define VECPOWOCL    4
#define VECPOWMAX    7
#define DEFEVTBLK    100
#define STRSIZ       1024

enum data_type       {GmlArgDat, GmlRawDat, GmlLnkDat, GmlEleDat, GmlRefDat};
enum memory_type     {GmlInternal, GmlInput, GmlOutput, GmlInout};


/*----------------------------------------------------------------------------*/
/* Library's internal data structures                                         */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int            EleTyp, EleIdx, ItmIdx, NxtDat, nod[4];
}BucSct;

typedef struct
{
   int            HshTyp, DatLen, KeyLen, *HshTab;
   int            NmbMis, NmbHit, TabSiz, NmbDat, NxtDat;
   BucSct         *DatTab;
}HshTabSct;

typedef struct
{
   int            ArgIdx, MshTyp, DatIdx, LnkTyp, LnkIdx, LnkDir, CntIdx;
   int            LnkDeg, MaxDeg, NmbItm, ItmLen, ItmTyp, FlgTab;
   const char     *nam, *VoyNam;
}ArgSct;

typedef struct
{
   int            AloTyp, MemAcs, MshTyp, LnkTyp, ItmTyp, RedIdx;
   int            NmbItm, ItmLen, ItmSiz, NmbLin, LinSiz;
   char           *src, use;
   const char     *nam, *VoyNam;
   size_t         MemSiz;
   cl_mem         GpuMem;
   void           *CpuMem;
}DatSct;

typedef struct
{
   int            idx, HghIdx, NmbLin[2], NmbDat, DatTab[ GmlMaxDat ];
   int            NmbEvt, EvtBlk, IniFlg;
   cl_event       *EvtTab;
   size_t         NmbGrp, GrpSiz;
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
   int            SizMatHgh[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   int            RedKrn[ GmlMaxRed ];
   char           *UsrTlk;
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
#define CHKDATIDX(p, i) if( ((i) < 1) || ((i) > GmlMaxDat) || !p->dat[(i)].use) return(0);
#define CHKELETYP(t)    if( ((t) < 0) || ((t) >= GmlMaxEleTyp)) return(0)
#define CHKOCLTYP(t)    if( ((t) < GmlInt) || ((t) >= GmlMaxOclTyp)) return(0)
#define GETGMLPTR(p,i)  GmlSct *p = (GmlSct *)(i)


/*----------------------------------------------------------------------------*/
/* Prototypes of local procedures                                             */
/*----------------------------------------------------------------------------*/

static int     NewData                 (GmlSct *, DatSct *);
static int     NewBallData             (GmlSct *, int, int, char *, char *, char *);
static int     UploadData              (GmlSct *, int);
static int     DownloadData            (GmlSct *, int);
static int     NewOclKrn               (GmlSct *, char *, char *);
static int     GetNewDatIdx            (GmlSct *);
static int     RunOclKrn               (GmlSct *, KrnSct *);
static void    WriteToolkitSource      (char *, char *);
static void    WriteUserToolkitSource  (char *, char *);
static void    WriteUserTypedef        (char *, char *);
static void    WriteProcedureHeader    (char *, char *, int, int, ArgSct *);
static void    WriteKernelVariables    (char *, int, int, ArgSct *);
static void    WriteKernelMemoryReads  (char *, int, int, ArgSct *);
static void    WriteKernelMemoryWrites (char *, int, int, ArgSct *);
static void    WriteUserKernel         (char *, char *);
static void    GetCntVec               (int , int *, int *, int *);
static void    GetItmNod               (int *, int, int, int, int *);
static int     CalHshKey               (HshTabSct *, int *);
static void    AddHsh                  (HshTabSct *, int, int, int, int, int *);
static int     GetHsh                  (HshTabSct *, int, int, int, int *, int *, int *);


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
   sizeof(cl_double16),
   sizeof(cl_char),
   sizeof(cl_char2),
   sizeof(cl_char4),
   sizeof(cl_char8),
   sizeof(cl_char16) };

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
   "double16 ",
   "char     ",
   "char2    ",
   "char4    ",
   "char8    ",
   "char16   " };

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
   "(double16){0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}",
   "(char){0}",
   "(char2){0,0}",
   "(char4){0,0,0,0}",
   "(char8){0,0,0,0,0,0,0,0}",
   "(char16){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}" };

static const int  TypVecSiz[ GmlMaxOclTyp ]  = {
   1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16 };

static const int  OclVecPow[ VECPOWOCL +1 ]  = {
   GmlInt, GmlInt2, GmlInt4, GmlInt8, GmlInt16};

static const int  MshItmTyp[ GmlMaxEleTyp ]  = {
   GmlFlt4, GmlInt2, GmlInt4, GmlInt4, GmlInt4, GmlInt8, GmlInt8, GmlInt8};

static const int  EleNmbNod[ GmlMaxEleTyp ]  = {0, 2, 3, 4, 4, 5, 6, 8};
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

static const int GmfMshKwdTab[8] = {
GmfVertices, GmfEdges, GmfTriangles, GmfQuadrilaterals,
GmfTetrahedra, GmfPyramids, GmfPrisms, GmfHexahedra };

static const int GmfSolKwdTab[8] = {
GmfSolAtVertices, GmfSolAtEdges, GmfSolAtTriangles, GmfSolAtQuadrilaterals,
GmfSolAtTetrahedra, GmfSolAtPyramids, GmfSolAtPrisms, GmfSolAtHexahedra };

static const int GmfTypTab[10] = {
GmfFloat,  GmfFloatVec,  GmfFloatVec,  GmfFloatVec,  GmfFloatVec,
GmfDouble, GmfDoubleVec, GmfDoubleVec, GmfDoubleVec, GmfDoubleVec };

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
   if( (DevIdx < 0) || (DevIdx >= (int)gml->NmbDev) )
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

   err = clGetDeviceInfo(  gml->device_id[ gml->CurDev ],
                           CL_DEVICE_EXTENSIONS, 1024, str, &retSiz );

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

   for(i=0;i<(int)num_devices;i++)
      if(clGetDeviceInfo(  device_id[i], CL_DEVICE_NAME, GmlMaxStrSiz,
                           GpuNam, &GpuNamSiz) == CL_SUCCESS )
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
      return(NULL);

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
      return((NULL));

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
   //gml->CntMat[ MshTyp ][ MshTyp ] = RefIdx;
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

static int NewBallData( GmlSct *gml, int SrcTyp, int DstTyp,
                        char *BalNam, char *DegNam, char *VoyNam )
{
   int         i, j, idx, cod[4], cpt, dir, HshKey, ItmTab[4];
   int         BalIdx, HghIdx, DegIdx;
   int         VecSiz, BalSiz, MaxSiz, HghSiz = 0;
   int         *EleTab, *BalTab, *DegTab, *HghTab, *PtrInt;
   int         MaxDeg = 0, MaxPos = 0, DegTot = 0, VecCnt, ItmTyp, NmbDat;
   int         SrcNmbItm, SrcLen, DstNmbItm, DstLen, *SrcNod, *DstNod, *EleNod;
   const char  *SrcNam, *DstNam;
   DatSct      *src, *dst, *bal, *hgh, *deg, *BalDat, *HghDat, *DegDat;
   HshTabSct   lnk;

   // Get and check the source and destination mesh datatypes
   CHKELETYP(SrcTyp);
   CHKELETYP(DstTyp);
   memset(&lnk, 0, sizeof(HshTabSct));

   src = &gml->dat[ gml->TypIdx[ SrcTyp ] ];
   dst = &gml->dat[ gml->TypIdx[ DstTyp ] ];
   printf("build link between typ: %d dat:%d, cpt:%d ->typ: %d dat:%d, cpt:%d\n",
         SrcTyp, gml->TypIdx[ SrcTyp ], src->NmbLin,
         DstTyp, gml->TypIdx[ DstTyp ], dst->NmbLin);

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
         printf(  "Building %s neighbours between %s\n",
                  SrcNam, BalTypStr[ NgbTyp[ SrcTyp ] ] );
   }

   // Setup a hash table
   lnk.DatLen = ItmNmbVer[ lnk.HshTyp ];
   lnk.KeyLen = HshLenTab[ lnk.DatLen ];
   lnk.NmbDat = lnk.TabSiz;
   lnk.NxtDat = 1;
   lnk.HshTab = calloc(lnk.TabSiz, sizeof(int));
   lnk.DatTab = malloc(lnk.NmbDat * sizeof(BucSct));

   if(!lnk.HshTab || !lnk.DatTab)
      return(0);

   if(gml->DbgFlg)
      printf(  "Hash table: lines=%d, stored items=%d, hash keys=%d\n",
               (int)lnk.TabSiz, lnk.DatLen, lnk.KeyLen);

   // Workaround to avoid reading the number of neighbours
   // in case a type is pointing to itself
   if(SrcTyp != lnk.HshTyp)
      SrcNmbItm = NmbTpoLnk[ SrcTyp ][ lnk.HshTyp ];
   else
      SrcNmbItm = 1;

   if(DstTyp != lnk.HshTyp)
      DstNmbItm = NmbTpoLnk[ DstTyp ][ lnk.HshTyp ];
   else
      DstNmbItm = 1;

   SrcLen = src->ItmLen;
   DstLen = dst->ItmLen;

   // Add destination entities to the hash table
   for(i=0;i<dst->NmbLin;i++)
   {
      EleNod = &DstNod[ i * DstLen ];

      for(j=0;j<DstNmbItm;j++)
      {
         GetItmNod(EleNod, DstTyp, lnk.HshTyp, j, ItmTab);
         HshKey = CalHshKey(&lnk, ItmTab);
         AddHsh(&lnk, HshKey, DstTyp, i, j, ItmTab);
      }
   }

   if(gml->DbgFlg)
      printf(  "Hashed %d entities: occupency=%d%%, collisions=%g\n",
               (int)lnk.NmbDat, (int)((100 * lnk.NmbHit) / lnk.TabSiz),
               (double)lnk.NmbMis / (double)lnk.TabSiz );

   // Allocate and fill a GPU data type to store the degrees in case uf uplink
   if(dir == 1)
   {
      // First allocate the GPU datatype
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
            DegTab[ idx ] = GetHsh(&lnk, HshKey, i, j, ItmTab, NULL, NULL);
         }
      }

      EleTab = dst->CpuMem;
      BalSiz = LenMatBas[ src->MshTyp ][ dst->MshTyp ];
      MaxSiz = LenMatMax[ src->MshTyp ][ dst->MshTyp ];

      if(BalSiz == MaxSiz)
      {
         for(i=0;i<src->NmbLin;i++)
            DegTot += DegTab[i];

         MaxDeg = BalSiz;
         HghSiz = 0;
         MaxPos = src->NmbLin;
         gml->SizMatHgh[ src->MshTyp ][ dst->MshTyp ] = 0;

         if(gml->DbgFlg)
            printf("Constant width uplink: %d\n", BalSiz);
      }
      else
      {
         for(i=0;i<src->NmbLin;i++)
         {
            if(!MaxPos && (DegTab[i] > BalSiz))
               MaxPos = i;

            DegTot += DegTab[i];
            MaxDeg = MAX(MaxDeg, DegTab[i]);
         }

         MaxDeg = (int)pow(2., ceil(log2(MaxDeg)));

         // If the max degree is greater than de base size,
         // create an extension ball table for high degree entities
         if(MaxDeg > BalSiz)
         {
            HghSiz = MIN(MaxDeg, LenMatMax[ src->MshTyp ][ dst->MshTyp ]);
            gml->SizMatHgh[ src->MshTyp ][ dst->MshTyp ] = HghSiz;

            if(gml->DbgFlg)
               printf(  "Width for lines 1..%d: %d, for lines %d..%d:%d\n",
                        MaxPos, BalSiz, MaxPos+1, src->NmbLin, MaxDeg);
         }
         else
         {
            // If the max degree fits in the base size vector,
            // fall back to a regular contant width table
            MaxDeg = BalSiz;
            HghSiz = 0;
            MaxPos = src->NmbLin;
            gml->SizMatHgh[ src->MshTyp ][ dst->MshTyp ] = 0;

            if(gml->DbgFlg)
               printf("Constant width uplink: %d\n", BalSiz);
         }
      }
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
      BalDat->ItmLen = NmbDat;
      BalDat->NmbLin = src->NmbLin;
      BalDat->LinSiz = BalDat->NmbItm * BalDat->ItmSiz;
      BalDat->MemSiz = (size_t)BalDat->NmbLin * (size_t)BalDat->LinSiz;
      BalDat->GpuMem = BalDat->CpuMem = NULL;
      BalDat->nam    = BalNam;

      if(!NewData(gml, BalDat))
         return(0);

      // fetch the pointed items from the hash table and store them as downlinks
      gml->LnkMat[ SrcTyp ][ DstTyp ] = BalIdx;
      bal = &gml->dat[ gml->LnkMat[ src->MshTyp ][ dst->MshTyp ] ];
      BalTab = bal->CpuMem;

      for(i=0;i<src->NmbLin;i++)
      {
         EleNod = &SrcNod[ i * SrcLen ];

         for(j=0;j<SrcNmbItm;j++)
         {
            idx = i * SrcNmbItm + j;

            ItmTab[0]=ItmTab[1]=ItmTab[2]=ItmTab[3]=0;
            GetItmNod(EleNod, SrcTyp, lnk.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&lnk, ItmTab);
            PtrInt = &BalTab[ i * BalDat->ItmLen + j ];

            if((cpt = GetHsh(&lnk, HshKey, i, j, ItmTab, cod, NULL)))
            {
               if(dir == -1)
                  BalTab[ i * BalDat->ItmLen + j ] = cod[0] >> 4;
               else
               {
                  if(cpt != 2)
                     BalTab[ i * BalDat->ItmLen + j ] = 0;
                  else if( ((cod[0] >> 4) != i) || ((cod[0] & 15) != j) )
                     BalTab[ i * BalDat->ItmLen + j ] = cod[0];
               }
            }
         }
      }

      // Upload the ball data to th GPU memory
      gml->MovSiz += UploadData(gml, BalIdx);
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
      BalDat->ItmLen = NmbDat;
      BalDat->NmbLin = MaxPos+1;
      BalDat->LinSiz = BalDat->NmbItm * BalDat->ItmSiz;
      BalDat->MemSiz = (size_t)BalDat->NmbLin * (size_t)BalDat->LinSiz;
      BalDat->GpuMem = BalDat->CpuMem = NULL;
      BalDat->nam    = BalNam;
      BalDat->VoyNam = VoyNam;

      if(!NewData(gml, BalDat))
         return(0);

      if(gml->DbgFlg)
         printf(  "Allocate a base table with %d lines of %d width vectors\n",
                  BalDat->NmbLin, NmbDat);

      gml->LnkMat[ SrcTyp ][ DstTyp ] = BalIdx;
      bal = &gml->dat[ gml->LnkMat[ src->MshTyp ][ dst->MshTyp ] ];

      // Allocate the high vector ball table
      if(HghSiz)
      {
         if(!(HghIdx = GetNewDatIdx(gml)))
            return(0);

         NmbDat = gml->SizMatHgh[ SrcTyp ][ DstTyp ];
         GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

         HghDat = &gml->dat[ HghIdx ];

         HghDat->AloTyp = GmlLnkDat;
         HghDat->MshTyp = SrcTyp;
         HghDat->LnkTyp = DstTyp;
         HghDat->MemAcs = GmlInout;
         HghDat->ItmTyp = ItmTyp;
         HghDat->NmbItm = VecCnt;
         HghDat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
         HghDat->ItmLen = NmbDat;
         HghDat->NmbLin = gml->NmbEle[ SrcTyp ] - MaxPos;
         HghDat->LinSiz = HghDat->NmbItm * HghDat->ItmSiz;
         HghDat->MemSiz = (size_t)HghDat->NmbLin * (size_t)HghDat->LinSiz;
         HghDat->GpuMem = HghDat->CpuMem = NULL;
         HghDat->nam    = BalNam;
         HghDat->VoyNam = VoyNam;

         if(!NewData(gml, HghDat))
            return(0);

         if(gml->DbgFlg)
            printf(  "Allocate a hash table with %d lines of %d width vectors\n",
                     HghDat->NmbLin, NmbDat);

         gml->LnkHgh[ SrcTyp ][ DstTyp ] = HghIdx;
         hgh = &gml->dat[ gml->LnkHgh[ src->MshTyp ][ dst->MshTyp ] ];
      }

      // Fill both ball tables at the same time
      BalTab = bal->CpuMem;

      if(HghSiz)
         HghTab = hgh->CpuMem;
      else
         HghTab = NULL;

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

         if(!HghTab || (i <= MaxPos))
            GetHsh(&lnk, HshKey, i, 0, ItmTab, &BalTab[ i * BalSiz ], NULL);
         else
            GetHsh(&lnk, HshKey, i, 0, ItmTab, &HghTab[ (i - MaxPos) * HghSiz ], NULL);
      }

      if(gml->DbgFlg)
         puts("Uploading the three tables");

      // Upload the ball data to th GPU memory
      gml->MovSiz += UploadData(gml, BalIdx);
      gml->MovSiz += UploadData(gml, DegIdx);

      if(HghSiz)
         gml->MovSiz += UploadData(gml, HghIdx);

      if(gml->DbgFlg)
      {
         puts(sep);
         printf(  "Ball generation: type %s -> %s\n",
                  BalTypStr[ SrcTyp ], BalTypStr[ DstTyp ] );
         printf(  "low degree ranging from 1 to %d, occupency = %g%%\n",
                  MaxPos, (float)(100 * DegTot) / (float)(MaxPos * BalSiz) );

         if(HghSiz)
            printf(  "high degree entities = %d\n", src->NmbLin - MaxPos);

         printf(  "Allocated degree     data: index=%2d, size=%zu bytes\n",
                  DegIdx, DegDat->MemSiz );
         printf(  "Allocated short ball data: index=%2d, size=%zu bytes\n",
                  BalIdx, BalDat->MemSiz );

         if(HghSiz)
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
   int i, j, tmp;

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

static void AddHsh(  HshTabSct *lnk, int HshKey, int EleTyp,
                     int EleIdx, int ItmIdx, int *ItmTab )
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
   buc->EleTyp = EleTyp;
   buc->EleIdx = EleIdx;
   buc->ItmIdx = ItmIdx;
   buc->NxtDat = nxt;

   for(i=0;i<lnk->DatLen;i++)
      buc->nod[i] = ItmTab[i];

   lnk->NxtDat++;

   if(nxt)
      lnk->NmbMis++;
   else
      lnk->NmbHit++;
}


/*----------------------------------------------------------------------------*/
/* Fetch an element/entity pair from the hash table                           */
/*----------------------------------------------------------------------------*/

static int GetHsh(   HshTabSct *lnk, int HshKey, int EleIdx,
                     int ItmIdx, int *ItmTab, int *UsrTab, int *TypTab )
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

         if(TypTab)
            TypTab[ deg ] = buc->EleTyp;

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
      printf(  "Cannot allocate %zd MB on the GPU (%zd MB already used)\n",
               dat->MemSiz / MB, GmlGetMemoryUsage((size_t)gml) / MB);
      return(0);
   }

   // Allocate the requested memory size on the CPU side
   if( (dat->MemAcs != GmlInternal) && !(dat->CpuMem = calloc(1, dat->MemSiz)) )
   {
      printf("Cannot allocate %zd MB on the CPU\n", dat->MemSiz/MB);
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
   CHKDATIDX(gml, idx);
   DatSct   *dat = &gml->dat[ idx ], *RefDat;
   char     *adr = (void *)dat->CpuMem;
   int      i, *EleTab, siz, *RefTab, *tab, RefIdx = 0;
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
      RefIdx = gml->RefIdx[ dat->MshTyp ];
      RefDat = &gml->dat[ RefIdx ];
      RefTab = (int *)RefDat->CpuMem;
      siz = 4;

      for(i=0;i<3;i++)
         CrdTab[ lin * siz + i ] = (float)va_arg(VarArg, double);

      CrdTab[ lin * siz + 3 ] = 0.;
      RefTab[ lin ] = va_arg(VarArg, int);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp > GmlVertices) )
   {
      EleTab = (int *)dat->CpuMem;
      siz = TypVecSiz[ MshItmTyp[ dat->MshTyp ] ];
      RefIdx = gml->RefIdx[ dat->MshTyp ];
      RefDat = &gml->dat[ RefIdx ];
      RefTab = (int *)RefDat->CpuMem;

      for(i=0;i<EleNmbNod[ dat->MshTyp ];i++)
         EleTab[ lin * siz + i ] = va_arg(VarArg, int);

      RefTab[ lin ] = va_arg(VarArg, int);
   }

   va_end(VarArg);

   if(lin == dat->NmbLin - 1)
   {
      gml->MovSiz += UploadData(gml, idx);

      if(RefIdx)
         gml->MovSiz += UploadData(gml, RefIdx);
   }

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Get a line of solution or vertex corrdinates from the librarie's buffers   */
/*----------------------------------------------------------------------------*/

int GmlGetDataLine(size_t GmlIdx, int idx, int lin, ...)
{
   GETGMLPTR(gml, GmlIdx);
   CHKDATIDX(gml, idx);
   DatSct   *dat = &gml->dat[ idx ], *RefDat;
   char     *adr = (void *)dat->CpuMem;
   int      i, *EleTab, siz, *RefTab, RefIdx = 0, *UsrDat;
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
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp > GmlVertices) )
   {
      EleTab = (int *)dat->CpuMem;
      siz = TypVecSiz[ MshItmTyp[ dat->MshTyp ] ];
      RefIdx = gml->RefIdx[ dat->MshTyp ];
      RefDat = &gml->dat[ RefIdx ];
      RefTab = (int *)RefDat->CpuMem;

      for(i=0;i<EleNmbNod[ dat->MshTyp ];i++)
      {
         UsrDat = va_arg(VarArg, int *);
         *UsrDat = EleTab[ lin * siz + i ];
      }

      UsrDat = va_arg(VarArg, int *);
      *UsrDat = RefTab[ lin ];
   }

   va_end(VarArg);

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Set a line of mesh, link or solution data                                  */
/*----------------------------------------------------------------------------*/

int GmlSetDataBlock( size_t GmlIdx, int   TypIdx,
                     int    BegIdx, int   EndIdx,
                     void  *DatBeg, void *DatEnd,
                     int   *RefBeg, int  *RefEnd )
{
   GETGMLPTR(gml, GmlIdx);
   CHKELETYP(TypIdx);
   int      DatIdx = gml->TypIdx[ TypIdx ], RefIdx = gml->RefIdx[ TypIdx ];
   DatSct   *dat = &gml->dat[ DatIdx ], *RefDat;
   int      i, j, *EleTab, siz, *RefTab, *UsrRef, *UsrEle;
   float    *CrdTab, *UsrCrd;
   size_t   DatLen, RefLen;

   if(EndIdx <= BegIdx)
      return(0);

   if(dat->AloTyp == GmlRawDat)
   {
      return(0);
   }
   else if(dat->AloTyp == GmlLnkDat)
   {
      return(0);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp == GmlVertices) )
   {
      CrdTab = (float *)dat->CpuMem;
      RefDat = &gml->dat[ RefIdx ];
      RefTab = (int *)RefDat->CpuMem;
      UsrCrd = (float *)DatBeg;
      UsrRef = (int *)RefBeg;
      DatLen = ((float *)DatEnd - (float *)DatBeg) / (EndIdx - BegIdx);
      RefLen = (RefEnd - RefBeg) / (EndIdx - BegIdx);

      for(i=BegIdx;i<=EndIdx;i++)
      {
         for(j=0;j<3;j++)
            CrdTab[ i * 4 + j ] = UsrCrd[ (i - BegIdx) * DatLen + j ];

         CrdTab[ i * 4 + 3 ] = 0.;

         if(UsrRef)
            RefTab[i] = UsrRef[ (i - BegIdx) * RefLen ];
      }
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->MshTyp > GmlVertices) )
   {
      EleTab = (int *)dat->CpuMem;
      siz = TypVecSiz[ MshItmTyp[ dat->MshTyp ] ];
      RefDat = &gml->dat[ gml->RefIdx[ dat->MshTyp ] ];
      RefTab = (int *)RefDat->CpuMem;
      UsrEle = (int *)DatBeg;
      UsrRef = (int *)RefBeg;
      DatLen = ((int *)DatEnd - (int *)DatBeg) / (EndIdx - BegIdx);
      RefLen = (RefEnd - RefBeg) / (EndIdx - BegIdx);

      for(i=BegIdx;i<=EndIdx;i++)
      {
         for(j=0; j<EleNmbNod[ dat->MshTyp ]; j++)
            EleTab[ i * siz + j ] =  UsrEle[ (i - BegIdx) * DatLen + j ];

         if(UsrRef)
            RefTab[i] = UsrRef[ (i - BegIdx) * RefLen ];
      }
   }

   if(EndIdx == dat->NmbLin - 1)
   {
      gml->MovSiz += UploadData(gml, DatIdx);
      gml->MovSiz += UploadData(gml, RefIdx);
   }

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
      return((int)dat->MemSiz);
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
      return((int)dat->MemSiz);
   }
}


/*----------------------------------------------------------------------------*/
/* Send the parameter structure data to the GPU                               */
/*----------------------------------------------------------------------------*/

int GmlUploadParameters(size_t idx)
{
   GETGMLPTR(gml, idx);

   if(!UploadData(gml, gml->ParIdx))
      return(0);
   else
      return(1);
}


/*----------------------------------------------------------------------------*/
/* Get the parameter structure data from the GPU                              */
/*----------------------------------------------------------------------------*/

int GmlDownloadParameters(size_t idx)
{
   GETGMLPTR(gml, idx);

   if(!DownloadData(gml, gml->ParIdx))
      return(0);
   else
      return(1);
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
   int      RefFlg, NmbHgh, HghVec, HghSiz, HghTyp, HghArg = -1, HghIdx = -1;
   char     *ParSrc, src[ GmlMaxSrcSiz ] = "\0", VoyNam[ GmlMaxStrSiz ];
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
         if(!gml->LnkMat[ MshTyp ][ dat->MshTyp ])
         {
            // Generate the default link between the two kinds of entities
            sprintf(BalNam, "%s%sLnk", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            sprintf(DegNam, "%s%sDeg", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            NewBallData(gml, MshTyp, DstTyp, BalNam, DegNam, NULL);
         }

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
            sprintf(VoyNam, "%s%sVoy", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            NewBallData(gml, MshTyp, DstTyp, BalNam, DegNam, VoyNam);
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
      SrcTyp = MshTyp;
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

         if( (HghIdx != -1) && (HghIdx != gml->LnkHgh[ MshTyp ][ DstTyp ]) )
         {
            puts("Current limitation prevents mixing two different kinds of uplink in the same kernel.");
            return(0);
         }

         HghArg      = arg->ArgIdx;
         HghIdx      = gml->LnkHgh[ MshTyp ][ DstTyp ];
         NmbHgh      = gml->SizMatHgh[ MshTyp ][ DstTyp ];

         // If this uplink requires voyeurs to be set, add the proper flag
         // to the argument and remove it from the user flag tab
         if(FlgTab[i] & GmlVoyeurs)
         {
            arg->VoyNam  =  gml->dat[ arg->DatIdx ].VoyNam;
            arg->FlgTab |=  GmlVoyeurs;
            FlgTab[i]   &= ~GmlVoyeurs;
         }

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

      if(FlgTab[i] & GmlRefFlag)
      {
         RefFlg = 1;
         FlgTab[i]   &= ~GmlRefFlag;
      }
      else
         RefFlg = 0;

      arg->NmbItm = dat->NmbItm;
      arg->ItmLen = dat->ItmLen;
      arg->ItmTyp = dat->ItmTyp;
      arg->FlgTab = FlgTab[i];
      arg->nam    = dat->nam;

      if(!RefFlg || (CptPos != -1))
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
   WriteToolkitSource      (src, toolkit);
   WriteUserToolkitSource  (src, gml->UsrTlk);
   WriteUserTypedef        (src, ParSrc);
   WriteProcedureHeader    (src, PrcNam, MshTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, MshTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, MshTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, MshTyp, NmbArg, ArgTab);

   // And Compile it
   KrnIdx = NewOclKrn      (gml, src, PrcNam);

   if(!KrnIdx)
      return(0);

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

   if(HghArg == -1 || !HghIdx)
      return(KrnIdx);

   // In case of uplink kernel, generate a second high degree kernel
   GetCntVec(NmbHgh, &HghVec, &HghSiz, &HghTyp);

   // Mofify the argument containing the uplink with the high count sizes
   ArgTab[ HghArg ].DatIdx = HghIdx;
   ArgTab[ HghArg ].MaxDeg = NmbHgh;
   ArgTab[ HghArg ].NmbItm = HghVec;
   ArgTab[ HghArg ].ItmLen = HghSiz;
   ArgTab[ HghArg ].ItmTyp = HghTyp;
   src[0] = '\0';

   // Generate the kernel source code
   WriteToolkitSource      (src, toolkit);
   WriteUserToolkitSource  (src, gml->UsrTlk);
   WriteUserTypedef        (src, ParSrc);
   WriteProcedureHeader    (src, PrcNam, MshTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, MshTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, MshTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, MshTyp, NmbArg, ArgTab);

   // And Compile it
   KrnHghIdx = NewOclKrn   (gml, src, PrcNam);

   if(!KrnHghIdx)
      return(0);

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
/* Add the geometrical toolkit prototypes and source code                     */
/*----------------------------------------------------------------------------*/

static void WriteToolkitSource(char *src, char *TlkSrc)
{
   strcat(src, "// GMlib MESHING AND GEOMETRICAL TOOLKIT\n");
   strcat(src, TlkSrc);
   strcat(src, "\n");
}


/*----------------------------------------------------------------------------*/
/* Add the geometrical toolkit prototypes and source code                     */
/*----------------------------------------------------------------------------*/

static void WriteUserToolkitSource(char *src, char *TlkSrc)
{
   if(!TlkSrc)
      return;

   strcat(src, "// USER'S TOOLKIT\n");
   strcat(src, TlkSrc);
   strcat(src, "\n");
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

      // If ball or shell voyeurs are required, define a vector char
      if(arg->FlgTab & GmlVoyeurs)
      {
         sprintf( str,  "   char      %s[%d];\n",
                  arg->VoyNam, arg->NmbItm * arg->ItmLen );
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
   int      i, j, k, l, siz;
   char     str   [ GmlMaxStrSiz ], ArgTd1[ GmlMaxStrSiz ], ArgTd2[ GmlMaxStrSiz ];
   char     LnkTd1[ GmlMaxStrSiz ], LnkTd2[ GmlMaxStrSiz ], LnkNam[ GmlMaxStrSiz ];
   char     CptNam[ GmlMaxStrSiz ], DegTst[ GmlMaxStrSiz ], DegNul[ GmlMaxStrSiz ];
   char     BalSft[ GmlMaxStrSiz ];
   ArgSct   *arg, *LnkArg, *CptArg;

   strcat (src, "   int       cnt = get_global_id(0);\n");
   sprintf(str, "   int       %sIdx = cnt + count.s1;\n\n", BalTypStr[ MshTyp ]);
   strcat (src, str);
   strcat (src, "   if(cnt >= count.s0)\n      return;\n\n");
   strcat (src, "// KERNEL MEMORY READINGS\n");

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
               sprintf(DegTst, "(%s >= %d) ?", CptNam, k + 1);
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

            // If voyeurs need to bet set, read the ball int vector, perform
            // a logical AND to get the four rightmost bits that store the voyeur
            // indices and convert the result to a vector char. Finaly perform
            // a 4 bits right shift to the ball data to get the right idices
            if(arg->FlgTab & GmlVoyeurs)
            {
               sprintf( str, "   %s%s = %sTab[ %s ]%s;\n",
                        arg->nam, ArgTd1, arg->nam, LnkNam, ArgTd1 );
               strcat(src, str);

               for(l=0;l<arg->ItmLen;l++)
               {
                  sprintf( str, "   %s[%d] = %s%s.s%c & 7;\n",
                           arg->VoyNam, j*arg->ItmLen + l,
                           arg->nam, ArgTd1, OclHexNmb[l] );
                  strcat(src, str);
               }

               sprintf( str, "   %s%s = %s%s %s;\n",
                        arg->nam, ArgTd1, arg->nam, ArgTd1, BalSft);
               strcat(src, str);
            }
            else
            {
               // Otherwise, read the ball data and perform the right shift on the fly
               sprintf( str, "   %s%s%s = %s %sTab[ %s%s%s ]%s %s %s;\n",
                        arg->nam, ArgTd2, ArgTd1, DegTst, arg->nam,
                        LnkNam, LnkTd1, LnkTd2, ArgTd1, BalSft, DegNul );

               strcat(src, str);
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

   p = (int)ceil(log2(siz));
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
         printf("binary executable of kernel %s: %zd bytes\n", PrcNam, len);
      else
         printf("could not get the size of kernel %s executable\n", PrcNam);
   }

   krn->EvtBlk = DEFEVTBLK;
   krn->NmbEvt = 0;
   krn->EvtTab = malloc(krn->EvtBlk * sizeof(cl_event));
   assert(krn->EvtTab);

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

int GmlLaunchKernel(size_t GmlIdx, int idx)
{
   GETGMLPTR(gml, GmlIdx);
   int      res;
   KrnSct   *krn = &gml->krn[ idx ];

   if( (idx < 1) || (idx > gml->NmbKrn) || !krn->kernel )
      return(-1);

   res = RunOclKrn(gml, krn);

   if(res != 1)
      return(res);

   if(krn->HghIdx)
      res = RunOclKrn(gml, &gml->krn[ krn->HghIdx ]);

   return(res);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

static int RunOclKrn(GmlSct *gml, KrnSct *krn)
{
   int      i, res;
   size_t   NmbGrp, GrpSiz, RetSiz = 0;
   DatSct   *dat;

   if(!krn->IniFlg)
   {
      for(i=0;i<krn->NmbDat;i++)
      {
         dat = &gml->dat[ krn->DatTab[i] ];

         if((krn->DatTab[i] < 1) || (krn->DatTab[i] > GmlMaxDat) || !dat->GpuMem)
         {
            printf(  "Invalid user argument %d, DatTab[i]=%d, GpuMem=%p\n",
                     i, krn->DatTab[i], dat->GpuMem );
            return(-1);
         }

         res = clSetKernelArg(krn->kernel, i, sizeof(cl_mem), &dat->GpuMem);

         if(res != CL_SUCCESS)
         {
            printf("Adding user argument %d failed with error: %d\n", i, res);
            return(-2);
         }
      }

      res = clSetKernelArg(krn->kernel, krn->NmbDat, sizeof(cl_mem),
                           &gml->dat[ gml->ParIdx ].GpuMem);

      if(res != CL_SUCCESS)
      {
         printf("Adding the GMlib parameters argument failed with error %d\n", res);
         return(-3);
      }

      res = clSetKernelArg(krn->kernel, krn->NmbDat+1, 2 * sizeof(int), krn->NmbLin);

      if(res != CL_SUCCESS)
      {
         printf("Adding the kernel loop counter argument failed with error %d\n", res);
         return(-4);
      }

      // Fit data loop size to the GPU kernel size
      res = clGetKernelWorkGroupInfo(  krn->kernel, gml->device_id[ gml->CurDev ],
                                       CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                       &GrpSiz, &RetSiz );

      if(res != CL_SUCCESS )
      {
         printf("Geting the kernel workgroup size failed with error %d\n", res);
         return(-5);
      }

      // Compute the hyperthreading level
      gml->CurGrpSiz = GrpSiz;
      NmbGrp = krn->NmbLin[0] / GrpSiz;
      NmbGrp *= GrpSiz;

      if(NmbGrp < krn->NmbLin[0])
         NmbGrp += GrpSiz;

      // Set the workgroup size and counter
      krn->IniFlg = 1;
      krn->NmbGrp = NmbGrp;
      krn->GrpSiz = GrpSiz;
   }

   if(krn->NmbEvt == krn->EvtBlk)
   {
      krn->EvtBlk *= 2;
      krn->EvtTab = realloc(krn->EvtTab, krn->EvtBlk * sizeof(cl_event));
      assert(krn->EvtTab);
   }

   // Launch GPU code
   if(clEnqueueNDRangeKernel( gml->queue, krn->kernel, 1, NULL, &krn->NmbGrp,
                              &krn->GrpSiz, 0, NULL, &krn->EvtTab[ krn->NmbEvt++ ]) )
   {
      return(-6);
   }

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Compute various reduction functions: min,max,L1,L2 norms                   */
/*----------------------------------------------------------------------------*/

int GmlReduceVector(size_t GmlIdx, int DatIdx, int RedOpp, double *nrm)
{
   GETGMLPTR(gml, GmlIdx);
   int      i, NmbLin, ret;
   float    res, *vec;
   DatSct   *dat, *red;
   KrnSct   *krn;
   char     *RedNam[] = { "reduce_min", "reduce_max", "reduce_sum",
            "reduce_L0", "reduce_L1", "reduce_L2", "reduce_Linf" };

   // Check indices and data conformity
   if( (DatIdx < 1) || (DatIdx > GmlMaxDat) )
   {
      printf("Invalid data index: %d\n", DatIdx);
      return(-1);
   }

   dat = &gml->dat[ DatIdx ];

   if( (dat->ItmTyp != GmlFlt) || (dat->NmbItm != 1) || (dat->ItmLen != 1) )
   {
      printf(  "Invalid data structure: count %d, type %s, length %d\n",
               dat->NmbItm, OclTypStr[ dat->ItmTyp ], dat->ItmLen );
      return(-2);
   }

   if( (RedOpp < 0) || (RedOpp > GmlMaxRed) )
   {
      printf("Invalid operation code %d\n", RedOpp);
      return(-3);
   }

   // Allocate an output vector the size of the input vector
   if(!dat->RedIdx)
      dat->RedIdx = GmlNewSolutionData(GmlIdx, dat->MshTyp, 1, GmlFlt, "reduce");

   if(!dat->RedIdx)
   {
      printf(  "Failed to allocate a reduction vector of %d bytes\n",
               OclTypSiz[ GmlFlt ] * dat->NmbLin );
      return(-4);
   }

   // Compile a reduction kernel with the required operation if needed
   if(!gml->RedKrn[ RedOpp ])
      gml->RedKrn[ RedOpp ] = NewOclKrn(gml, reduce, RedNam[ RedOpp ]);

   if(!gml->RedKrn[ RedOpp ])
   {
      printf("Failed to compile the %s reduction kernel\n", RedNam[ RedOpp ]);
      return(-5);
   }

   // Set the kernel with two vectors: an input and a reduced output one
   krn = &gml->krn[ gml->RedKrn[ RedOpp ] ];
   krn->NmbDat    = 2;
   krn->DatTab[0] = DatIdx;
   krn->DatTab[1] = dat->RedIdx;
   krn->NmbLin[0] = dat->NmbLin;

   // Launch the right reduction kernel according to the requested opperation
   ret = RunOclKrn(gml, krn);

   if(ret < 0)   
      return(ret);

   // Trim the size of the output vector down to the number of OpenCL groups
   // used by the kernel and download this amount of data
   red = &gml->dat[ dat->RedIdx ];
   red->MemSiz = dat->MemSiz / krn->GrpSiz;
   NmbLin = (int)(dat->NmbLin / krn->GrpSiz);
   DownloadData(gml, dat->RedIdx);
   red->MemSiz = dat->MemSiz;
   vec = (float *)red->CpuMem;

   // Perform the last reduction step on the CPU
   switch(RedOpp)
   {
      case GmlMin :
      {
         res = FLT_MAX;
         for(i=0;i<NmbLin;i++)
            res = MIN(res, vec[i]);
      }break;

      case GmlSum : case GmlL0 : case GmlL1 : case GmlL2 :
      {
         res = 0.;
         for(i=0;i<NmbLin;i++)
            res += vec[i];
      }break;

      case GmlMax : case GmlLinf :
      {
         res = -FLT_MAX;
         for(i=0;i<NmbLin;i++)
            res = MAX(res, vec[i]);
      }break;
   }

   *nrm = res;

   return(1);
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
/* Check the 64-bit floating point extension GPU's capacity                   */
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
   int         i, j, typ, cod, HshKey, ItmTab[3], (*EdgTab)[3] = NULL;
   int         EdgIdx, NmbItm, EleLen, NmbEdg = 0, *EleNod, *nod;
   int         OldNmbEdg, IdxLst[2], EdgNod[2];
   DatSct      *dat;
   HshTabSct   EdgHsh;

   GETGMLPTR(gml, GmlIdx);

   // Count the number of inner and surface edges
   for(typ=GmlEdges+1; typ<GmlMaxEleTyp; typ++)
      if(gml->TypIdx[ typ ])
         NmbEdg += gml->dat[ gml->TypIdx[ typ ] ].NmbLin;

   // Setup a hash table
   memset(&EdgHsh, 0, sizeof(HshTabSct));
   EdgHsh.HshTyp = GmlEdges;
   EdgHsh.TabSiz = NmbEdg;
   EdgHsh.DatLen = ItmNmbVer[ EdgHsh.HshTyp ];
   EdgHsh.KeyLen = HshLenTab[ EdgHsh.DatLen ];
   EdgHsh.NmbDat = EdgHsh.TabSiz;
   EdgHsh.NxtDat = 1;
   EdgHsh.HshTab = calloc(EdgHsh.TabSiz, sizeof(int));
   EdgHsh.DatTab = malloc(EdgHsh.TabSiz * sizeof(BucSct));
   printf("evaluated edges = %d\n",NmbEdg);
   NmbEdg = 0;

   if(!EdgHsh.HshTab || !EdgHsh.DatTab)
      return(0);

   if(gml->DbgFlg)
      printf(  "Hash table: lines=%d, stored items=%d, hash keys=%d\n",
               (int)EdgHsh.TabSiz, EdgHsh.DatLen, EdgHsh.KeyLen);

   for(typ=GmlEdges+1; typ<GmlMaxEleTyp; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      EleNod = (int *)dat->CpuMem;
      EleLen = dat->ItmLen;
      NmbItm = ItmNmbEdg[ typ ];

      // Add edges to the hash table
      for(i=0;i<dat->NmbLin;i++)
      {
         nod = &EleNod[ i * EleLen ];

         for(j=0;j<NmbItm;j++)
         {
            GetItmNod(nod, typ, EdgHsh.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&EdgHsh, ItmTab);

            if(GetHsh(&EdgHsh, HshKey, i, j, ItmTab, NULL, NULL))
               continue;

            AddHsh(&EdgHsh, HshKey, typ, i, j, ItmTab);
            NmbEdg++;
         }
      }
   }
   printf("inserted edges = %d\n",NmbEdg);

   if(gml->DbgFlg)
      printf(  "Hashed %d entities: occupency=%d%%, collisions=%g\n",
               (int)EdgHsh.NxtDat-1, (int)((100 * EdgHsh.NmbHit) / EdgHsh.TabSiz),
               (double)EdgHsh.NmbMis / (double)EdgHsh.TabSiz );

   // If there are surface edges, save their references and hash them
   GmlGetMeshInfo(GmlIdx, GmlEdges, &OldNmbEdg, &EdgIdx);

   if(OldNmbEdg)
   {
      EdgTab = malloc(OldNmbEdg * 3 * sizeof(int));

      if(!EdgTab)
         return(0);

      for(i=0;i<OldNmbEdg;i++)
          GmlGetDataLine(GmlIdx, EdgIdx, i, &EdgTab[i][0], &EdgTab[i][1], &EdgTab[i][2]);

      // Then free the existing edge data type
      // and allocate a new one with the increased size
      GmlFreeData(GmlIdx, EdgIdx);
      EdgIdx = GmlNewMeshData(GmlIdx, GmlEdges, NmbEdg);
      NmbEdg = 0;

      for(i=0;i<OldNmbEdg;i++)
      {
         GetItmNod(EdgTab[i], GmlEdges, GmlEdges, 0, EdgNod);
         HshKey = CalHshKey(&EdgHsh, EdgNod);

         // If it is in the hash table, send its data to the GMlib
         if(GetHsh(&EdgHsh, HshKey, i, 0, EdgNod, IdxLst, NULL) != 1)
            continue;

         if(IdxLst[0] >> 4 != i)
            continue;

         GmlSetDataLine(GmlIdx, EdgIdx, NmbEdg, EdgNod[0], EdgNod[1], EdgTab[i][2]);

         NmbEdg++;
      }
   }
   else
   {
      EdgIdx = GmlNewMeshData(GmlIdx, GmlEdges, NmbEdg);
      NmbEdg = 0;
   }

   // Loop over all kinds of elements and setup the inner edges
   for(typ=GmlEdges+1; typ<GmlMaxEleTyp; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      EleNod = (int *)dat->CpuMem;
      EleLen = dat->ItmLen;
      NmbItm = ItmNmbEdg[ typ ];

      // Get edges from the hash table
      for(i=0;i<dat->NmbLin;i++)
      {
         nod = &EleNod[ i * EleLen ];

         for(j=0;j<NmbItm;j++)
         {
            GetItmNod(nod, typ, EdgHsh.HshTyp, j, ItmTab);
            HshKey = CalHshKey(&EdgHsh, ItmTab);
            cod = 0;

            if(!GetHsh(&EdgHsh, HshKey, i, j, ItmTab, &cod, NULL))
               continue;

            if(cod >> 4 != i)
               continue;

            GmlSetDataLine(GmlIdx, EdgIdx, NmbEdg, ItmTab[0], ItmTab[1], 0);
            NmbEdg++;
         }
      }
   }

   if(EdgTab)
      free(EdgTab);

   free(EdgHsh.HshTab);
   free(EdgHsh.DatTab);

   if(gml->DbgFlg)
      printf("Hashed, setup and transfered %d edges to the GMlib.\n", NmbEdg);

   return(NmbEdg);
}


/*----------------------------------------------------------------------------*/
/* Build the list of inner faces and add them to the existing boundary ones   */
/*----------------------------------------------------------------------------*/

int GmlExtractFaces(size_t GmlIdx)
{
   int         i, j, typ, idx, TriIdx, QadIdx, HshKey;
   int         NmbFac, NmbTri = 0, NmbQad = 0, OldNmbTri, OldNmbQad;
   int         EleLen, *EleNod, *MshNod, FacNod[4], IdxLst[4];
   int         (*QadTab)[5], (*TriTab)[4], TypTab[4];
   DatSct      *dat;
   HshTabSct   TriHsh, QadHsh;

   GETGMLPTR(gml, GmlIdx);

   // Count the number of inner and surface triangles and quads
   for(typ=GmlTriangles; typ<GmlMaxEleTyp; typ++)
      if((idx = gml->TypIdx[ typ ]))
      {
         NmbTri += gml->dat[ idx ].NmbLin * ItmNmbTri[ typ ];
         NmbQad += gml->dat[ idx ].NmbLin * ItmNmbQad[ typ ];
      }

   // Setup a triangle hash table
   if(NmbTri)
   {
      memset(&TriHsh, 0, sizeof(HshTabSct));
      TriHsh.HshTyp = GmlTriangles;
      TriHsh.TabSiz = NmbTri;
      TriHsh.DatLen = ItmNmbVer[ TriHsh.HshTyp ];
      TriHsh.KeyLen = HshLenTab[ TriHsh.DatLen ];
      TriHsh.NmbDat = TriHsh.TabSiz;
      TriHsh.NxtDat = 1;
      TriHsh.HshTab = calloc(TriHsh.TabSiz, sizeof(int));
      TriHsh.DatTab = malloc(TriHsh.TabSiz * sizeof(BucSct));

      if(!TriHsh.HshTab || !TriHsh.DatTab)
         return(0);

      NmbTri = 0;

      if(gml->DbgFlg)
         printf(  "Triangle hash table: heads=%d, storage=%d, hash keys=%d\n",
                  (int)TriHsh.TabSiz, TriHsh.DatLen, TriHsh.KeyLen);
   }

   // Setup a quad hash table
   if(NmbQad)
   {
      memset(&QadHsh, 0, sizeof(HshTabSct));
      QadHsh.HshTyp = GmlQuadrilaterals;
      QadHsh.TabSiz = NmbQad;
      QadHsh.DatLen = ItmNmbVer[ QadHsh.HshTyp ];
      QadHsh.KeyLen = HshLenTab[ QadHsh.DatLen ];
      QadHsh.NmbDat = QadHsh.TabSiz;
      QadHsh.NxtDat = 1;
      QadHsh.HshTab = calloc(QadHsh.TabSiz, sizeof(int));
      QadHsh.DatTab = malloc(QadHsh.TabSiz * sizeof(BucSct));

      if(!QadHsh.HshTab || !QadHsh.DatTab)
         return(0);

      NmbQad = 0;

      if(gml->DbgFlg)
         printf(  "Quad hash table: heads=%d, storage=%d, hash keys=%d\n",
                  (int)QadHsh.TabSiz, QadHsh.DatLen, QadHsh.KeyLen);
   }

   // Loop over each element of each type
   for(typ=GmlTriangles; typ<GmlMaxEleTyp; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      // Get the nodes pointer, table width and the number of faces
      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      MshNod = (int *)dat->CpuMem;
      EleLen = dat->ItmLen;
      NmbFac = ItmNmbFac[ typ ];

      // Add faces to the dedicated hash table
      for(i=0;i<dat->NmbLin;i++)
      {
         EleNod = &MshNod[ i * EleLen ];

         for(j=0;j<NmbFac;j++)
         {
            // Add a triangle to the hash table
            if(ItmFacDeg[ typ ][j] == 3)
            {
               // Get its nodes and hash key
               GetItmNod(EleNod, typ, TriHsh.HshTyp, j, FacNod);
               HshKey = CalHshKey(&TriHsh, FacNod);

               // If it is not in the hash table, add it
               if(!GetHsh(&TriHsh, HshKey, i, j, FacNod, NULL, NULL))
               {
                  AddHsh(&TriHsh, HshKey, typ, i, j, FacNod);
                  NmbTri++;
               }
            }
            else if(ItmFacDeg[ typ ][j] == 4)
            {
               // Add a quad to the hash table
               GetItmNod(EleNod, typ, QadHsh.HshTyp, j, FacNod);
               HshKey = CalHshKey(&QadHsh, FacNod);

               // If it is not in the hash table, add it
               if(!GetHsh(&QadHsh, HshKey, i, j, FacNod, NULL, NULL))
               {
                  AddHsh(&QadHsh, HshKey, typ, i, j, FacNod);
                  NmbQad++;
               }
            }
            else
               continue;
         }
      }
   }

   if(gml->DbgFlg && NmbTri)
      printf(  "Hashed %d triangles: occupency=%d%%, collisions=%g\n",
               (int)TriHsh.NxtDat-1, (int)((100 * TriHsh.NmbHit) / TriHsh.TabSiz),
               (double)TriHsh.NmbMis / (double)TriHsh.TabSiz );

   if(gml->DbgFlg && NmbQad)
      printf(  "Hashed %d quads: occupency=%d%%, collisions=%g\n",
               (int)QadHsh.NxtDat-1, (int)((100 * QadHsh.NmbHit) / QadHsh.TabSiz),
               (double)QadHsh.NmbMis / (double)QadHsh.TabSiz );

   // Setup a new triangle data type and transfer the old data
   if(NmbTri)
   {
      // If there are surface triangles, save their references
      GmlGetMeshInfo(GmlIdx, GmlTriangles, &OldNmbTri, &TriIdx);

      TriTab = malloc(OldNmbTri * 4 * sizeof(int));

      if(!TriTab)
         return(0);

      for(i=0;i<OldNmbTri;i++)
          GmlGetDataLine(GmlIdx, TriIdx, i, &TriTab[i][0], &TriTab[i][1],
                        &TriTab[i][2], &TriTab[i][3]);

      // Then free the existing triangle data type
      // and allocate a new one with the increased size
      GmlFreeData(GmlIdx, TriIdx);
      TriIdx = GmlNewMeshData(GmlIdx, GmlTriangles, NmbTri);
      NmbTri = 0;

      for(i=0;i<OldNmbTri;i++)
      {
         GetItmNod(TriTab[i], GmlTriangles, GmlTriangles, 0, FacNod);
         HshKey = CalHshKey(&TriHsh, FacNod);

         // If it is in the hash table, send its data to the GMlib
         if(GetHsh(&TriHsh, HshKey, i, j, FacNod, IdxLst, NULL) != 1)
            continue;

         if(IdxLst[0] >> 4 != i)
            continue;

         GmlSetDataLine(GmlIdx, TriIdx, NmbTri, FacNod[0],
                        FacNod[1], FacNod[2], TriTab[i][3]);

         NmbTri++;
      }
   }

   // Setup a new quad data type
   if(NmbQad)
   {
      // If there are surface quads, save their references
      GmlGetMeshInfo(GmlIdx, GmlQuadrilaterals, &OldNmbQad, &QadIdx);
      QadTab = malloc(OldNmbQad * 5 * sizeof(int));

      if(!QadTab)
         return(0);

      for(i=0;i<OldNmbQad;i++)
         GmlGetDataLine(GmlIdx, QadIdx, i, &QadTab[i][0], &QadTab[i][1],
                        &QadTab[i][2], &QadTab[i][3], &QadTab[i][4]);

      // Then free the existing quad data type
      // and allocate a new one with the increased size
      GmlFreeData(GmlIdx, QadIdx);
      QadIdx = GmlNewMeshData(GmlIdx, GmlQuadrilaterals, NmbQad);
      NmbQad = 0;

      for(i=0;i<OldNmbQad;i++)
      {
         GetItmNod(QadTab[i], GmlQuadrilaterals, GmlQuadrilaterals, 0, FacNod);
         HshKey = CalHshKey(&QadHsh, FacNod);

         // If it is in the hash table, send its data to the GMlib
         if(GetHsh(&QadHsh, HshKey, i, j, FacNod, IdxLst, NULL) != 1)
            continue;

         if(IdxLst[0] >> 4 != i)
            continue;

         GmlSetDataLine(GmlIdx, QadIdx, NmbQad, FacNod[0],
                        FacNod[1], FacNod[2], FacNod[3], QadTab[i][3]);

         NmbQad++;
      }
   }

   // Loop over all kinds of elements and setup the faces
   for(typ=GmlTetrahedra; typ<=GmlHexahedra; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      // Get the nodes pointer, table width and the number of faces
      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      MshNod = (int *)dat->CpuMem;
      EleLen = dat->ItmLen;
      NmbFac = ItmNmbFac[ typ ];

      // Get edges from the hash table
      for(i=0;i<dat->NmbLin;i++)
      {
         EleNod = &MshNod[ i * EleLen ];

         for(j=0;j<NmbFac;j++)
         {
            // Get a triangle to the hash table
            if(ItmFacDeg[ typ ][j] == 3)
            {
               // Get its nodes and hash key
               GetItmNod(EleNod, typ, TriHsh.HshTyp, j, FacNod);
               HshKey = CalHshKey(&TriHsh, FacNod);

               // If it is in the hash table, send its data to the GMlib
               if(GetHsh(&TriHsh, HshKey, i, j, FacNod, IdxLst, TypTab) != 1)
                  continue;

               if( (IdxLst[0] >> 4 != i) || (TypTab[0] != typ) )
                  continue;

               GmlSetDataLine(GmlIdx, TriIdx, NmbTri, FacNod[0],
                              FacNod[1], FacNod[2], 0);

               NmbTri++;
            }
            else if(ItmFacDeg[ typ ][j] == 4)
            {
               // Get the quad's nodes and hash key
               GetItmNod(EleNod, typ, QadHsh.HshTyp, j, FacNod);
               HshKey = CalHshKey(&QadHsh, FacNod);

               // If it is in the hash table, send its data to the GMlib
               if(GetHsh(&QadHsh, HshKey, i, j, FacNod, IdxLst, TypTab) != 1)
                  continue;

               if( (IdxLst[0] >> 4 != i) || (TypTab[0] != typ) )
                  continue;

               GmlSetDataLine(GmlIdx, QadIdx, NmbQad, FacNod[0],
                              FacNod[1], FacNod[2], FacNod[3], 0);

               NmbQad++;
            }
            else
               continue;
         }
      }
   }

   // Cleanup hash and ref tables
   if(NmbTri)
   {
      free(TriTab);
      free(TriHsh.HshTab);
      free(TriHsh.DatTab);

      if(gml->DbgFlg)
         printf("Hashed, setup and transfered %d triangles to the GMlib.\n", NmbTri);
   }

   if(NmbQad)
   {
      free(QadTab);
      free(QadHsh.HshTab);
      free(QadHsh.DatTab);

      if(gml->DbgFlg)
         printf("Hashed, setup and transfered %d quads to the GMlib.\n", NmbQad);
   }

   // As there are two resulting values, return the sum instead of the data index
   return(NmbTri + NmbQad);
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

   if(!lnk.HshTab || !lnk.DatTab)
      return(0);

   if(gml->DbgFlg)
      printf(  "Hash table: lines=%d, stored items=%d, hash keys=%d\n",
               (int)lnk.TabSiz, lnk.DatLen, lnk.KeyLen);

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
         AddHsh(&lnk, HshKey, typ, i, j, ItmTab);
      }
   }

   if(gml->DbgFlg)
      printf(  "Hashed %d entities: occupency=%d%%, collisions=%g\n",
               (int)lnk.NmbDat, (int)((100 * lnk.NmbHit) / lnk.TabSiz),
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
         cpt = GetHsh(&lnk, HshKey, i, j, ItmTab, cod, NULL);
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


/*----------------------------------------------------------------------------*/
/* Return a mesh type number of lines and data index                          */
/*----------------------------------------------------------------------------*/

int GmlGetMeshInfo(size_t GmlIdx, int typ, int *NmbLin, int *DatIdx)
{
   GETGMLPTR   (gml, GmlIdx);
   CHKELETYP   (typ);
   DatSct      *dat;
   int         idx;

   if(!(idx = gml->TypIdx[ typ ]))
      return(0);

   if(!(dat = &gml->dat[ idx ]))
      return(0);

   if(!dat->NmbLin)
      return(0);

   if(NmbLin)
      *NmbLin = dat->NmbLin;

   if(DatIdx)
      *DatIdx = idx;

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Return an internal link lengths and widths                                 */
/*----------------------------------------------------------------------------*/

int GmlGetLinkInfo(  size_t GmlIdx, int SrcTyp, int DstTyp,
                     int *n, int *w, int *N, int *W )
{
   GETGMLPTR   (gml, GmlIdx);
   CHKELETYP   (SrcTyp);
   CHKELETYP   (DstTyp);
   int         BalIdx, HghIdx;
   DatSct      *BalDat, *HghDat;

   BalIdx = gml->LnkMat[ SrcTyp ][ DstTyp ];
   HghIdx = gml->LnkHgh[ SrcTyp ][ DstTyp ];

   if(!BalIdx)
      return(0);

   BalDat = &gml->dat[ BalIdx ];
   *n = BalDat->NmbLin;
   *w = BalDat->ItmLen;

   if(HghIdx)
   {
      HghDat = &gml->dat[ HghIdx ];
      *N = HghDat->NmbLin;
      *W = HghDat->ItmLen;
   }
   else
      *N = *W = 0;

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Compute the total kernel profiling time from the events table              */
/*----------------------------------------------------------------------------*/

double GmlGetKernelRunTime(size_t GmlIdx, int KrnIdx)
{
   GETGMLPTR(gml, GmlIdx);
   int      i;
   double   RunTim = 0.;
   cl_ulong start, end;
   KrnSct   *krn = &gml->krn[ KrnIdx ];

   if( (KrnIdx < 1) || (KrnIdx > gml->NmbKrn) || !krn->kernel )
      return(-1);

   for(i=0;i<krn->NmbEvt;i++)
   {
      if( (clGetEventProfilingInfo( krn->EvtTab[i], CL_PROFILING_COMMAND_QUEUED,
                                    sizeof(start), &start, NULL) == CL_SUCCESS)
      &&  (clGetEventProfilingInfo( krn->EvtTab[i], CL_PROFILING_COMMAND_END,
                                    sizeof(end),   &end,   NULL) == CL_SUCCESS) )
      {
         RunTim += (double)(end - start) * 1e-9;
      }
   }

   return(RunTim);
}


/*----------------------------------------------------------------------------*/
/* Compute the total reduction kernel profiling time from the events table    */
/*----------------------------------------------------------------------------*/

double GmlGetReduceRunTime(size_t GmlIdx, int RedOpp)
{
   GETGMLPTR(gml, GmlIdx);

   if( (RedOpp < 0) || (RedOpp > GmlMaxRed) )
   {
      printf("Invalid operation code %d\n", RedOpp);
      return(-1.);
   }

   return(GmlGetKernelRunTime(GmlIdx, gml->RedKrn[ RedOpp ]));
}


/*----------------------------------------------------------------------------*/
/* Return the wall clock in seconds                                           */
/*----------------------------------------------------------------------------*/

double GmlGetWallClock()
{
#ifdef _WIN32
   struct __timeb64 tb;
   _ftime64(&tb);
   return((double)tb.time + (double)tb.millitm/1000.);
#else
   struct timeval tp;
   gettimeofday(&tp, NULL);
   return(tp.tv_sec + tp.tv_usec / 1000000.);
#endif
}


/*----------------------------------------------------------------------------*/
/* Return the percentage of cache hit while doing an elements to nodes access */
/*----------------------------------------------------------------------------*/

float GmlEvaluateNumbering(size_t GmlIdx)
{
   GETGMLPTR(gml, GmlIdx);
   DatSct   *dat;
   int      i, j, typ, VerIdx, HshPos, HshAdr;
   int      HshTab[64]={0}, *EleTab, NmbHit = 0, NmbMis = 0;

   for(typ=GmlEdges; typ<GmlMaxEleTyp; typ++)
   {
      if(!gml->TypIdx[ typ ])
         continue;

      dat = &gml->dat[ gml->TypIdx[ typ ] ];
      EleTab = (int *)dat->CpuMem;

      for(i=0;i<dat->NmbLin;i++)
         for(j=0;j<dat->NmbItm;j++)
         {
            VerIdx = EleTab[ i * dat->ItmLen + j ];
            HshPos = (VerIdx >> 4) & 0xf;
            HshAdr = VerIdx >> 10;

            if(HshTab[ HshPos ] == HshAdr)
               NmbHit++;
            else
            {
               NmbMis++;
               HshTab[ HshPos ] = HshAdr;
            }
         }
   }

   return(100 * (float)NmbHit / (NmbHit + NmbMis));
}


/*----------------------------------------------------------------------------*/
/* Add a custom user's toolkit after the GMlib's toolkit                      */
/*----------------------------------------------------------------------------*/

void GmlIncludeUserToolkit(size_t GmlIdx, char *PtrSrc)
{
   GETGMLPTR(gml, GmlIdx);
   gml->UsrTlk = PtrSrc;
}


#ifdef WITH_LIBMESHB

/*----------------------------------------------------------------------------*/
/* Convert a libMeshb keyword into a GMlib one                                */
/*----------------------------------------------------------------------------*/

static int Gmf2Gml(int kwd)
{
   int GmfKwdTab[ GmlMaxEleTyp ] = {
      GmfVertices, GmfEdges, GmfTriangles, GmfQuadrilaterals,
      GmfTetrahedra, GmfPyramids, GmfPrisms, GmfHexahedra };

   for (int i=0;i<GmlMaxEleTyp;i++)
      if(kwd == GmfKwdTab[i])
         return(i);

   return(-1);
}


/*----------------------------------------------------------------------------*/
/* Read a mesh file and allocate and set the requested keywords in the GMlib  */
/*----------------------------------------------------------------------------*/

int GmlImportMesh(size_t GmlIdx, char *MshNam, ...)
{
   int         i, j, k, NmbLin, typ, ver, dim, kwd, DatIdx, NmbKwd = 0, EleSiz;
   int         KwdTab[10][4]={0}, *RefTab, *EleTab;
   float       (*CrdTab)[3];
   int64_t     InpMsh;
   va_list     VarArg;

   /*------------------------*/
   /* PARSE USER'S ARGUMENTS */
   /*------------------------*/

   va_start(VarArg, MshNam);

   while( (kwd = va_arg(VarArg, int)) && (NmbKwd < 10) )
   {
      typ = Gmf2Gml(kwd);

      if(typ == -1)
         continue;

      KwdTab[ NmbKwd ][0] = kwd;
      KwdTab[ NmbKwd ][1] = typ;
      NmbKwd++;
   }

   va_end(VarArg);


   /*--------------*/
   /* MESH READING */
   /*--------------*/

   // Open the mesh
   if( !(InpMsh = GmfOpenMesh(MshNam, GmfRead, &ver, &dim)) )
   {
      printf("Could not open file %s\n", MshNam);
      return(0);
   }

   for(k=0;k<NmbKwd;k++)
   {
      // Check of the required kwd exists in the mesh file
      if(!(NmbLin = (int)GmfStatKwd(InpMsh, KwdTab[k][0])))
         continue;

      typ = KwdTab[k][1];

      // Allocate the corresponding GMlib data type
      if(!(DatIdx = GmlNewMeshData(GmlIdx, typ, NmbLin)))
         continue;

      KwdTab[k][2] = NmbLin;
      KwdTab[k][3] = DatIdx;

      if(typ == GmlVertices)
      {
         CrdTab = malloc( (NmbLin+1) * 3 * sizeof(float));
         RefTab = malloc( (NmbLin+1)     * sizeof(int));

         if(!CrdTab || !RefTab)
            return(0);

         GmfGetBlock(InpMsh, GmfVertices, 1, NmbLin, 0, NULL, NULL,
                     GmfFloatVec, 3, CrdTab[1],  CrdTab[ NmbLin ],
                     GmfInt,        &RefTab[1], &RefTab[ NmbLin ]);

         GmlSetDataBlock(  GmlIdx, GmlVertices, 0, NmbLin-1,
                            CrdTab[1],  CrdTab[ NmbLin],
                           &RefTab[1], &RefTab[ NmbLin ]);

         free(CrdTab);
         free(RefTab);
      }
      else if( (typ >= GmlEdges) && (typ <= GmlHexahedra) )
      {
         EleSiz = EleNmbNod[ typ ];
         EleTab = malloc( (NmbLin+1) * EleSiz * sizeof(int));
         RefTab = malloc( (NmbLin+1) * sizeof(int));

         if(!EleTab || !RefTab)
            return(0);

         GmfGetBlock(InpMsh, KwdTab[k][0], 1, NmbLin, 0, NULL, NULL,
                     GmfIntVec, EleSiz, &EleTab[ 1 * EleSiz ], &EleTab[ NmbLin * EleSiz ],
                     GmfInt, &RefTab[1], &RefTab[ NmbLin ] );

         for(i=1;i<=NmbLin;i++)
            for(j=0;j<EleSiz;j++)
               EleTab[ i * EleSiz + j ]--;

         GmlSetDataBlock(  GmlIdx, typ, 0, NmbLin-1,
                           &EleTab[ 1 * EleSiz ], &EleTab[ NmbLin * EleSiz ],
                           &RefTab[ 1 ], &RefTab[ NmbLin ] );

         free(EleTab);
         free(RefTab);
      }
   }

   // And close the mesh
   GmfCloseMesh(InpMsh);

   return(NmbKwd);
}


/*----------------------------------------------------------------------------*/
/* Read a mesh file and allocate and set the requested keywords in the GMlib  */
/*----------------------------------------------------------------------------*/

int GmlExportSolution(size_t GmlIdx, char *SolNam, ...)
{
   GETGMLPTR   (gml, GmlIdx);
   char        RefStr[100], DimChr[4] = {'x', 'y', 'z', 't'};
   int         i, j, k, NmbLin, GmlTyp, GmfKwd, DatIdx, NmbDat = 0, NmbKwd = 0;
   int         DatTab[10][4], KwdDatTab[10][15] = {0}, NewKwdFlg, SolKwd, cpt;
   int         NmbTyp, TypTab[100], MshKwd, NmbArg, ArgTab[2][10], NmbFld;
   float       *AdrTab[10][2], *DatPtr, *PtrTab[2][10];
   DatSct      *dat;
   int64_t     OutSol;
   va_list     VarArg;

   va_start(VarArg, SolNam);

   // Scan each user's GML datatypes
   while( (DatIdx = va_arg(VarArg, int)) && (NmbDat < 10) )
   {
      if(!(dat = &gml->dat[ DatIdx ]))
         continue;

      // Add a new GML datatyp to the list and download its data from the GPU
      GmfKwd = GmfMshKwdTab[ dat->MshTyp ];
      DatPtr = (float *)dat->CpuMem;
      DatTab[ NmbDat ][0] = DatIdx;
      DatTab[ NmbDat ][1] = GmfTypTab[ dat->ItmTyp ];
      DatTab[ NmbDat ][2] = dat->ItmLen;
      AdrTab[ NmbDat ][0] = &DatPtr[ 0 ];
      AdrTab[ NmbDat ][1] = &DatPtr[ dat->NmbLin * dat->ItmLen ];
      DownloadData(gml, DatIdx);

      // Now, try to associate this GML type to a GMF keyword
      NewKwdFlg = 1;

      for(i=0;i<NmbKwd;i++)
         if(KwdDatTab[i][0] == GmfKwd)
         {
            // If a sol keyword was found, add this GML data to the field
            KwdDatTab[i][ KwdDatTab[i][1] + 4 ] = NmbDat;
            KwdDatTab[i][1]++;
            KwdDatTab[i][3] += dat->ItmLen;
            NewKwdFlg = 0;
            break;
         }

      // If not GMF solution was found, create a new one
      if(NewKwdFlg)
      {
         KwdDatTab[ NmbKwd ][0] = GmfKwd;
         KwdDatTab[ NmbKwd ][1] = 1;
         KwdDatTab[ NmbKwd ][2] = GmfTypTab[ dat->ItmTyp ];
         KwdDatTab[ NmbKwd ][3] = dat->ItmLen;
         KwdDatTab[ NmbKwd ][4] = NmbDat;
         NmbKwd++;
      }

      NmbDat++;
   }

   va_end(VarArg);

   // Create the sol file
   if( !(OutSol = GmfOpenMesh(SolNam, GmfWrite, 1, 3)) )
   {
      printf("Could not create file %s\n", SolNam);
      return(0);
   }

   // For each GMF solution keyword, set the header and write the data block
   for(i=0;i<NmbKwd;i++)
   {
      MshKwd = KwdDatTab[i][0];
      GmlTyp = Gmf2Gml(MshKwd);
      SolKwd = GmfSolKwdTab[ GmlTyp ];
      NmbTyp = NmbArg = 0;

      if(!(GmlGetMeshInfo(GmlIdx, GmlTyp, &NmbLin, &DatIdx)))
         continue;

      // This GMF keyword may be built out of several GML datatypes
      for(j=0;j<KwdDatTab[i][1];j++)
      {
         // Get the mesh type, vec size and start and end adresses
         DatIdx = KwdDatTab[i][ j+4 ];
         ArgTab[0][ NmbArg ] = GmfFloatVec;
         ArgTab[1][ NmbArg ] = DatTab[ DatIdx ][2];
         PtrTab[0][ NmbArg ] = AdrTab[ DatIdx ][0];
         PtrTab[1][ NmbArg ] = AdrTab[ DatIdx ][1];
         NmbArg++;

         // Use the GmfSca scalar type duplicated as many times as needed
         // because the arbitrary GML data sizes do not fit into the fixed
         // GMF scalar, 2D-3D vectors and matrices
         for(k=0;k<DatTab[ DatIdx ][2];k++)
            TypTab[ NmbTyp++ ] = GmfSca;
      }

      // Write the header and the field
      GmfSetKwd(OutSol, SolKwd, NmbLin, NmbTyp, TypTab);
      GmfSetBlock(OutSol, SolKwd, 1, NmbLin, 0, NULL, NULL, GmfArgTab,
                  ArgTab[0], ArgTab[1], PtrTab[0], PtrTab[1]);
   }

   // Count the total number of solution fields because the vector fields
   // need to be expanded into as many scalars
   cpt = 0;

   for(i=0;i<NmbKwd;i++)
      for(j=0; j<KwdDatTab[i][1]; j++)
         for(k=0; k<gml->dat[ DatTab[ KwdDatTab[i][ j+4 ] ][0] ].ItmLen; k++)
            cpt++;

   // Set the ref comment strings with user's data names
   GmfSetKwd(OutSol, GmfReferenceStrings, cpt);

   cpt = 0;

   // Write a line with a single scalar field
   for(i=0;i<NmbKwd;i++)
   {
      RefStr[0] = '\0';
      GmfKwd = Gmf2Gml(KwdDatTab[i][0]);
      SolKwd = GmfSolKwdTab[ GmfKwd ];
      NmbFld = KwdDatTab[i][1];

      for(j=0;j<NmbFld;j++)
      {
         DatIdx = DatTab[ KwdDatTab[i][ j+4 ] ][0];
         dat = &gml->dat[ DatIdx ];

         for(k=0;k<dat->ItmLen;k++)
         {
            cpt ++;

            if(dat->ItmLen > 1)
               sprintf(RefStr, "%s.%c %d", dat->nam, DimChr[k], cpt);
            else
               sprintf(RefStr, "%s %d", dat->nam, cpt);

            GmfSetLin(OutSol, GmfReferenceStrings, SolKwd, 1, RefStr);
         }
      }
   }

   // And close the mesh
   GmfCloseMesh(OutSol);

   return(NmbKwd);
}

#endif
