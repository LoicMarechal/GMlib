

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.01                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: feb 18 2020                                           */
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


/*----------------------------------------------------------------------------*/
/* Macro instructions                                                         */
/*----------------------------------------------------------------------------*/

#define MIN(a,b)     ((a) < (b) ? (a) : (b))
#define MAX(a,b)     ((a) > (b) ? (a) : (b))
#define MB           1048576
enum    data_type    {GmlArgDat, GmlRawDat, GmlLnkDat, GmlEleDat, GmlRefDat};


/*----------------------------------------------------------------------------*/
/* Library internal data structures                                           */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int            MshTyp, DatIdx, LnkTyp, LnkIdx, LnkDir, CntIdx;
   int            LnkDeg, MaxDeg, NmbItm, ItmLen, ItmTyp, FlgTab;
   char           *nam;
}ArgSct;

typedef struct
{
   int            AloTyp, MemAcs, MshTyp, LnkTyp, ItmTyp;
   int            NmbItm, ItmLen, ItmSiz, NmbLin, LinSiz;
   char           *nam, use;
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
   int            NmbKrn, CurDev, ParIdx, DbgFlg;
   int            TypIdx[ GmlMaxEleTyp ];
   int            RefIdx[ GmlMaxEleTyp ];
   int            NmbEle[ GmlMaxEleTyp ];
   int            LnkMat[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   int            LnkHgh[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   int            CntMat[ GmlMaxEleTyp ][ GmlMaxEleTyp ];
   cl_uint        NmbDev;
   size_t         MemSiz, CurGrpSiz, MovSiz;
   DatSct         dat[ GmlMaxDat + 1 ];
   KrnSct         krn[ GmlMaxKrn + 1 ];
   cl_device_id   device_id[ MaxGpu ];
   cl_context     context;
   cl_command_queue queue;
}GmlSct;


/*----------------------------------------------------------------------------*/
/* Prototypes of local procedures                                             */
/*----------------------------------------------------------------------------*/

static int     NewData                 (DatSct *);
static int     UploadData              (int);
static int     DownloadData            (int);
static int     NewKernel               (char *, char *);
static void    WriteUserTypedef        (char *, char *);
static void    WriteProcedureHeader    (char *, char *, int, int, ArgSct *);
static void    WriteKernelVariables    (char *, int, int, ArgSct *);
static void    WriteKernelMemoryReads  (char *, int, int, ArgSct *);
static void    WriteKernelMemoryWrites (char *, int, int, ArgSct *);
static void    WriteUserKernel         (char *, char *);
static void    GetCntVec               (int , int *, int *, int *);
static int     GetNewDatIdx            ();
static double  RunOclKrn               (KrnSct *);


/*----------------------------------------------------------------------------*/
/* Global library variables                                                   */
/*----------------------------------------------------------------------------*/

GmlSct gml;


/*----------------------------------------------------------------------------*/
/* Global tables                                                              */
/*----------------------------------------------------------------------------*/

char  OclHexNmb[16] = { '0','1','2','3','4','5','6','7',
                        '8','9','a','b','c','d','e','f' };

int   OclTypSiz[ GmlMaxOclTyp ] = {
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

char *OclTypStr[ GmlMaxOclTyp ]  = {
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

char *OclNulVec[ GmlMaxOclTyp ]  = {
   "(int){0}",
   "(int2){0,0}",
   "(int4){0,0,0,0}",
   "(int8){0,0,0,0,0,0,0,0}",
   "(int16){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}",
   "(float){0.}",
   "(float2){0.,0.}",
   "(float4){0.,0.,0.,0.}",
   "(float8){0.,0.,0.,0.,0.,0.,0.,0.}",
   "(float16){0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}" };

int   TypVecSiz[ GmlMaxOclTyp ]  = {1,2,4,8,16,1,2,4,8,16};

int   MshItmTyp[ GmlMaxEleTyp ]  = {GmlFlt4, GmlInt2, GmlInt4, GmlInt4,
                                    GmlInt4, GmlInt8, GmlInt8, GmlInt8};

int   EleNmbNod[ GmlMaxEleTyp ]  = {0, 2, 3, 4, 4, 5, 6, 8};
int   EleItmLen[ GmlMaxEleTyp ]  = {4, 2, 4, 4, 4, 8, 8, 8};
int   EleNmbItm[ GmlMaxEleTyp ]  = {1, 1, 1, 1, 1, 1, 1, 1};
int   MshTypDim[ GmlMaxEleTyp ]  = {0,1,2,2,3,3,3,3};

char *BalTypStr[ GmlMaxEleTyp ]  = {"Ver", "Edg", "Tri", "Qad",
                                    "Tet", "Pyr", "Pri", "Hex"};

char *MshTypStr[ GmlMaxEleTyp ]  = {"VerCrd", "EdgVer", "TriVer", "QadVer",
                                    "TetVer", "PyrVer", "PriVer", "HexVer"};

char *MshRefStr[ GmlMaxEleTyp ]  = {"VerRef", "EdgRef", "TriRef", "QadRef",
                                    "TetRef", "PyrRef", "PriRef", "HexRef"};

int LenMatBas[ GmlMaxEleTyp ][ GmlMaxEleTyp ] = {
   {0,16, 8, 4,32,16,16, 8},
   {2, 0, 2, 2, 8, 8, 8, 4},
   {4, 4, 0, 0, 2, 2, 2, 0},
   {4, 4, 0, 0, 0, 2, 2, 2},
   {4, 8, 4, 0, 0, 0, 0, 0},
   {8, 8, 4, 1, 0, 0, 0, 0},
   {8,16, 2, 4, 0, 0, 0, 0},
   {8,16, 0, 8, 0, 0, 0, 0} };

int SizMatHgh[ GmlMaxEleTyp ][ GmlMaxEleTyp ];

int LenMatMax[ GmlMaxEleTyp ][ GmlMaxEleTyp ] = {
   {0,64,16, 8,64,32,32,16},
   {2, 0, 2, 2,16,16,16, 8},
   {4, 4, 0, 0, 2, 2, 2, 0},
   {4, 4, 0, 0, 0, 2, 2, 2},
   {4, 8, 4, 0, 0, 0, 0, 0},
   {8, 8, 4, 1, 0, 0, 0, 0},
   {8,16, 2, 4, 0, 0, 0, 0},
   {8,16, 0, 8, 0, 0, 0, 0} };

char *sep="\n\n############################################################\n";


/*----------------------------------------------------------------------------*/
/* Init device, context and queue                                             */
/*----------------------------------------------------------------------------*/

GmlParSct *GmlInit(int DevIdx)
{
   int            err, res;
   cl_platform_id PlfTab[ GmlMaxOclTyp ];
   cl_uint        NmbPlf;

   // Select which device to run on
   memset(&gml, 0, sizeof(GmlSct));
   gml.CurDev = DevIdx;

   // Init the OpenCL software platform
   res = clGetPlatformIDs(10, PlfTab, &NmbPlf);

   if(res != CL_SUCCESS)
   {
      puts("Could not find a valid platform to compile and run OpenCL sources.");
      return(NULL);
   }

   // Get the list of available OpenCL devices
   res = clGetDeviceIDs(PlfTab[0], CL_DEVICE_TYPE_ALL,
                        MaxGpu, gml.device_id, &gml.NmbDev);

   if(res != CL_SUCCESS)
   {
      puts("Could not get the OpenCL devices list.");
      return(NULL);
   }

   // Check the user choosen device index against the bounds
   if( (DevIdx < 0) || (DevIdx >= gml.NmbDev) )
   {
      printf("Selected device Id is out of bounds (1 -> %d)\n", gml.NmbDev - 1);
      return(NULL);
   }

   // Create the context based on the platform and the selected device
   gml.context = clCreateContext(0, 1, &gml.device_id[ gml.CurDev ],
                                 NULL, NULL, &err);

   if(!gml.context)
   {
      printf("OpenCL context creation failed with error: %d\n", err);
      return(NULL);
   }

   // Create a command queue for this platform and device
   gml.queue = clCreateCommandQueue(gml.context, gml.device_id[ gml.CurDev ],
                                    CL_QUEUE_PROFILING_ENABLE, &err);

   if(!gml.queue)
   {
      printf("OpenCL command queue creation failed with error: %d\n", err);
      return(NULL);
   }

   // Allocate and return a public user customizable parameter structure
   if(!(gml.ParIdx = GmlNewParameters(sizeof(GmlParSct), "par")))
   {
      puts("Could not allocate the GMlib internal parameters structure.");
      return(NULL);
   }

   // Return a pointer on the allocated and initialize GMlib structure
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
   int            i, res;
   size_t         GpuNamSiz;
   char           GpuNam[100];
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
   int idx;
   DatSct *dat;

   if(!(idx = GetNewDatIdx()))
      return(0);

   dat = &gml.dat[ idx ];

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
   dat->MemSiz    = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(dat))
      return(0);

   if(gml.DbgFlg)
   {
      puts(sep);
      printf(  "Allocated a parameters data: index=%2d, size=%zu bytes\n",
               idx, dat->MemSiz );
   }

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Allocate one of the 8 mesh data types                                      */
/*----------------------------------------------------------------------------*/

int GmlNewMeshData(int MshTyp, int NmbLin)
{
   int      EleIdx, RefIdx;
   DatSct   *EleDat, *RefDat;

   if( (MshTyp < 0) || (MshTyp >= GmlMaxEleTyp) )
      return(0);

   if(!(EleIdx = GetNewDatIdx()))
      return(0);

   EleDat = &gml.dat[ EleIdx ];

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

   if(!NewData(EleDat))
      return(0);

   if(!(RefIdx = GetNewDatIdx()))
      return(0);

   RefDat = &gml.dat[ RefIdx ];

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

   if(!NewData(RefDat))
      return(0);

   gml.NmbEle[ MshTyp ] = NmbLin;
   gml.TypIdx[ MshTyp ] = EleIdx;
   gml.LnkMat[ MshTyp ][ GmlVertices ] = EleIdx;
   gml.CntMat[ MshTyp ][ MshTyp ] = RefIdx;
   gml.RefIdx[ MshTyp ] = RefIdx;

   if(gml.DbgFlg)
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

int GmlNewSolutionData(int MshTyp, int NmbDat, int ItmTyp, char *nam)
{
   int      idx;
   DatSct   *dat;

   if( (MshTyp < GmlVertices) || (MshTyp > GmlHexahedra) )
      return(0);

   if( (ItmTyp < GmlInt) || (ItmTyp > GmlFlt16) )
      return(0);

   if(!(idx = GetNewDatIdx()))
      return(0);

   dat = &gml.dat[ idx ];

   dat->AloTyp = GmlRawDat;
   dat->MshTyp = MshTyp;
   dat->LnkTyp = 0;
   dat->MemAcs = GmlInout;
   dat->ItmTyp = ItmTyp;
   dat->NmbItm = NmbDat;
   dat->ItmSiz = OclTypSiz[ ItmTyp ];
   dat->ItmLen = TypVecSiz[ ItmTyp ];
   dat->NmbLin = gml.NmbEle[ MshTyp ];
   dat->LinSiz = dat->NmbItm * dat->ItmSiz;
   dat->MemSiz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(dat))
      return(0);

   if(gml.DbgFlg)
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

int GmlNewLinkData(int MshTyp, int LnkTyp, int NmbDat, char *nam)
{
   int      LnkIdx, VecCnt, VecSiz, ItmTyp;
   DatSct   *dat;

   if( (MshTyp < GmlVertices) || (MshTyp > GmlHexahedra) )
      return(0);

   if( (LnkTyp < GmlVertices) || (LnkTyp > GmlHexahedra) )
      return(0);

   if(!(LnkIdx = GetNewDatIdx()))
      return(0);

   GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

   dat = &gml.dat[ LnkIdx ];

   dat->AloTyp = GmlLnkDat;
   dat->MshTyp = MshTyp;
   dat->LnkTyp = LnkTyp;
   dat->MemAcs = GmlInout;
   dat->ItmTyp = ItmTyp;
   dat->NmbItm = VecCnt;
   dat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
   dat->ItmLen = 1;
   dat->NmbLin = gml.NmbEle[ MshTyp ];
   dat->LinSiz = dat->NmbItm * dat->ItmSiz;
   dat->MemSiz = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(dat))
      return(0);

   gml.LnkMat[ MshTyp ][ LnkTyp ] = LnkIdx;

   if(gml.DbgFlg)
   {
      puts(sep);
      printf(  "Allocated link data: index=%2d, size=%zu bytes\n",
               LnkIdx, dat->MemSiz );
   }

   return(LnkIdx);
}


/*----------------------------------------------------------------------------*/
/* Create an arbitrary link table between two kinds of mesh data types        */
/*----------------------------------------------------------------------------*/

int GmlNewBallData(int MshTyp, int LnkTyp, char *BalNam, char *DegNam)
{
   int         i, j, BalIdx, HghIdx, DegIdx, VerIdx, SrcIdx, DstIdx;
   int         EleSiz, VecSiz, BalSiz, MaxSiz, HghSiz = 0;
   int         *EleTab, *BalTab, *DegTab, *HghTab;
   int         MaxDeg = 0, MaxPos = 0, VecCnt, ItmTyp, NmbDat;
   int64_t     DegTot = 0;
   DatSct      *src, *dst, *bal, *hgh, *deg, *BalDat, *HghDat, *DegDat;


   // Check and prepare some data
   if( (MshTyp < 0) || (MshTyp >= GmlMaxEleTyp) )
      return(0);

   if( (LnkTyp < 0) || (LnkTyp >= GmlMaxEleTyp) )
      return(0);

   if(!BalNam || !DegNam)
      return(0);

   SrcIdx = gml.TypIdx[ MshTyp ];
   DstIdx = gml.TypIdx[ LnkTyp ];
   src = &gml.dat[ SrcIdx ];
   dst = &gml.dat[ DstIdx ];


   // Build the degrees table: count and allocate gml data,
   // base and high vector sizes for the next two phases
   if(!(DegIdx = GetNewDatIdx()))
      return(0);

   DegDat = &gml.dat[ DegIdx ];

   DegDat->AloTyp = GmlLnkDat;
   DegDat->MshTyp = MshTyp;
   DegDat->LnkTyp = LnkTyp;
   DegDat->MemAcs = GmlInout;
   DegDat->ItmTyp = GmlInt;
   DegDat->NmbItm = 1;
   DegDat->ItmSiz = OclTypSiz[ GmlInt ];
   DegDat->ItmLen = 0;
   DegDat->NmbLin = gml.NmbEle[ MshTyp ];
   DegDat->LinSiz = DegDat->NmbItm * DegDat->ItmSiz;
   DegDat->MemSiz = (size_t)DegDat->NmbLin * (size_t)DegDat->LinSiz;
   DegDat->GpuMem = DegDat->CpuMem = NULL;
   DegDat->nam    = DegNam;

   if(!NewData(DegDat))
      return(0);

   gml.CntMat[ MshTyp ][ LnkTyp ] = DegIdx;
   deg = &gml.dat[ gml.CntMat[ src->MshTyp ][ dst->MshTyp ] ];

   DegTab = deg->CpuMem;
   EleTab = dst->CpuMem;
   BalSiz = LenMatBas[ src->MshTyp ][ dst->MshTyp ];
   MaxSiz = LenMatMax[ src->MshTyp ][ dst->MshTyp ];
   EleSiz = EleNmbNod[ dst->MshTyp ];

   for(i=0;i<dst->NmbLin;i++)
      for(j=0;j<EleSiz;j++)
         DegTab[ EleTab[ i * EleSiz + j ] ]++;

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

   // Allocate the base vector ball table
   if(!(BalIdx = GetNewDatIdx()))
      return(0);

   NmbDat = LenMatBas[ MshTyp ][ LnkTyp ];
   GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

   BalDat = &gml.dat[ BalIdx ];

   BalDat->AloTyp = GmlLnkDat;
   BalDat->MshTyp = MshTyp;
   BalDat->LnkTyp = LnkTyp;
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

   if(!NewData(BalDat))
      return(0);

   gml.LnkMat[ MshTyp ][ LnkTyp ] = BalIdx;
   bal = &gml.dat[ gml.LnkMat[ src->MshTyp ][ dst->MshTyp ] ];


   // Allocate the high vector ball table
   if(!(HghIdx = GetNewDatIdx()))
      return(0);

   NmbDat = SizMatHgh[ MshTyp ][ LnkTyp ];
   GetCntVec(NmbDat, &VecCnt, &VecSiz, &ItmTyp);

   HghDat = &gml.dat[ HghIdx ];

   HghDat->AloTyp = GmlLnkDat;
   HghDat->MshTyp = MshTyp;
   HghDat->LnkTyp = LnkTyp;
   HghDat->MemAcs = GmlInout;
   HghDat->ItmTyp = ItmTyp;
   HghDat->NmbItm = VecCnt;
   HghDat->ItmSiz = VecCnt * OclTypSiz[ ItmTyp ];
   HghDat->ItmLen = 1;
   HghDat->NmbLin = gml.NmbEle[ MshTyp ] - MaxPos;
   HghDat->LinSiz = HghDat->NmbItm * HghDat->ItmSiz;
   HghDat->MemSiz = (size_t)HghDat->NmbLin * (size_t)HghDat->LinSiz;
   HghDat->GpuMem = HghDat->CpuMem = NULL;
   HghDat->nam    = BalNam;

   if(!NewData(HghDat))
      return(0);

   gml.LnkHgh[ MshTyp ][ LnkTyp ] = HghIdx;
   hgh = &gml.dat[ gml.LnkHgh[ src->MshTyp ][ dst->MshTyp ] ];


   // Fill both ball tables at the same time
   BalTab = bal->CpuMem;
   HghTab = hgh->CpuMem;

   for(i=0;i<src->NmbLin;i++)
      DegTab[i] = 0;

   for(i=0;i<dst->NmbLin;i++)
      for(j=0;j<EleSiz;j++)
      {
         VerIdx = EleTab[ i * EleSiz + j ];

         if(VerIdx < MaxPos)
            BalTab[ VerIdx * BalSiz + DegTab[ VerIdx ] ] = i;
         else if(DegTab[ VerIdx ] < HghSiz)
            HghTab[ (VerIdx - MaxPos) * HghSiz + DegTab[ VerIdx ] ] = i;

         DegTab[ VerIdx ]++;
      }

   // Upload the ball data to th GPU memory
   gml.MovSiz += UploadData(BalIdx);
   gml.MovSiz += UploadData(HghIdx);
   gml.MovSiz += UploadData(DegIdx);

   if(gml.DbgFlg)
   {
      puts(sep);
      printf(  "Ball generation: type %s -> %s\n",
               BalTypStr[ MshTyp ], BalTypStr[ LnkTyp ] );
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

   return(BalIdx);
}


/*----------------------------------------------------------------------------*/
/* Find and return a free data slot int the GML structure                     */
/*----------------------------------------------------------------------------*/

static int GetNewDatIdx()
{
   for(int i=1;i<=GmlMaxDat;i++)
      if(!gml.dat[i].use)
      {
         gml.dat[i].use = 1;
         return(i);
      }

   return(0);
}


/*----------------------------------------------------------------------------*/
/* Allocate an OpenCL buffer plus 10% more for resizing                       */
/*----------------------------------------------------------------------------*/

static int NewData(DatSct *dat)
{
   int MemAcs[4] = {0, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE};

   // Allocate the requested memory size on the GPU
   dat->GpuMem = clCreateBuffer( gml.context, MemAcs[ dat->MemAcs ],
                                 dat->MemSiz, NULL, NULL );

   if(!dat->GpuMem)
   {
      printf(  "Cannot allocate %ld MB on the GPU (%ld MB already used)\n",
               dat->MemSiz / MB, GmlGetMemoryUsage() / MB);
      return(0);
   }

   // Allocate the requested memory size on the CPU side
   if( (dat->MemAcs != GmlInternal) && !(dat->CpuMem = calloc(1, dat->MemSiz)) )
   {
      printf("Cannot allocate %ld MB on the CPU\n", dat->MemSiz/MB);
      return(0);
   }

   // Keep track of allocated memory
   gml.MemSiz += dat->MemSiz;

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Release an OpenCL buffer                                                   */
/*----------------------------------------------------------------------------*/

int GmlFreeData(int idx)
{
   DatSct *dat = &gml.dat[ idx ];

   // Free both GPU and CPU memory buffers
   if( (idx >= 1) && (idx <= GmlMaxDat) && dat->GpuMem )
   {
      if(clReleaseMemObject(dat->GpuMem) != CL_SUCCESS)
         return(0);

      dat->GpuMem = NULL;
      dat->use = 0;
      gml.MemSiz -= dat->MemSiz;

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

int GmlSetDataLine(int idx, int lin, ...)
{
   DatSct   *dat = &gml.dat[ idx ], *RefDat;
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
      RefDat = &gml.dat[ gml.CntMat[ dat->MshTyp ][ dat->MshTyp ] ];
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
      RefDat = &gml.dat[ gml.CntMat[ dat->MshTyp ][ dat->MshTyp ] ];
      RefTab = (int *)RefDat->CpuMem;

      for(i=0;i<EleNmbNod[ dat->MshTyp ];i++)
         EleTab[ lin * siz + i ] = va_arg(VarArg, int);

      RefTab[ lin ] = va_arg(VarArg, int);
   }

   va_end(VarArg);

   if(lin == dat->NmbLin - 1)
      gml.MovSiz += UploadData(idx);

   return(1);
}


/*----------------------------------------------------------------------------*/
/* Get a line of solution or vertex corrdinates from the librarie's buffers   */
/*----------------------------------------------------------------------------*/

int GmlGetDataLine(int idx, int lin, ...)
{
   DatSct   *dat = &gml.dat[ idx ];
   char     *adr = (void *)dat->CpuMem;
   int      i;
   float    *GpuCrd;
   double   *UsrCrd;
   va_list  VarArg;

   if(lin == 0)
      gml.MovSiz += DownloadData(idx);

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

static int UploadData(int idx)
{
   int      res;
   DatSct   *dat = &gml.dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem
   || !dat->CpuMem || (dat->MemAcs == GmlOutput) )
   {
      return(0);
   }

   // Upload buffer from CPU ram to GPU ram
   // and keep track of the amount of uploaded data
   res = clEnqueueWriteBuffer(gml.queue, dat->GpuMem, CL_FALSE, 0,
                              dat->MemSiz, dat->CpuMem, 0, NULL,NULL);

   if(res != CL_SUCCESS)
   {
      printf("Uploading the data to the GPu failed with error %d\n", res);
      return(0);
   }
   else
   {
      gml.MovSiz += dat->MemSiz;
      return(dat->MemSiz);
   }
}


/*----------------------------------------------------------------------------*/
/* Copy an OpenCL buffer into user's data                                     */
/*----------------------------------------------------------------------------*/

static int DownloadData(int idx)
{
   int      res;
   DatSct   *dat = &gml.dat[ idx ];

   // Check indices
   if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || !dat->CpuMem )
      return(0);

   // Download buffer from GPU ram to CPU ram
   // and keep track of the amount of downloaded data
   res = clEnqueueReadBuffer( gml.queue, dat->GpuMem, CL_TRUE, 0,
                              dat->MemSiz, dat->CpuMem, 0, NULL, NULL );

   if(res != CL_SUCCESS)
   {
      printf("Downloading the data from the GPu failed with error %d\n", res);
      return(0);
   }
   else
   {
      gml.MovSiz += dat->MemSiz;
      return(dat->MemSiz);
   }
}


/*----------------------------------------------------------------------------*/
/* Generate the kernel from user's source and data and compile it             */
/*----------------------------------------------------------------------------*/

int GmlCompileKernel(char *KrnSrc, char *PrcNam, char *DefSrc,
                     int MshTyp, int NmbTyp, ...)
{
   int      i, j, flg, KrnIdx, KrnHghIdx, SrcTyp, DstTyp, NmbArg = 0, RefIdx;
   int      FlgTab[ GmlMaxDat ], IdxTab[ GmlMaxDat ];
   int      LnkTab[ GmlMaxDat ], CntTab[ GmlMaxDat ];
   int      LnkItm, NmbItm, ItmTyp, ItmLen, LnkPos, CptPos, ArgHghPos;
   int      NmbHgh, HghVec, HghSiz, HghTyp, HghArg = -1, HghIdx = -1;
   char     src[10000] = "\0", BalNam[100], DegNam[100];
   va_list  VarArg;
   DatSct   *dat, *RefDat;
   ArgSct   ArgTab[ GmlMaxOclTyp ];
   KrnSct   *krn;

   // Decode datatypes arguments
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
      dat = &gml.dat[ IdxTab[i] ];

      if( LnkTab[i] || (dat->MshTyp == MshTyp) )
         continue;

      SrcTyp = MshTyp;
      DstTyp = dat->MshTyp;

      if(MshTypDim[ SrcTyp ] > MshTypDim[ DstTyp ])
      {
         LnkTab[i] = gml.LnkMat[ MshTyp ][ DstTyp ];
         CntTab[i] = 0;
      }
      else if(MshTypDim[ SrcTyp ] == MshTypDim[ DstTyp ])
      {
         // build neighbours
      }
      else
      {
         if(!gml.LnkMat[ MshTyp ][ dat->MshTyp ])
         {
            // Generate the default link between the two kinds of entities
            sprintf(BalNam, "%s%sBal", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            sprintf(DegNam, "%s%sDeg", BalTypStr[ MshTyp ], BalTypStr[ DstTyp ]);
            GmlNewBallData(MshTyp, DstTyp, BalNam, DegNam);
         }

         LnkTab[i] = gml.LnkMat[ MshTyp ][ DstTyp ];
         CntTab[i] = gml.CntMat[ MshTyp ][ DstTyp ];
      }
   }

   for(i=0;i<NmbTyp;i++)
   {
      if(!LnkTab[i])
         continue;

      dat = &gml.dat[ IdxTab[i] ];
      flg = 0;

      for(j=0;j<NmbArg;j++)
         if(ArgTab[j].DatIdx == LnkTab[i])
            flg = 1;

      if(flg)
         continue;

      DstTyp = dat->MshTyp;
      LnkItm = LenMatBas[ MshTyp ][ DstTyp ];
      GetCntVec(LnkItm, &NmbItm, &ItmLen, &ItmTyp);

      if(MshTypDim[ SrcTyp ] > MshTypDim[ DstTyp ])
      {
         ArgTab[ NmbArg ].MshTyp = DstTyp;
         ArgTab[ NmbArg ].DatIdx = LnkTab[i];
         ArgTab[ NmbArg ].LnkDir = -1;
         ArgTab[ NmbArg ].LnkTyp = -1;
         ArgTab[ NmbArg ].LnkIdx = -1;
         ArgTab[ NmbArg ].CntIdx = -1;
         ArgTab[ NmbArg ].LnkDeg = LnkItm;
         ArgTab[ NmbArg ].MaxDeg = LnkItm;
         ArgTab[ NmbArg ].NmbItm = NmbItm;
         ArgTab[ NmbArg ].ItmLen = ItmLen;
         ArgTab[ NmbArg ].ItmTyp = ItmTyp;
         ArgTab[ NmbArg ].FlgTab = GmlReadMode;
         ArgTab[ NmbArg ].nam = gml.dat[ ArgTab[ NmbArg ].DatIdx ].nam;
         NmbArg++;
      }
      else if(MshTypDim[ SrcTyp ] == MshTypDim[ DstTyp ])
      {
      }
      else if(MshTypDim[ SrcTyp ] < MshTypDim[ DstTyp ])
      {
         ArgTab[ NmbArg ].MshTyp = DstTyp;
         ArgTab[ NmbArg ].DatIdx = LnkTab[i];
         ArgTab[ NmbArg ].LnkDir = 1;
         ArgTab[ NmbArg ].LnkTyp = -1;
         ArgTab[ NmbArg ].LnkIdx = -1;
         ArgTab[ NmbArg ].CntIdx = -1;
         ArgTab[ NmbArg ].LnkDeg = -1;
         ArgTab[ NmbArg ].MaxDeg = LnkItm;
         ArgTab[ NmbArg ].NmbItm = NmbItm;
         ArgTab[ NmbArg ].ItmLen = ItmLen;
         ArgTab[ NmbArg ].ItmTyp = ItmTyp;
         ArgTab[ NmbArg ].FlgTab = GmlReadMode;
         ArgTab[ NmbArg ].nam = gml.dat[ ArgTab[ NmbArg ].DatIdx ].nam;
         HghArg = NmbArg;
         HghIdx = gml.LnkHgh[ MshTyp ][ DstTyp ];
         NmbArg++;

         ArgTab[ NmbArg ].MshTyp = DstTyp;
         ArgTab[ NmbArg ].DatIdx = gml.CntMat[ MshTyp ][ DstTyp ];
         ArgTab[ NmbArg ].LnkDir = 0;
         ArgTab[ NmbArg ].LnkTyp = -1;
         ArgTab[ NmbArg ].LnkIdx = -1;
         ArgTab[ NmbArg ].CntIdx = -1;
         ArgTab[ NmbArg ].LnkDeg = 1;
         ArgTab[ NmbArg ].MaxDeg = 1;
         ArgTab[ NmbArg ].NmbItm = 1;
         ArgTab[ NmbArg ].ItmLen = 1;
         ArgTab[ NmbArg ].ItmTyp = GmlInt;
         ArgTab[ NmbArg ].FlgTab = GmlReadMode;
         ArgTab[ NmbArg ].nam = gml.dat[ ArgTab[ NmbArg ].DatIdx ].nam;
         NmbArg++;
      }
   }

   for(i=0;i<NmbTyp;i++)
   {
      dat = &gml.dat[ IdxTab[i] ];
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

      ArgTab[ NmbArg ].MshTyp = DstTyp;
      ArgTab[ NmbArg ].DatIdx = IdxTab[i];
      ArgTab[ NmbArg ].LnkTyp = DstTyp;
      ArgTab[ NmbArg ].LnkIdx = LnkPos;
      ArgTab[ NmbArg ].CntIdx = CptPos;

      if(LnkPos != -1 && CptPos != -1)
      {
         ArgTab[ NmbArg ].LnkDeg = -1;
         ArgTab[ NmbArg ].MaxDeg = ArgTab[ LnkPos ].MaxDeg;
         ArgHghPos = NmbArg;
      }
      else if(LnkPos != -1 && CptPos == -1)
      {
         ArgTab[ NmbArg ].LnkDeg = ArgTab[ LnkPos ].LnkDeg;
         ArgTab[ NmbArg ].MaxDeg = ArgTab[ LnkPos ].MaxDeg;
      }
      else
      {
         ArgTab[ NmbArg ].LnkDeg = -1;
         ArgTab[ NmbArg ].MaxDeg = -1;
      }

      ArgTab[ NmbArg ].NmbItm = dat->NmbItm;
      ArgTab[ NmbArg ].ItmLen = dat->ItmLen;
      ArgTab[ NmbArg ].ItmTyp = dat->ItmTyp;
      ArgTab[ NmbArg ].FlgTab = FlgTab[i];
      ArgTab[ NmbArg ].nam = dat->nam;
      NmbArg++;

      if(!(FlgTab[i] & GmlRefFlag) || (CptPos != -1))
         continue;

      RefIdx = gml.RefIdx[ DstTyp ];
      RefDat = &gml.dat[ RefIdx ];
      ArgTab[ NmbArg ].MshTyp = DstTyp;
      ArgTab[ NmbArg ].DatIdx = RefIdx;
      ArgTab[ NmbArg ].LnkTyp = DstTyp;
      ArgTab[ NmbArg ].LnkIdx = LnkPos;
      ArgTab[ NmbArg ].CntIdx = CptPos;
      ArgTab[ NmbArg ].LnkDeg = -1;
      ArgTab[ NmbArg ].MaxDeg = ArgTab[ LnkPos ].MaxDeg;
      ArgTab[ NmbArg ].NmbItm = RefDat->NmbItm;
      ArgTab[ NmbArg ].ItmLen = RefDat->ItmLen;
      ArgTab[ NmbArg ].ItmTyp = RefDat->ItmTyp;
      ArgTab[ NmbArg ].FlgTab = FlgTab[i];
      ArgTab[ NmbArg ].nam = RefDat->nam;
      NmbArg++;
   }

   // Generate the kernel source code
   WriteUserTypedef        (src, DefSrc);
   WriteProcedureHeader    (src, PrcNam, MshTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, MshTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, MshTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, MshTyp, NmbArg, ArgTab);

   // And Compile it
   KrnIdx = NewKernel      (src, PrcNam);

   if(gml.DbgFlg)
   {
      puts(sep);
      printf("Generated source for kernel=%s, index=%2d\n", PrcNam, KrnIdx);
      puts(src);
   }

   krn = &gml.krn[ KrnIdx ];
   krn->NmbDat = NmbArg;
   krn->NmbLin[1] = 0;

   if(HghIdx == -1)
      krn->NmbLin[0] = gml.dat[ gml.TypIdx[ MshTyp ] ].NmbLin;
   else
      krn->NmbLin[0] = gml.dat[ gml.TypIdx[ MshTyp ] ].NmbLin - gml.dat[ HghIdx ].NmbLin;

   for(i=0;i<NmbArg;i++)
      krn->DatTab[i] = ArgTab[i].DatIdx;

   if(HghArg == -1)
      return(KrnIdx);

   // In case of uplink kernel, generate a second high degree kernel
   NmbHgh = SizMatHgh[ MshTyp ][ DstTyp ];
   GetCntVec(NmbHgh, &HghVec, &HghSiz, &HghTyp);

   ArgTab[ HghArg ].DatIdx = HghIdx;
   ArgTab[ HghArg ].MaxDeg = NmbHgh;
   ArgTab[ HghArg ].NmbItm = HghVec;
   ArgTab[ HghArg ].ItmLen = HghSiz;
   ArgTab[ HghArg ].ItmTyp = HghTyp;
   src[0] = '\0';

   // Generate the kernel source code
   WriteUserTypedef        (src, DefSrc);
   WriteProcedureHeader    (src, PrcNam, MshTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, MshTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, MshTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, MshTyp, NmbArg, ArgTab);

   // And Compile it
   KrnHghIdx = NewKernel   (src, PrcNam);
   gml.krn[ KrnIdx ].HghIdx = KrnHghIdx;

   krn = &gml.krn[ KrnHghIdx ];
   krn->NmbDat = NmbArg;
   krn->NmbLin[0] = gml.dat[ HghIdx ].NmbLin;
   krn->NmbLin[1] = gml.dat[ gml.TypIdx[ MshTyp ] ].NmbLin - gml.dat[ HghIdx ].NmbLin;

   for(i=0;i<NmbArg;i++)
      krn->DatTab[i] = ArgTab[i].DatIdx;

   if(gml.DbgFlg)
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

static void WriteUserTypedef(char *src, char *TypSct)
{
   strcat(src, "// USER'S ARGUMENTS STRUCTURE\n");
   strcat(src, TypSct);
   strcat(src, "\n");
}


/*----------------------------------------------------------------------------*/
/* Write the procedure name and typedef with all arguments types and names    */
/*----------------------------------------------------------------------------*/

static void WriteProcedureHeader(char *src, char *PrcNam, int MshTyp,
                                 int NmbArg, ArgSct *ArgTab)
{
   int      i;
   char     str[100];
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

   strcat(src, "\n   __global GmlParSct *par,");
   strcat(src, "\n   const    int2       count )\n{\n");
}


/*----------------------------------------------------------------------------*/
/* Write definition of automatic local variables                              */
/*----------------------------------------------------------------------------*/

static void WriteKernelVariables(char *src, int MshTyp,
                                 int NmbArg, ArgSct *ArgTab)
{
   int      i;
   char     str[100];
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
   }
}


/*----------------------------------------------------------------------------*/
/* Write the memory reading from the global structure to the local variables  */
/*----------------------------------------------------------------------------*/

static void WriteKernelMemoryReads( char *src, int MshTyp,
                                    int NmbArg, ArgSct *ArgTab)
{
   int      i, j, k, siz;
   char     str[100], ArgTd1[100], ArgTd2[100], LnkTd1[100], LnkTd2[100];
   char     LnkNam[100], CptNam[100], DegTst[100], DegNul[100];
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
               sprintf(DegNul, ": %sNul", arg->nam);
               sprintf(DegTst, "(%s <= %d) ?", CptNam, LnkArg->MaxDeg);
            }
            else
            {
               DegNul[0] = DegTst[0] = '\0';
            }

            sprintf( str, "   %s%s%s = %s %sTab[ %s%s%s ]%s %s;\n",
                     arg->nam, ArgTd2, ArgTd1,
                     DegTst, arg->nam, LnkNam, LnkTd1, LnkTd2, ArgTd1, DegNul);

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
   char     str[100];
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
   // Handle only power of 2 vector sizes
   switch(siz)
   {
      case 1 : {
         *cnt = 1;
         *vec = 1;
         *typ = GmlInt;
      }break;

      case 2 : {
         *cnt = 1;
         *vec = 2;
         *typ = GmlInt2;
      }break;

      case 4 : {
         *cnt = 1;
         *vec = 4;
         *typ = GmlInt4;
      }break;

      case 8 : {
         *cnt = 1;
         *vec = 8;
         *typ = GmlInt8;
      }break;

      case 16 : {
         *cnt = 1;
         *vec = 16;
         *typ = GmlInt16;
      }break;

      case 32 : {
         *cnt = 2;
         *vec = 16;
         *typ = GmlInt16;
      }break;

      case 64 : {
         *cnt = 4;
         *vec = 16;
         *typ = GmlInt16;
      }break;

      case 128 : {
         *cnt = 8;
         *vec = 16;
         *typ = GmlInt16;
      }break;
   }
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

double GmlLaunchKernel(int idx)
{
   double   RunTim;
   KrnSct   *hgh, *krn = &gml.krn[ idx ];

   if( (idx < 1) || (idx > gml.NmbKrn) || !krn->kernel )
      return(-1);

   // First send the parameters to the GPU memmory
   if(!UploadData(gml.ParIdx))
      return(-2);

   if(!krn->HghIdx)
      RunTim = RunOclKrn(krn);
   else
   {
      hgh = &gml.krn[ krn->HghIdx ];
      RunTim  = RunOclKrn(krn);
      RunTim += RunOclKrn(hgh);
   }

   // Finaly, get back the parameters from the GPU memmory
   if(!DownloadData(gml.ParIdx))
      return(-10);

   return(RunTim);
}


/*----------------------------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel                               */
/*----------------------------------------------------------------------------*/

static double RunOclKrn(KrnSct *krn)
{
   int      i, res;
   size_t   NmbGrp, GrpSiz, RetSiz = 0;
   DatSct   *dat;
   cl_event event;
   cl_ulong start, end;

   for(i=0;i<krn->NmbDat;i++)
   {
      dat = &gml.dat[ krn->DatTab[i] ];

      if((krn->DatTab[i] < 1) || (krn->DatTab[i] > GmlMaxDat) || !dat->GpuMem)
      {
         printf(  "Invalid user argument %d, DatTab[i]=%d, GpuMem=%p\n",
                  i, krn->DatTab[i], dat->GpuMem );
         return(0.);
      }

      res = clSetKernelArg(krn->kernel, i, sizeof(cl_mem), &dat->GpuMem);

      if(res != CL_SUCCESS)
      {
         printf("Adding user argument %d failed with error: %d\n", i, res);
         return(0.);
      }
   }

   res = clSetKernelArg(krn->kernel, krn->NmbDat, sizeof(cl_mem),
                        &gml.dat[ gml.ParIdx ].GpuMem);

   if(res != CL_SUCCESS)
   {
      printf("Adding the GMlib parameters argument failed with error %d\n", res);
      return(0.);
   }

   res = clSetKernelArg(krn->kernel, krn->NmbDat+1, 2 * sizeof(int), krn->NmbLin);

   if(res != CL_SUCCESS)
   {
      printf("Adding the kernel loop counter argument failed with error %d\n", res);
      return(0.);
   }

   // Fit data loop size to the GPU kernel size
   res = clGetKernelWorkGroupInfo(  krn->kernel, gml.device_id[ gml.CurDev ],
                                    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                    &GrpSiz, &RetSiz );

   if(res != CL_SUCCESS )
   {
      printf("Geting the kernel workgroup size failed with error %d\n", res);
      return(0.);
   }

   // Compute the hyperthreading level
   gml.CurGrpSiz = GrpSiz;
   NmbGrp = krn->NmbLin[0] / GrpSiz;
   NmbGrp *= GrpSiz;

   if(NmbGrp < krn->NmbLin[0])
      NmbGrp += GrpSiz;

   // Launch GPU code
   clFinish(gml.queue);

   if(clEnqueueNDRangeKernel( gml.queue, krn->kernel, 1, NULL,
                              &NmbGrp, &GrpSiz, 0, NULL, &event) )
   {
      return(-7);
   }

   clFinish(gml.queue);

   res = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                 sizeof(start), &start, NULL);
   if(res != CL_SUCCESS )
   {
      printf("Geting the start time event failed with error %d\n", res);
      return(0.);
   }

   res = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                 sizeof(end), &end, NULL);

   if(res != CL_SUCCESS )
   {
      printf("Geting the end time event failed with error %d\n", res);
      return(0.);
   }

   return((double)(end - start) / 1e9);
}


/*----------------------------------------------------------------------------*/
/* Read and compile an OpenCL source code                                     */
/*----------------------------------------------------------------------------*/

static int NewKernel(char *KernelSource, char *PrcNam)
{
   char     *buffer, *StrTab[1];
   int      err, res, idx = ++gml.NmbKrn;
   KrnSct   *krn = &gml.krn[ idx ];
   size_t   len, LenTab[10];

   if(idx > GmlMaxKrn)
      return(0);

   StrTab[0] = KernelSource;
   LenTab[0] = strlen(KernelSource) - 1;

   // Compile source code
   krn->program = clCreateProgramWithSource( gml.context, 1, (const char **)StrTab,
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
      clGetProgramBuildInfo(  krn->program, gml.device_id[ gml.CurDev ],
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

      if(!(buffer = malloc(len)))
         return(0);

      clGetProgramBuildInfo(  krn->program, gml.device_id[ gml.CurDev ],
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

   if(gml.DbgFlg)
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


/*----------------------------------------------------------------------------*/
/* Turning the printing of debugging information on or off                    */
/*----------------------------------------------------------------------------*/

void GmlDebugOn()
{
   gml.DbgFlg = 1;
}

void GmlDebugOff()
{
   gml.DbgFlg = 0;
}
