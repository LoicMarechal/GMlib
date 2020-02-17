

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.01                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: feb 14 2020                                           */
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

#define MIN(a,b)  ((a) < (b) ? (a) : (b))
#define MAX(a,b)  ((a) > (b) ? (a) : (b))
#define MB        1048576


/*----------------------------------------------------------------------------*/
/* Library internal data structures                                           */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int            BasTyp, BasIdx, LnkTyp, LnkIdx, LnkDir, CptIdx;
   int            ItmDeg, MaxDeg, NmbVec, VecSiz, VecTyp, MemMod;
   char           *nam;
}ArgSct;


typedef struct
{
   int            AloTyp, MemTyp, BasTyp, LnkTyp, DatTyp, RedVecIdx;
   int            NmbCol, ColSiz, VecSiz, NmbLin, LinSiz;
   char           *nam, use;
   size_t         siz;
   cl_mem         GpuMem;
   void           *CpuMem;
}DatSct;

typedef struct
{
   int            idx, HghIdx, SizTab[2], NmbDat, DatTab[ GmlMaxDat ];
   cl_kernel      kernel;
   cl_program     program; 
}KrnSct;

typedef struct
{
   int            NmbKrn, CurDev, RedKrnIdx[10], ParIdx;
   int            TypIdx[ GmlMaxTyp ], RefIdx[ GmlMaxTyp ], EleCnt[ GmlMaxTyp ];
   int            LnkMat[ GmlMaxTyp ][ GmlMaxTyp ];
   int            LnkHgh[ GmlMaxTyp ][ GmlMaxTyp ];
   int            CntMat[ GmlMaxTyp ][ GmlMaxTyp ];
   cl_uint        NmbDev;
   size_t         MemSiz, CurLocSiz, MovSiz;
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

char OclHexNmb[16] = {  '0','1','2','3','4','5','6','7',
                     '8','9','a','b','c','d','e','f' };

int  OclTypSiz[10] = {
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

char *OclTypStr[10]  = {
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

char *OclNulVec[10]  = {
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

int TypVecSiz[10] = {1,2,4,8,16,1,2,4,8,16};

int MshDatTyp[ GmlMaxTyp ] = {GmlFlt4, GmlInt2, GmlInt4, GmlInt4,
                              GmlInt4, GmlInt8, GmlInt8, GmlInt8};
int MshDatEleSiz[ GmlMaxTyp ] = {1, 2, 3, 4, 4, 5, 6, 8};
int MshDatVecSiz[ GmlMaxTyp ] = {4, 2, 4, 4, 4, 8, 8, 8};
int MshDatNmbCol[ GmlMaxTyp ] = {1, 1, 1, 1, 1, 1, 1, 1};
int MshTypDim[ GmlMaxTyp ]    = {0,1,2,2,3,3,3,3};
char *MshOclTypStr[ GmlMaxTyp ]  = { "Ver", "Edg", "Tri", "Qad",
                                  "Tet", "Pyr", "Pri", "Hex" };
char *MshEleStr[ GmlMaxTyp ]  = { "VerCrd", "EdgVer", "TriVer", "QadVer",
                                  "TetVer", "PyrVer", "PriVer", "HexVer" };
char *MshRefStr[ GmlMaxTyp ]  = { "VerRef", "EdgRef", "TriRef", "QadRef",
                                  "TetRef", "PyrRef", "PriRef", "HexRef" };

int SizMatBas[ GmlMaxTyp ][ GmlMaxTyp ] = {
   {0,16, 8, 4,32,16,16, 8},
   {2, 0, 2, 2, 8, 8, 8, 4},
   {4, 4, 0, 0, 2, 2, 2, 0},
   {4, 4, 0, 0, 0, 2, 2, 2},
   {4, 8, 4, 0, 0, 0, 0, 0},
   {8, 8, 4, 1, 0, 0, 0, 0},
   {8,16, 2, 4, 0, 0, 0, 0},
   {8,16, 0, 8, 0, 0, 0, 0} };

int SizMatHgh[ GmlMaxTyp ][ GmlMaxTyp ];

int SizMatMax[ GmlMaxTyp ][ GmlMaxTyp ] = {
   {0,64,16, 8,64,32,32,16},
   {2, 0, 2, 2,16,16,16, 8},
   {4, 4, 0, 0, 2, 2, 2, 0},
   {4, 4, 0, 0, 0, 2, 2, 2},
   {4, 8, 4, 0, 0, 0, 0, 0},
   {8, 8, 4, 1, 0, 0, 0, 0},
   {8,16, 2, 4, 0, 0, 0, 0},
   {8,16, 0, 8, 0, 0, 0, 0} };


/*----------------------------------------------------------------------------*/
/* Init device, context and queue                                             */
/*----------------------------------------------------------------------------*/

GmlParSct *GmlInit(int mod)
{
   int            err;
   cl_platform_id platforms[10];
   cl_uint        num_platforms;

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
   int idx;
   DatSct *dat;

   if(!(idx = GetNewDatIdx()))
      return(0);

   dat = &gml.dat[ idx ];

   dat->AloTyp = GmlArgDat;
   dat->BasTyp = 0;
   dat->LnkTyp = 0;
   dat->MemTyp = GmlInout;
   dat->DatTyp = 0;
   dat->NmbCol = 1;
   dat->ColSiz = siz;
   dat->VecSiz = 0;
   dat->NmbLin = 1;
   dat->LinSiz = dat->NmbCol * dat->ColSiz;
   dat->siz    = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(dat))
      return(0);

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Allocate one of the 8 mesh data types                                      */
/*----------------------------------------------------------------------------*/

int GmlNewMeshData(int MshTyp, int NmbLin)
{
   int      EleIdx, RefIdx;
   DatSct   *EleDat, *RefDat;

   if( (MshTyp < 0) || (MshTyp >= GmlMaxTyp) )
      return(0);

   if(!(EleIdx = GetNewDatIdx()))
      return(0);

   EleDat = &gml.dat[ EleIdx ];

   EleDat->AloTyp = GmlEleDat;
   EleDat->BasTyp = MshTyp;
   EleDat->LnkTyp = 0;
   EleDat->MemTyp = GmlInout;
   EleDat->DatTyp = MshDatTyp[ MshTyp ];
   EleDat->NmbCol = 1;
   EleDat->ColSiz = OclTypSiz[ EleDat->DatTyp ];
   EleDat->VecSiz = TypVecSiz[ EleDat->DatTyp ];
   EleDat->NmbLin = NmbLin;
   EleDat->LinSiz = EleDat->NmbCol * EleDat->ColSiz;
   EleDat->siz    = (size_t)EleDat->NmbLin * (size_t)EleDat->LinSiz;
   EleDat->GpuMem = EleDat->CpuMem = NULL;
   EleDat->nam    = MshEleStr[ MshTyp ];

   if(!NewData(EleDat))
      return(0);

   if(!(RefIdx = GetNewDatIdx()))
      return(0);

   RefDat = &gml.dat[ RefIdx ];

   RefDat->AloTyp = GmlRefDat;
   RefDat->BasTyp = MshTyp;
   RefDat->LnkTyp = 0;
   RefDat->MemTyp = GmlInout;
   RefDat->DatTyp = GmlInt;
   RefDat->NmbCol = 1;
   RefDat->ColSiz = OclTypSiz[ GmlInt ];
   RefDat->VecSiz = 0;
   RefDat->NmbLin = NmbLin;
   RefDat->LinSiz = RefDat->NmbCol * RefDat->ColSiz;
   RefDat->siz    = (size_t)RefDat->NmbLin * (size_t)RefDat->LinSiz;
   RefDat->GpuMem = RefDat->CpuMem = NULL;
   RefDat->nam    = MshRefStr[ MshTyp ];

   if(!NewData(RefDat))
      return(0);

   gml.EleCnt[ MshTyp ] = NmbLin;
   gml.TypIdx[ MshTyp ] = EleIdx;
   gml.LnkMat[ MshTyp ][ GmlVertices ] = EleIdx;
   gml.CntMat[ MshTyp ][ MshTyp ] = RefIdx;
   gml.RefIdx[ MshTyp ] = RefIdx;

   return(EleIdx);
}


/*----------------------------------------------------------------------------*/
/* Allocate a free solution/raw data associated with a mesh data type         */
/*----------------------------------------------------------------------------*/

int GmlNewSolutionData(int MshTyp, int NmbDat, int DatTyp, char *nam)
{
   int      idx;
   DatSct   *dat;

   if( (MshTyp < GmlVertices) || (MshTyp > GmlHexahedra) )
      return(0);

   if( (DatTyp < GmlInt) || (DatTyp > GmlFlt16) )
      return(0);

   if(!(idx = GetNewDatIdx()))
      return(0);

   dat = &gml.dat[ idx ];

   dat->AloTyp = GmlRawDat;
   dat->BasTyp = MshTyp;
   dat->LnkTyp = 0;
   dat->MemTyp = GmlInout;
   dat->DatTyp = DatTyp;
   dat->NmbCol = NmbDat;
   dat->ColSiz = OclTypSiz[ DatTyp ];
   dat->VecSiz = TypVecSiz[ DatTyp ];
   dat->NmbLin = gml.EleCnt[ MshTyp ];
   dat->LinSiz = dat->NmbCol * dat->ColSiz;
   dat->siz    = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(dat))
      return(0);

   return(idx);
}


/*----------------------------------------------------------------------------*/
/* Create an arbitrary link table between two kinds of mesh data types        */
/*----------------------------------------------------------------------------*/

int GmlNewLinkData(int BasTyp, int LnkTyp, int NmbDat, char *nam)
{
   int      LnkIdx, VecCnt, VecSiz, VecTyp;
   DatSct   *dat;

   if( (BasTyp < GmlVertices) || (BasTyp > GmlHexahedra) )
      return(0);

   if( (LnkTyp < GmlVertices) || (LnkTyp > GmlHexahedra) )
      return(0);

   if(!(LnkIdx = GetNewDatIdx()))
      return(0);

   GetCntVec(NmbDat, &VecCnt, &VecSiz, &VecTyp);

   dat = &gml.dat[ LnkIdx ];

   dat->AloTyp = GmlLnkDat;
   dat->BasTyp = BasTyp;
   dat->LnkTyp = LnkTyp;
   dat->MemTyp = GmlInout;
   dat->DatTyp = VecTyp;
   dat->NmbCol = VecCnt;
   dat->ColSiz = VecCnt * OclTypSiz[ VecTyp ];
   dat->VecSiz = 1;
   dat->NmbLin = gml.EleCnt[ BasTyp ];
   dat->LinSiz = dat->NmbCol * dat->ColSiz;
   dat->siz    = (size_t)dat->NmbLin * (size_t)dat->LinSiz;
   dat->GpuMem = dat->CpuMem = NULL;
   dat->nam    = nam;

   if(!NewData(dat))
      return(0);

   gml.LnkMat[ BasTyp ][ LnkTyp ] = LnkIdx;

   return(LnkIdx);
}


/*----------------------------------------------------------------------------*/
/* Create an arbitrary link table between two kinds of mesh data types        */
/*----------------------------------------------------------------------------*/

int GmlNewBallData(int BasTyp, int LnkTyp, char *BalNam, char *DegNam)
{
   int         i, j, BalIdx, HghIdx, DegIdx, VerIdx, SrcIdx, DstIdx;
   int         EleSiz, VecSiz, BalSiz, MaxSiz, HghSiz = 0;
   int         *EleTab, *BalTab, *DegTab, *HghTab;
   int         MaxDeg = 0, MaxPos = 0, VecCnt, VecTyp, NmbDat;
   int64_t     DegTot = 0;
   DatSct      *src, *dst, *bal, *hgh, *deg, *BalDat, *HghDat, *DegDat;


   // Check and prepare some data
   if( (BasTyp < 0) || (BasTyp >= GmlMaxTyp) )
      return(0);

   if( (LnkTyp < 0) || (LnkTyp >= GmlMaxTyp) )
      return(0);

   if(!BalNam || !DegNam)
      return(0);

   SrcIdx = gml.TypIdx[ BasTyp ];
   DstIdx = gml.TypIdx[ LnkTyp ];
   src = &gml.dat[ SrcIdx ];
   dst = &gml.dat[ DstIdx ];


   // Build the degrees table: count and allocate gml data,
   // base and high vector sizes for the next two phases
   if(!(DegIdx = GetNewDatIdx()))
      return(0);

   DegDat = &gml.dat[ DegIdx ];

   DegDat->AloTyp = GmlLnkDat;
   DegDat->BasTyp = BasTyp;
   DegDat->LnkTyp = LnkTyp;
   DegDat->MemTyp = GmlInout;
   DegDat->DatTyp = GmlInt;
   DegDat->NmbCol = 1;
   DegDat->ColSiz = OclTypSiz[ GmlInt ];
   DegDat->VecSiz = 0;
   DegDat->NmbLin = gml.EleCnt[ BasTyp ];
   DegDat->LinSiz = DegDat->NmbCol * DegDat->ColSiz;
   DegDat->siz    = (size_t)DegDat->NmbLin * (size_t)DegDat->LinSiz;
   DegDat->GpuMem = DegDat->CpuMem = NULL;
   DegDat->nam    = DegNam;

   if(!NewData(DegDat))
      return(0);

   gml.CntMat[ BasTyp ][ LnkTyp ] = DegIdx;
   deg = &gml.dat[ gml.CntMat[ src->BasTyp ][ dst->BasTyp ] ];

   DegTab = deg->CpuMem;
   EleTab = dst->CpuMem;
   BalSiz = SizMatBas[ src->BasTyp ][ dst->BasTyp ];
   MaxSiz = SizMatMax[ src->BasTyp ][ dst->BasTyp ];
   EleSiz = MshDatEleSiz[ dst->BasTyp ];

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

   printf(  "base table up to ver %d, occupency = %g\n", MaxPos,
            (float)(100 * DegTot) / (float)(MaxPos * BalSiz) );

   MaxDeg = pow(2., ceil(log2(MaxDeg)));
   HghSiz = MIN(MaxDeg, SizMatMax[ src->BasTyp ][ dst->BasTyp ]);
   SizMatHgh[ src->BasTyp ][ dst->BasTyp ] = HghSiz;
   printf("%d over connected vertices\n", src->NmbLin - MaxPos);


   // Allocate the base vector ball table
   if(!(BalIdx = GetNewDatIdx()))
      return(0);

   NmbDat = SizMatBas[ BasTyp ][ LnkTyp ];
   GetCntVec(NmbDat, &VecCnt, &VecSiz, &VecTyp);

   BalDat = &gml.dat[ BalIdx ];

   BalDat->AloTyp = GmlLnkDat;
   BalDat->BasTyp = BasTyp;
   BalDat->LnkTyp = LnkTyp;
   BalDat->MemTyp = GmlInout;
   BalDat->DatTyp = VecTyp;
   BalDat->NmbCol = VecCnt;
   BalDat->ColSiz = VecCnt * OclTypSiz[ VecTyp ];
   BalDat->VecSiz = 1;
   BalDat->NmbLin = MaxPos+1;
   BalDat->LinSiz = BalDat->NmbCol * BalDat->ColSiz;
   BalDat->siz    = (size_t)BalDat->NmbLin * (size_t)BalDat->LinSiz;
   BalDat->GpuMem = BalDat->CpuMem = NULL;
   BalDat->nam    = BalNam;

   if(!NewData(BalDat))
      return(0);

   gml.LnkMat[ BasTyp ][ LnkTyp ] = BalIdx;
   bal = &gml.dat[ gml.LnkMat[ src->BasTyp ][ dst->BasTyp ] ];


   // Allocate the high vector ball table
   if(!(HghIdx = GetNewDatIdx()))
      return(0);

   NmbDat = SizMatHgh[ BasTyp ][ LnkTyp ];
   GetCntVec(NmbDat, &VecCnt, &VecSiz, &VecTyp);

   HghDat = &gml.dat[ HghIdx ];

   HghDat->AloTyp = GmlLnkDat;
   HghDat->BasTyp = BasTyp;
   HghDat->LnkTyp = LnkTyp;
   HghDat->MemTyp = GmlInout;
   HghDat->DatTyp = VecTyp;
   HghDat->NmbCol = VecCnt;
   HghDat->ColSiz = VecCnt * OclTypSiz[ VecTyp ];
   HghDat->VecSiz = 1;
   HghDat->NmbLin = gml.EleCnt[ BasTyp ] - MaxPos;
   HghDat->LinSiz = HghDat->NmbCol * HghDat->ColSiz;
   HghDat->siz    = (size_t)HghDat->NmbLin * (size_t)HghDat->LinSiz;
   HghDat->GpuMem = HghDat->CpuMem = NULL;
   HghDat->nam    = BalNam;

   if(!NewData(HghDat))
      return(0);

   gml.LnkHgh[ BasTyp ][ LnkTyp ] = HghIdx;
   hgh = &gml.dat[ gml.LnkHgh[ src->BasTyp ][ dst->BasTyp ] ];


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
   printf("transfered ball data %d %d %d to the GPU\n", BalIdx, HghIdx, DegIdx);

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

      for(i=0;i<dat->NmbCol;i++)
         tab[ lin * dat->NmbCol + i ] = va_arg(VarArg, int);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->BasTyp == GmlVertices) )
   {
      CrdTab = (float *)dat->CpuMem;
      RefDat = &gml.dat[ gml.CntMat[ dat->BasTyp ][ dat->BasTyp ] ];
      RefTab = (int *)RefDat->CpuMem;
      siz = 4;

      for(i=0;i<3;i++)
         CrdTab[ lin * siz + i ] = va_arg(VarArg, double);

      CrdTab[ lin * siz + 3 ] = 0.;
      RefTab[ lin ] = va_arg(VarArg, int);
   }
   else if( (dat->AloTyp == GmlEleDat) && (dat->BasTyp > GmlVertices) )
   {
      EleTab = (int *)dat->CpuMem;
      siz = TypVecSiz[ MshDatTyp[ dat->BasTyp ] ];
      RefDat = &gml.dat[ gml.CntMat[ dat->BasTyp ][ dat->BasTyp ] ];
      RefTab = (int *)RefDat->CpuMem;

      for(i=0;i<MshDatEleSiz[ dat->BasTyp ];i++)
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
   else if( (dat->AloTyp == GmlEleDat) && (dat->BasTyp == GmlVertices) )
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
   DatSct *dat = &gml.dat[ idx ];

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

static int DownloadData(int idx)
{
   DatSct *dat = &gml.dat[ idx ];

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
   int      i, j, flg, KrnIdx, KrnHghIdx, SrcTyp, DstTyp, NmbArg = 0, RefIdx;
   int      FlgTab[ GmlMaxDat ], IdxTab[ GmlMaxDat ];
   int      LnkTab[ GmlMaxDat ], CntTab[ GmlMaxDat ];
   int      NmbItm, NmbVec, VecTyp, VecSiz, LnkPos, CptPos, ArgHghPos;
   int      NmbHgh, HghVec, HghSiz, HghTyp, HghArg = -1, HghIdx = -1;
   char     src[10000] = "\0", BalNam[100], DegNam[100];
   va_list  VarArg;
   DatSct   *dat, *RefDat;
   ArgSct   ArgTab[10];
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

      if( LnkTab[i] || (dat->BasTyp == BasTyp) )
         continue;

      SrcTyp = BasTyp;
      DstTyp = dat->BasTyp;

      if(MshTypDim[ SrcTyp ] > MshTypDim[ DstTyp ])
      {
         LnkTab[i] = gml.LnkMat[ BasTyp ][ DstTyp ];
         CntTab[i] = 0;
      }
      else if(MshTypDim[ SrcTyp ] == MshTypDim[ DstTyp ])
      {
         // build neighbours
      }
      else
      {
         if(!gml.LnkMat[ BasTyp ][ dat->BasTyp ])
         {
            // Generate the default link between the two kinds of entities
            sprintf(BalNam, "%s%sBal", MshOclTypStr[ BasTyp ], MshOclTypStr[ DstTyp ]);
            sprintf(DegNam, "%s%sDeg", MshOclTypStr[ BasTyp ], MshOclTypStr[ DstTyp ]);
            GmlNewBallData(BasTyp, DstTyp, BalNam, DegNam);
         }

         LnkTab[i] = gml.LnkMat[ BasTyp ][ DstTyp ];
         CntTab[i] = gml.CntMat[ BasTyp ][ DstTyp ];
      }
   }

   for(i=0;i<NmbTyp;i++)
   {
      if(!LnkTab[i])
         continue;

      dat = &gml.dat[ IdxTab[i] ];
      flg = 0;

      for(j=0;j<NmbArg;j++)
         if(ArgTab[j].BasIdx == LnkTab[i])
            flg = 1;

      if(flg)
         continue;

      DstTyp = dat->BasTyp;
      NmbItm = SizMatBas[ BasTyp ][ DstTyp ];
      GetCntVec(NmbItm, &NmbVec, &VecSiz, &VecTyp);

      if(MshTypDim[ SrcTyp ] > MshTypDim[ DstTyp ])
      {
         ArgTab[ NmbArg ].BasTyp = DstTyp;
         ArgTab[ NmbArg ].BasIdx = LnkTab[i];
         ArgTab[ NmbArg ].LnkDir = -1;
         ArgTab[ NmbArg ].LnkTyp = -1;
         ArgTab[ NmbArg ].LnkIdx = -1;
         ArgTab[ NmbArg ].CptIdx = -1;
         ArgTab[ NmbArg ].ItmDeg = NmbItm;
         ArgTab[ NmbArg ].MaxDeg = NmbItm;
         ArgTab[ NmbArg ].NmbVec = NmbVec;
         ArgTab[ NmbArg ].VecSiz = VecSiz;
         ArgTab[ NmbArg ].VecTyp = VecTyp;
         ArgTab[ NmbArg ].MemMod = GmlReadMode;
         ArgTab[ NmbArg ].nam = gml.dat[ ArgTab[ NmbArg ].BasIdx ].nam;
         NmbArg++;
      }
      else if(MshTypDim[ SrcTyp ] == MshTypDim[ DstTyp ])
      {
      }
      else if(MshTypDim[ SrcTyp ] < MshTypDim[ DstTyp ])
      {
         ArgTab[ NmbArg ].BasTyp = DstTyp;
         ArgTab[ NmbArg ].BasIdx = LnkTab[i];
         ArgTab[ NmbArg ].LnkDir = 1;
         ArgTab[ NmbArg ].LnkTyp = -1;
         ArgTab[ NmbArg ].LnkIdx = -1;
         ArgTab[ NmbArg ].CptIdx = -1;
         ArgTab[ NmbArg ].ItmDeg = -1;
         ArgTab[ NmbArg ].MaxDeg = NmbItm;
         ArgTab[ NmbArg ].NmbVec = NmbVec;
         ArgTab[ NmbArg ].VecSiz = VecSiz;
         ArgTab[ NmbArg ].VecTyp = VecTyp;
         ArgTab[ NmbArg ].MemMod = GmlReadMode;
         ArgTab[ NmbArg ].nam = gml.dat[ ArgTab[ NmbArg ].BasIdx ].nam;
         HghArg = NmbArg;
         HghIdx = gml.LnkHgh[ BasTyp ][ DstTyp ];
         NmbArg++;

         ArgTab[ NmbArg ].BasTyp = DstTyp;
         ArgTab[ NmbArg ].BasIdx = gml.CntMat[ BasTyp ][ DstTyp ];
         ArgTab[ NmbArg ].LnkDir = 0;
         ArgTab[ NmbArg ].LnkTyp = -1;
         ArgTab[ NmbArg ].LnkIdx = -1;
         ArgTab[ NmbArg ].CptIdx = -1;
         ArgTab[ NmbArg ].ItmDeg = 1;
         ArgTab[ NmbArg ].MaxDeg = 1;
         ArgTab[ NmbArg ].NmbVec = 1;
         ArgTab[ NmbArg ].VecSiz = 1;
         ArgTab[ NmbArg ].VecTyp = GmlInt;
         ArgTab[ NmbArg ].MemMod = GmlReadMode;
         ArgTab[ NmbArg ].nam = gml.dat[ ArgTab[ NmbArg ].BasIdx ].nam;
         NmbArg++;
      }
   }

   for(i=0;i<NmbTyp;i++)
   {
      dat = &gml.dat[ IdxTab[i] ];
      DstTyp = dat->BasTyp;
      LnkPos = CptPos = -1;

      if(LnkTab[i])
         for(j=0;j<NmbArg;j++)
            if(ArgTab[j].BasIdx == LnkTab[i])
               LnkPos = j;

      if(CntTab[i])
         for(j=0;j<NmbArg;j++)
            if(ArgTab[j].BasIdx == CntTab[i])
               CptPos = j;

      ArgTab[ NmbArg ].BasTyp = DstTyp;
      ArgTab[ NmbArg ].BasIdx = IdxTab[i];
      ArgTab[ NmbArg ].LnkTyp = DstTyp;
      ArgTab[ NmbArg ].LnkIdx = LnkPos;
      ArgTab[ NmbArg ].CptIdx = CptPos;

      if(LnkPos != -1 && CptPos != -1)
      {
         ArgTab[ NmbArg ].ItmDeg = -1;
         ArgTab[ NmbArg ].MaxDeg = ArgTab[ LnkPos ].MaxDeg;
         ArgHghPos = NmbArg;
      }
      else if(LnkPos != -1 && CptPos == -1)
      {
         ArgTab[ NmbArg ].ItmDeg = ArgTab[ LnkPos ].ItmDeg;
         ArgTab[ NmbArg ].MaxDeg = ArgTab[ LnkPos ].MaxDeg;
      }
      else
      {
         ArgTab[ NmbArg ].ItmDeg = -1;
         ArgTab[ NmbArg ].MaxDeg = -1;
      }

      ArgTab[ NmbArg ].NmbVec = dat->NmbCol;
      ArgTab[ NmbArg ].VecSiz = dat->VecSiz;
      ArgTab[ NmbArg ].VecTyp = dat->DatTyp;
      ArgTab[ NmbArg ].MemMod = FlgTab[i];
      ArgTab[ NmbArg ].nam = dat->nam;
      NmbArg++;

      if(!(FlgTab[i] & GmlRefFlag) || (CptPos != -1))
         continue;

      puts("Add refs");
      RefIdx = gml.RefIdx[ DstTyp ];
      RefDat = &gml.dat[ RefIdx ];
      ArgTab[ NmbArg ].BasTyp = DstTyp;
      ArgTab[ NmbArg ].BasIdx = RefIdx;
      ArgTab[ NmbArg ].LnkTyp = DstTyp;
      ArgTab[ NmbArg ].LnkIdx = LnkPos;
      ArgTab[ NmbArg ].CptIdx = CptPos;
      ArgTab[ NmbArg ].ItmDeg = -1;
      ArgTab[ NmbArg ].MaxDeg = ArgTab[ LnkPos ].MaxDeg;
      ArgTab[ NmbArg ].NmbVec = RefDat->NmbCol;
      ArgTab[ NmbArg ].VecSiz = RefDat->VecSiz;
      ArgTab[ NmbArg ].VecTyp = RefDat->DatTyp;
      ArgTab[ NmbArg ].MemMod = FlgTab[i];
      ArgTab[ NmbArg ].nam = RefDat->nam;
      NmbArg++;
   }

   // Generate the kernel source code
   WriteUserTypedef        (src, DefSrc);
   WriteProcedureHeader    (src, PrcNam, BasTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, BasTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, BasTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, BasTyp, NmbArg, ArgTab);

   // And Compile it
   KrnIdx = NewKernel      (src, PrcNam);
   puts(src);

   krn = &gml.krn[ KrnIdx ];
   krn->NmbDat = NmbArg;
   krn->SizTab[1] = 0;

   if(HghIdx == -1)
      krn->SizTab[0] = gml.dat[ gml.TypIdx[ BasTyp ] ].NmbLin;
   else
      krn->SizTab[0] = gml.dat[ gml.TypIdx[ BasTyp ] ].NmbLin - gml.dat[ HghIdx ].NmbLin;

   for(i=0;i<NmbArg;i++)
      krn->DatTab[i] = ArgTab[i].BasIdx;

   if(HghArg == -1)
      return(KrnIdx);

   // In case of uplink kernel, generate a second high degree kernel
   NmbHgh = SizMatHgh[ BasTyp ][ DstTyp ];
   GetCntVec(NmbHgh, &HghVec, &HghSiz, &HghTyp);

   ArgTab[ HghArg ].BasIdx = HghIdx;
   ArgTab[ HghArg ].MaxDeg = NmbHgh;
   ArgTab[ HghArg ].NmbVec = HghVec;
   ArgTab[ HghArg ].VecSiz = HghSiz;
   ArgTab[ HghArg ].VecTyp = HghTyp;
   src[0] = '\0';

   // Generate the kernel source code
   WriteUserTypedef        (src, DefSrc);
   WriteProcedureHeader    (src, PrcNam, BasTyp, NmbArg, ArgTab);
   WriteKernelVariables    (src, BasTyp, NmbArg, ArgTab);
   WriteKernelMemoryReads  (src, BasTyp, NmbArg, ArgTab);
   WriteUserKernel         (src, KrnSrc);
   WriteKernelMemoryWrites (src, BasTyp, NmbArg, ArgTab);

   // And Compile it
   KrnHghIdx = NewKernel   (src, PrcNam);
   gml.krn[ KrnIdx ].HghIdx = KrnHghIdx;
   puts(src);

   krn = &gml.krn[ KrnHghIdx ];
   krn->NmbDat = NmbArg;
   krn->SizTab[0] = gml.dat[ HghIdx ].NmbLin;
   krn->SizTab[1] = gml.dat[ gml.TypIdx[ BasTyp ] ].NmbLin - gml.dat[ HghIdx ].NmbLin;

   for(i=0;i<NmbArg;i++)
      krn->DatTab[i] = ArgTab[i].BasIdx;

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

static void WriteProcedureHeader(char *src, char *PrcNam, int BasTyp,
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

      sprintf(str,  "\n   __global %s ", OclTypStr[ arg->VecTyp ]);
      strcat(src, str);

      if(arg->NmbVec <= 1)
         sprintf(str, "*%sTab,", arg->nam);
      else
         sprintf(str, "(*%sTab)[%d],", arg->nam, arg->NmbVec);

      strcat(src, str);
   }

   strcat(src, "\n   __global GmlParSct *par,");
   strcat(src, "\n   const    int2       count )\n{\n");
}


/*----------------------------------------------------------------------------*/
/* Write definition of automatic local variables                              */
/*----------------------------------------------------------------------------*/

static void WriteKernelVariables(char *src, int BasTyp,
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
      CptArg = (arg->CptIdx != -1) ? &ArgTab[ arg->CptIdx ] : NULL;

      if(LnkArg && LnkArg->MaxDeg > 1)
      {
         if(arg->NmbVec > 1)
            sprintf( str,  "   %s %s[%d][%d];\n", OclTypStr[ arg->VecTyp ],
                     arg->nam, LnkArg->MaxDeg, arg->NmbVec );
         else
            sprintf( str,  "   %s %s[%d];\n", OclTypStr[ arg->VecTyp ],
                     arg->nam, LnkArg->MaxDeg );
      }
      else
      {
         if(arg->NmbVec > 1)
            sprintf( str,  "   %s %s[%d];\n", OclTypStr[ arg->VecTyp ],
                     arg->nam, arg->NmbVec );
         else
            sprintf(str,  "   %s %s;\n", OclTypStr[ arg->VecTyp ], arg->nam);
      }

      strcat(src, str);

      if(CptArg)
      {
         sprintf(str,  "   %s %sDeg;\n", OclTypStr[ CptArg->VecTyp ], arg->nam);
         strcat(src, str);
         sprintf(str,  "   %s %sNul;\n", OclTypStr[ arg->VecTyp ], arg->nam);
         strcat(src, str);
         sprintf(str,  "   #define   %sDegMax %d\n", arg->nam, LnkArg->MaxDeg);
         strcat(src, str);
      }
   }
}


/*----------------------------------------------------------------------------*/
/* Write the memory reading from the global structure to the local variables  */
/*----------------------------------------------------------------------------*/

static void WriteKernelMemoryReads( char *src, int BasTyp,
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

      if(!(arg->MemMod & GmlReadMode))
         continue;

      LnkArg = (arg->LnkIdx != -1) ? &ArgTab[ arg->LnkIdx ] : NULL;
      CptArg = (arg->CptIdx != -1) ? &ArgTab[ arg->CptIdx ] : NULL;

      for(j=0;j<arg->NmbVec;j++)
      {
         if(CptArg)
         {
            sprintf(CptNam,  "%sDeg", arg->nam);
            sprintf(str,  "   %s = %s;\n", CptNam, CptArg->nam);
            strcat(src, str);
            sprintf(str,  "   %sNul = %s;\n", arg->nam, OclNulVec[ arg->VecTyp ]);
            strcat(src, str);
            strcat(src, "\n");
         }

         if(arg->NmbVec > 1)
            sprintf(ArgTd1, "[%d]", j);
         else
            ArgTd1[0] = '\0';

         if(LnkArg && LnkArg->NmbVec * LnkArg->VecSiz > 1)
            siz = LnkArg->NmbVec * LnkArg->VecSiz;
         else
            siz = 1;

         for(k=0;k<siz;k++)
         {
            if(siz > 1)
               sprintf(ArgTd2, "[%d]", k);
            else
               ArgTd2[0] = '\0';

            if(LnkArg && LnkArg->NmbVec > 1)
               sprintf(LnkTd1, "[%d]", k/16);
            else
               LnkTd1[0] = '\0';

            if(LnkArg && LnkArg->VecSiz > 1)
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

static void WriteKernelMemoryWrites(char *src, int BasTyp,
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

      if(!(arg->MemMod & GmlWriteMode))
         continue;

      if(arg->NmbVec == 1)
      {
         sprintf( str, "   %sTab[ cnt + count.s1 ] = %s;\n",
                  arg->nam, arg->nam );
         strcat(src, str);
      }
      else
      {
         for(c=0;c<arg->NmbVec;c++)
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
   int      i;
   size_t   GloSiz, LocSiz, RetSiz = 0;
   DatSct   *dat;
   cl_event event;
   cl_ulong start, end;

   //printf("run kernel %d, nmblin=%d, shift=%d\n", krn->idx, krn->SizTab[0], krn->SizTab[1]);
   for(i=0;i<krn->NmbDat;i++)
   {
      dat = &gml.dat[ krn->DatTab[i] ];
      /*printf(  "DatTab[%d]=%d, GpuMem=%p, NmbLin=%d, NmbCol=%d, ColSiz=%d, VecSiz=%d, LinSiz=%d\n",
               i,krn->DatTab[i],dat->GpuMem, dat->NmbLin, dat->NmbCol, dat->ColSiz, dat->VecSiz, dat->LinSiz);*/

      if( (krn->DatTab[i] < 1) || (krn->DatTab[i] > GmlMaxDat) || !dat->GpuMem
      || (clSetKernelArg(krn->kernel, i, sizeof(cl_mem), &dat->GpuMem) != CL_SUCCESS) )
      {
         printf("i=%d, DatTab[i]=%d, GpuMem=%p\n",i,krn->DatTab[i],dat->GpuMem);
         return(-3);
      }
   }

   if(clSetKernelArg(krn->kernel, krn->NmbDat, sizeof(cl_mem),
                     &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
   {
      return(-4);
   }

   if(clSetKernelArg(krn->kernel, krn->NmbDat+1,
                     2 * sizeof(int), krn->SizTab) != CL_SUCCESS)
   {
      return(-5);
   }

   // Fit data loop size to the GPU kernel size
   if(clGetKernelWorkGroupInfo(  krn->kernel, gml.device_id[ gml.CurDev ],
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                 &LocSiz, &RetSiz) != CL_SUCCESS )
   {
      return(-6);
   }

   gml.CurLocSiz = LocSiz;
   GloSiz = krn->SizTab[0] / LocSiz;
   GloSiz *= LocSiz;

   if(GloSiz < krn->SizTab[0])
      GloSiz += LocSiz;

   // Launch GPU code
   clFinish(gml.queue);
   //printf("LocSiz=%zu, GloSiz=%zu\n",LocSiz,GloSiz);

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

   return((double)(end - start) / 1e9);
}


/*----------------------------------------------------------------------------*/
/* Read and compile an OpenCL source code                                     */
/*----------------------------------------------------------------------------*/

static int NewKernel(char *KernelSource, char *PrcNam)
{
   char     *buffer, *StrTab[1];
   int      err, idx = ++gml.NmbKrn;
   KrnSct   *krn = &gml.krn[ idx ];
   size_t   len, LenTab[1];

   if(idx > GmlMaxKrn)
      return(0);

   StrTab[0] = KernelSource;
   LenTab[0] = strlen(KernelSource) - 1;

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

   krn->idx = idx;

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
