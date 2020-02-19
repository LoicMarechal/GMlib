

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.10                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: feb 19 2020                                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Includes                                                                   */
/*----------------------------------------------------------------------------*/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


/*----------------------------------------------------------------------------*/
/* Set max user data                                                          */
/*----------------------------------------------------------------------------*/

#define GmlMaxDat    100
#define GmlMaxBal    100
#define GmlMaxKrn    100
#define GmlMaxSrcSiz 10000
#define MaxGpu       10
#define GmlCpu       MaxGpu + 1
#define GmlGpu       MaxGpu + 2
#define GmlRefFlag   1
#define GmlReadMode  2
#define GmlWriteMode 4
#ifndef MAX_WORKGROUP_SIZE
#define MAX_WORKGROUP_SIZE 1024
#endif

enum  memory_type    {GmlInternal, GmlInput, GmlOutput, GmlInout};
enum  element_type   {GmlVertices, GmlEdges, GmlTriangles, GmlQuadrilaterals,
                      GmlTetrahedra, GmlPyramids, GmlPrisms, GmlHexahedra,
                      GmlMaxEleTyp};
enum  opencl_type    {GmlInt, GmlInt2, GmlInt4, GmlInt8, GmlInt16,
                      GmlFlt, GmlFlt2, GmlFlt4, GmlFlt8, GmlFlt16,
                      GmlMaxOclTyp};


/*----------------------------------------------------------------------------*/
/* GML public parameters structure:                                           */
/* feel free to add any fields to your convenience                            */
/*----------------------------------------------------------------------------*/

typedef struct
{
   int   wei;
   float MinQal, rlx;
}GmlParSct;


/*----------------------------------------------------------------------------*/
/* User available procedures                                                  */
/*----------------------------------------------------------------------------*/

size_t      GmlInit              (int);
void        GmlStop              (size_t);
void        GmlListGPU           ();
int         GmlNewParameters     (size_t, int, char *);
int         GmlNewMeshData       (size_t, int, int);
int         GmlNewSolutionData   (size_t, int, int, int, char *);
int         GmlNewLinkData       (size_t, int, int, int, char *);
int         GmlFreeData          (size_t, int);
int         GmlSetDataLine       (size_t, int, int, ...);
int         GmlGetDataLine       (size_t, int, int, ...);
int         GmlCompileKernel     (size_t, char *, char *, char *, int, int, ...);
double      GmlLaunchKernel      (size_t, int);
double      GmlReduceVector      (size_t, int, int, double *);
size_t      GmlGetMemoryUsage    (size_t);
size_t      GmlGetMemoryTransfer (size_t);
GmlParSct  *GmlGetParameters     (size_t);
void        GmlDebugOn           (size_t);
void        GmlDebugOff          (size_t);
