

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                         GPU Meshing Library 3.01                           */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCL                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: nov 18 2019                                           */
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

enum  memory_types {GmlInternal, GmlInput, GmlOutput, GmlInout};
enum  reduction_opperations {GmlMin, GmlSum, GmlMax};
enum  meshing_type {
      GmlParameters, GmlRawData, GmlLnkData,
      GmlVertices, GmlEdges, GmlTriangles, GmlQuadrilaterals, GmlTetrahedra,
      GmlPyramids, GmlPrisms, GmlHexahedra, GmlMaxTyp };
enum  opencl_type {GmlInt, GmlInt2, GmlInt4, GmlInt8, GmlInt16,
                   GmlFlt, GmlFlt2, GmlFlt4, GmlFlt8, GmlFlt16};


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

GmlParSct  *GmlInit              (int);
void        GmlStop              ();
void        GmlListGPU           ();
int         GmlNewParameters     (int, char *);
int         GmlNewMeshData       (int, int);
int         GmlNewSolutionData   (int, int, int, char *);
int         GmlNewLinkData       (int, int, int, char *);
int         GmlFreeData          (int);
int         GmlSetDataLine       (int, int, ...);
int         GmlGetDataLine       (int, int, ...);
int         GmlCompileKernel     (char *, char *, int, int, ...);
double      GmlLaunchKernel      (int, int);
double      GmlReduceVector      (int, int, double *);
size_t      GmlGetMemoryUsage    ();
size_t      GmlGetMemoryTransfer ();
GmlParSct  *GmlGetParameters     ();
