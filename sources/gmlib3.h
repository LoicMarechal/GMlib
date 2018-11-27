

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                      GPU Meshing Library 3.00                              */
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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


/*----------------------------------------------------------------------------*/
/* Set max user data                                                          */
/*----------------------------------------------------------------------------*/

#define GmlMaxDat 100
#define GmlMaxBal 100
#define GmlMaxKrn 100
#define GmlMaxSrcSiz 10000
#define MaxGpu 10
#define GmlCpu MaxGpu + 1
#define GmlGpu MaxGpu + 2
#ifndef MAX_WORKGROUP_SIZE
#define MAX_WORKGROUP_SIZE 1024
#endif

enum access_mode {GmlInternal, GmlRead, GmlWrite, GmlReadWrite};
enum reduction_opperations {GmlMin, GmlSum, GmlMax};
enum meshing_type {
   GmlRawData, GmlVertices, GmlEdges, GmlTriangles,
   GmlQuadrilaterals, GmlTetrahedra, GmlHexahedra, GmlEnd };
enum opencl_type {
   GmlInt,   GmlInt2,   GmlInt4,   GmlInt8,   GmlInt16, GmlInt32,
   GmlFloat, GmlFloat2, GmlFloat4, GmlFloat8, GmlFloat16 };


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

GmlParSct *GmlInit               (int);
void       GmlStop               ();
void       GmlListGPU            ();
int        GmlNewData            (int, char *, int, ...);
int        GmlFreeData           (int);
int        GmlSetDataLine        (int, int, void *);
int        GmlGetDataLine        (int, int, void *);
int        GmlSetDataBlock       (int, void *, void *);
int        GmlGetDataBlock       (int, void *, void *);
int        GmlNewBall            (int, int);
int        GmlFreeBall           (int);
int        GmlUploadBall         (int);
int        GmlNewKernel          (char *);
double     GmlLaunchKernel       (int, int ,...);
double     GmlLaunchBallKernel   (int, int, int ,int, ...);
double     GmlReduceVector       (int, int, double *);
size_t     GmlGetMemoryUsage     ();
size_t     GmlGetMemoryTransfer  ();
GmlParSct *GmlGetParameters      ();
