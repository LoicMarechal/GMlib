

/*----------------------------------------------------------------------------*/
/*                                                                            */
/*                      GPU Meshing Library 3.00                              */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*   Description:       Easy mesh programing with OpenCl                      */
/*   Author:            Loic MARECHAL                                         */
/*   Creation date:     jul 02 2010                                           */
/*   Last modification: jun 14 2018                                           */
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

enum memory_types {GmlInternal, GmlInput, GmlOutput, GmlInout};
enum reduction_opperations {GmlMin, GmlSum, GmlMax};
enum meshing_type {GmlRawData, GmlVertices, GmlEdges, GmlTriangles, \
      GmlQuadrilaterals, GmlTetrahedra, GmlHexahedra};


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
int        GmlNewData            (int, int, int, int);
int        GmlFreeData           (int);
int        GmlSetRawData         (int, int, void *);
int        GmlGetRawData         (int, int, void *);
int        GmlSetVertex          (int, int, float, float, float);
int        GmlGetVertex          (int, int, float *, float *, float *);
int        GmlSetEdge            (int, int, int, int);
int        GmlSetTriangle        (int, int, int, int, int);
int        GmlSetQuadrilateral   (int, int, int, int, int, int);
int        GmlSetTetrahedron     (int, int, int, int, int, int);
int        GmlSetHexahedron      (int, int, int, int, int, int, int, int, int, int);
int        GmlUploadData         (int);
int        GmlDownloadData       (int);
int        GmlNewBall            (int, int);
int        GmlFreeBall           (int);
int        GmlUploadBall         (int);
int        GmlNewKernel          (char *, char *);
double     GmlLaunchKernel       (int, int ,int, ...);
double     GmlLaunchBallKernel   (int, int, int ,int, ...);
double     GmlReduceVector       (int, int, double *);
size_t     GmlGetMemoryUsage     ();
size_t     GmlGetMemoryTransfer  ();
GmlParSct *GmlGetParameters      ();
