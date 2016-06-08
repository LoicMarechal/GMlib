

/*----------------------------------------------------------*/
/*															*/
/*					GPU Meshing Library 1.10				*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Easy mesh programing with OpenCl	*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		jul 02 2010							*/
/*	Last modification:	jan 19 2011							*/
/*															*/
/*----------------------------------------------------------*/


/*----------------------------------------------------------*/
/* Includes													*/
/*----------------------------------------------------------*/

#include <OpenCL/opencl.h>

/*----------------------------------------------------------*/
/* Set max users data										*/
/*----------------------------------------------------------*/

#define GmlMaxDat 100
#define GmlMaxKrn 100
#define GmlMaxSrcSiz 10000
#define GmlInternal 0
#define GmlInput 1
#define GmlOutput 2
#define GmlInout 3
#define GmlCpu 1
#define GmlGpu 2

typedef struct
{
	int NmbVer, NmbEle, EleSiz, *EleTab, VecSiz, NmbExtVer, NmbExtDat, *VerVec, (*VerHdr)[3], *VerExt;
}GmlBall;


/*----------------------------------------------------------*/
/* User procedures											*/
/*----------------------------------------------------------*/

int InitGPU(int);
void StopGPU();
int NewData(size_t, int, void **);
int FreeData(int);
int SetData(int);
int GetData(int);
int NewGPUCode(char *, char *);
double LaunchGPUCode(int, int ,int, ...);
size_t GetGPUMemoryUsage();
int BuildVerticesBall(GmlBall *);
