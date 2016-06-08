

/*----------------------------------------------------------*/
/*															*/
/*					GPU Meshing Library 2.01				*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Easy mesh programing with OpenCl	*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		jul 02 2010							*/
/*	Last modification:	oct 31 2013							*/
/*															*/
/*----------------------------------------------------------*/


/*----------------------------------------------------------*/
/* Includes													*/
/*----------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <unistd.h> 
#include <math.h> 
#include <limits.h> 
#include <stdarg.h>
#include "gmlib2.h"
#include "reduce.h"


/*----------------------------------------------------------*/
/* Macro instructions										*/
/*----------------------------------------------------------*/

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))
#define MB 1048576


/*----------------------------------------------------------*/
/* Library internal data structures							*/
/*----------------------------------------------------------*/

typedef struct
{
	int MshTyp, MemTyp, NmbLin, LinSiz, RedVecIdx;
	size_t siz;
	cl_mem GpuMem;
	void *CpuMem;
}GmlDatSct;

typedef struct
{
	int NmbPri, NmbSec, SecSiz, SecPadSiz, VecSiz, NmbExtPri, NmbExtDat;
	int PriVecIdx, PriExtIdx, ExtDatIdx, PriDegIdx;
}GmlBalSct;

typedef struct
{
	int idx, siz, DatTab[ GmlMaxDat ];
	cl_kernel kernel;
	cl_program program; 
}GmlKrnSct;

typedef struct
{
	int NmbKrn, CurDev, RedKrnIdx[10], ParIdx, NmbVer[ GmlHexahedra+1 ];
	cl_uint NmbDev;
	size_t MemSiz, CurLocSiz, MovSiz, MshSiz[ GmlHexahedra+1 ];
	GmlDatSct dat[ GmlMaxDat+1 ];
	GmlBalSct bal[ GmlMaxBal+1 ];
	GmlKrnSct krn[ GmlMaxKrn+1 ];
	cl_device_id device_id[ MaxGpu ];
	cl_context context;
	cl_command_queue queue;
}GmlSct;


/*----------------------------------------------------------*/
/* Global library variables									*/
/*----------------------------------------------------------*/

GmlSct gml;


/*----------------------------------------------------------*/
/* Init device, context and queue							*/
/*----------------------------------------------------------*/

GmlParSct *GmlInit(int mod)
{
	int err;
	cl_platform_id platforms[10];
	cl_uint num_platforms;
	GmlParSct *par;

	/* Select which device to run on */

	memset(&gml, 0, sizeof(GmlSct));
	gml.CurDev = mod;

	/* Set the mesh type size table */

	gml.MshSiz[ GmlRawData ]        = 1;
	gml.MshSiz[ GmlVertices ]       = sizeof(cl_float4);
	gml.MshSiz[ GmlEdges ]          = sizeof(cl_int2);
	gml.MshSiz[ GmlTriangles ]      = sizeof(cl_int4);
	gml.MshSiz[ GmlQuadrilaterals ] = sizeof(cl_int4);
	gml.MshSiz[ GmlTetrahedra ]     = sizeof(cl_int4);
	gml.MshSiz[ GmlHexahedra ]      = sizeof(cl_int8);

	/* Set the mesh type number of vertices table */

	gml.NmbVer[ GmlRawData ]        = 0;
	gml.NmbVer[ GmlVertices ]       = 0;
	gml.NmbVer[ GmlEdges ]          = 2;
	gml.NmbVer[ GmlTriangles ]      = 3;
	gml.NmbVer[ GmlQuadrilaterals ] = 4;
	gml.NmbVer[ GmlTetrahedra ]     = 4;
	gml.NmbVer[ GmlHexahedra ]      = 8;

	/* Init the GPU */

	if(clGetPlatformIDs(10, platforms, &num_platforms) != CL_SUCCESS)
		return(NULL);

	if(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MaxGpu, gml.device_id, &gml.NmbDev) != CL_SUCCESS)
		return(NULL);

	if( (mod < 0) || (mod >= gml.NmbDev) )
		return(NULL);

	if(!(gml.context = clCreateContext(0, 1, &gml.device_id[ gml.CurDev ], NULL, NULL, &err)))
		return(NULL);

	if(!(gml.queue = clCreateCommandQueue(gml.context, gml.device_id[ gml.CurDev ], CL_QUEUE_PROFILING_ENABLE, &err)))
		return(NULL);

	/* Load all internal reduction kernels */

	if(!(gml.RedKrnIdx[ GmlMin ] = GmlNewKernel(reduce, "reduce_min")))
		return(NULL);

	if(!(gml.RedKrnIdx[ GmlSum ] = GmlNewKernel(reduce, "reduce_sum")))
		return(NULL);

	if(!(gml.RedKrnIdx[ GmlMax ] = GmlNewKernel(reduce, "reduce_max")))
		return(NULL);

	/* Allocate and return a public user customizable parameter structure */

	if(!(gml.ParIdx = GmlNewData(GmlRawData, 1, sizeof(GmlParSct), GmlInout)))
		return(NULL);

	return(gml.dat[ gml.ParIdx ].CpuMem);
}


/*----------------------------------------------------------*/
/* Free OpenCL buffers and close the library				*/
/*----------------------------------------------------------*/

void GmlStop()
{
	int i;

	/* Free GPU memories, kernels and queue */

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


/*----------------------------------------------------------*/
/* Print all available OpenCL capable GPUs					*/
/*----------------------------------------------------------*/

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

	if(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MaxGpu, device_id, &num_devices) != CL_SUCCESS)
		return;

	for(i=0;i<num_devices;i++)
		if(clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, 100, GpuNam, &GpuNamSiz) == CL_SUCCESS)
			printf("      %d      : %s\n", i, GpuNam);
}


/*----------------------------------------------------------*/
/* Allocate an OpenCL buffer plus 10% more for resizing		*/
/*----------------------------------------------------------*/

int GmlNewData(int MshTyp, int NmbLin, int LinSiz, int MemTyp)
{
	int idx;
	GmlDatSct *dat;

	if( (MshTyp < GmlRawData) || (MshTyp > GmlHexahedra) || (MemTyp < GmlInternal) || (MemTyp > GmlInout) )
		return(0);

	/* Look for a free data socket */

	for(idx=1;idx<=GmlMaxDat;idx++)
		if(!gml.dat[ idx ].GpuMem)
			break;

	if(idx > GmlMaxDat)
		return(0);

	dat = &gml.dat[ idx ];
	dat->MshTyp = MshTyp;
	dat->MemTyp = MemTyp;
	dat->NmbLin = NmbLin;

	if(MshTyp == GmlRawData)
		dat->LinSiz = LinSiz;
	else
		dat->LinSiz = gml.MshSiz[ MshTyp ];

	dat->siz = dat->NmbLin * dat->LinSiz;
	dat->GpuMem = dat->CpuMem = NULL;

	/* Allocate the requested memory size on the GPU */

	if(MemTyp == GmlInput)
		dat->GpuMem = clCreateBuffer(gml.context, CL_MEM_READ_ONLY, dat->siz, NULL, NULL);
	else if(MemTyp == GmlOutput)
		dat->GpuMem = clCreateBuffer(gml.context, CL_MEM_WRITE_ONLY, dat->siz, NULL, NULL);
	else if((MemTyp == GmlInout) || (MemTyp == GmlInternal))
		dat->GpuMem = clCreateBuffer(gml.context, CL_MEM_READ_WRITE, dat->siz, NULL, NULL);

	if(!dat->GpuMem)
	{
		printf("Cannot allocate %ld MB on the GPU (%ld MB already used)\n", dat->siz/MB, GmlGetMemoryUsage()/MB);
		return(0);
	}

	/* Allocate the requested memory size on the CPU side */

	if( (MemTyp != GmlInternal) && !(dat->CpuMem = calloc(1, dat->siz)) )
		return(0);

	/* Keep track of allocated memory */

	gml.MemSiz += dat->siz;

	return(idx);
}


/*----------------------------------------------------------*/
/* Release an OpenCL buffer 								*/
/*----------------------------------------------------------*/

int GmlFreeData(int idx)
{
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Free both GPU and CPU memory buffers */

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


/*----------------------------------------------------------*/
/* Set a procedures to set or get lines of data				*/
/*----------------------------------------------------------*/

int GmlSetRawData(int idx, int lin, void *raw)
{
	char *adr;
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlRawData) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	adr = (void *)dat->CpuMem;
	memcpy(&adr[ lin*dat->LinSiz ], raw, dat->LinSiz);

	return(1);
}

int GmlGetRawData(int idx, int lin, void *raw)
{
	char *adr;
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlRawData) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	adr = (void *)dat->CpuMem;
	memcpy(raw, &adr[ lin*dat->LinSiz ], dat->LinSiz);

	return(1);
}

int GmlSetVertex(int idx, int lin, float x, float y, float z)
{
	float (*crd)[4];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlVertices) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	crd = (void *)dat->CpuMem;
	crd[ lin ][0] = x;
	crd[ lin ][1] = y;
	crd[ lin ][2] = z;
	crd[ lin ][3] = 0;

	return(1);
}

int GmlGetVertex(int idx, int lin, float *x, float *y, float *z)
{
	float (*crd)[4];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlVertices) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	crd = (void *)dat->CpuMem;
	*x = crd[ lin ][0];
	*y = crd[ lin ][1];
	*z = crd[ lin ][2];

	return(1);
}

int GmlSetEdge(int idx, int lin, int v1, int v2)
{
	int (*edg)[2];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlEdges) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	edg = (void *)dat->CpuMem;
	edg[ lin ][0] = v1;
	edg[ lin ][1] = v2;

	return(1);
}

int GmlSetTriangle(int idx, int lin, int v1, int v2, int v3)
{
	int (*tri)[4];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlTriangles) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	tri = (void *)dat->CpuMem;
	tri[ lin ][0] = v1;
	tri[ lin ][1] = v2;
	tri[ lin ][2] = v3;
	tri[ lin ][3] = -1;

	return(1);
}

int GmlSetQuadrilateral(int idx, int lin, int v1, int v2, int v3, int v4)
{
	int (*qad)[4];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlQuadrilaterals) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	qad = (void *)dat->CpuMem;
	qad[ lin ][0] = v1;
	qad[ lin ][1] = v2;
	qad[ lin ][2] = v3;
	qad[ lin ][3] = v4;

	return(1);
}

int GmlSetTetrahedron(int idx, int lin, int v1, int v2, int v3, int v4)
{
	int (*tet)[4];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlTetrahedra) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	tet = (void *)dat->CpuMem;
	tet[ lin ][0] = v1;
	tet[ lin ][1] = v2;
	tet[ lin ][2] = v3;
	tet[ lin ][3] = v4;

	return(1);
}

int GmlSetHexahedron(int idx, int lin, int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
{
	int (*hex)[8];
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || (dat->MshTyp != GmlHexahedra) || (lin < 0) || (lin >= dat->NmbLin) )
		return(0);

	hex = (void *)dat->CpuMem;
	hex[ lin ][0] = v1;
	hex[ lin ][1] = v2;
	hex[ lin ][2] = v3;
	hex[ lin ][3] = v4;
	hex[ lin ][4] = v5;
	hex[ lin ][5] = v6;
	hex[ lin ][6] = v7;
	hex[ lin ][7] = v8;

	return(1);
}


/*----------------------------------------------------------*/
/* Copy user's data into an OpenCL buffer 					*/
/*----------------------------------------------------------*/

int GmlUploadData(int idx)
{
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || !dat->CpuMem || (dat->MemTyp == GmlOutput) )
		return(0);

	/* Upload buffer from CPU ram to GPU ram and keep track of the amount of uploaded data */

	if(clEnqueueWriteBuffer(gml.queue, dat->GpuMem, CL_FALSE, 0, dat->siz, dat->CpuMem, 0, NULL,NULL) != CL_SUCCESS)
		return(0);
	else
	{
		gml.MovSiz += dat->siz;
		return(dat->siz);
	}
}


/*----------------------------------------------------------*/
/* Copy an OpenCL buffer into user's data					*/
/*----------------------------------------------------------*/

int GmlDownloadData(int idx)
{
	GmlDatSct *dat = &gml.dat[ idx ];

	/* Check indices */

	if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || !dat->CpuMem )
		return(0);

	/* Download buffer from GPU ram to CPU ram and keep track of the amount of downloaded data */

	if(clEnqueueReadBuffer(gml.queue, dat->GpuMem, CL_TRUE, 0, dat->siz, dat->CpuMem, 0, NULL, NULL ) != CL_SUCCESS)
		return(0);
	else
	{
		gml.MovSiz += dat->siz;
		return(dat->siz);
	}
}


/*----------------------------------------------------------*/
/* Vectorized arbitrary sized data type						*/
/*----------------------------------------------------------*/

int GmlNewBall(int typ1, int typ2)
{
	int i, j, PriIdx, SecIdx, ExtIdx, (*DegTab)[2], BalIdx, *SecTab, *PriVec, (*PriExt)[3], *ExtDat;
	unsigned char *PriDeg;
	GmlBalSct *bal;
	GmlDatSct *PriDat = &gml.dat[ typ1 ], *SecDat = &gml.dat[ typ2 ];

	/* Check input indices */

	if( (typ1 < 0) || (typ1 > GmlMaxDat) || (PriDat->MshTyp != GmlVertices) )
		return(0);

	if( (typ2 < 0) || (typ2 > GmlMaxDat) || (SecDat->MshTyp < GmlEdges) || (SecDat->MshTyp > GmlHexahedra) )
		return(0);

	/* Find a free ball structure */

	for(BalIdx=1; BalIdx<=GmlMaxBal; BalIdx++)
		if(!gml.bal[ BalIdx ].NmbPri)
			break;

	if(BalIdx > GmlMaxBal)
		return(0);

	/* Init the structure header and allocate primary tables */

	bal = &gml.bal[ BalIdx ];
	bal->NmbPri = PriDat->NmbLin;
	bal->NmbSec = SecDat->NmbLin;
	bal->SecSiz = gml.NmbVer[ SecDat->MshTyp ];
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

	bal->PriDegIdx = GmlNewData(GmlRawData, bal->NmbPri, sizeof(cl_char), GmlInput);
	bal->PriVecIdx = GmlNewData(GmlRawData, bal->NmbPri, bal->VecSiz * sizeof(cl_int), GmlInput);
	PriDeg = gml.dat[ bal->PriDegIdx ].CpuMem;
	PriVec = gml.dat[ bal->PriVecIdx ].CpuMem;

	/* Build temporary vertex degrees table */

	DegTab = calloc(bal->NmbPri, 2 * sizeof(int));

	for(i=0;i<bal->NmbSec;i++)
		for(j=0;j<bal->SecSiz;j++)
			DegTab[ SecTab[ i * bal->SecPadSiz + j ] ][0]++;

	/* Count the number of primary type elements whose size exceeds that of the vector
		and the number of extra datas needed */

	for(i=0;i<bal->NmbPri;i++)
		if(DegTab[i][0] > bal->VecSiz)
		{
			bal->NmbExtPri++;
			bal->NmbExtDat += DegTab[i][0] - bal->VecSiz;
		}

	/* Allocate both tables */

	bal->PriExtIdx = GmlNewData(GmlRawData, bal->NmbExtPri, 3 * sizeof(cl_int), GmlInput);
	bal->ExtDatIdx = GmlNewData(GmlRawData, bal->NmbExtDat, sizeof(cl_int), GmlInput);
	PriExt = gml.dat[ bal->PriExtIdx ].CpuMem;
	ExtDat = gml.dat[ bal->ExtDatIdx ].CpuMem;
	bal->NmbExtPri = bal->NmbExtDat = 0;

	/* Fill the primary type extension table */

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

	/* Fill the extension datas */

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

	/* Set the base degree for each vertices and set to -1 useless vector data */

	for(i=0;i<bal->NmbPri;i++)
	{
		PriDeg[i] = (DegTab[i][0] <= bal->VecSiz) ? DegTab[i][0] : bal->VecSiz;

		for(j=DegTab[i][0]; j<bal->VecSiz; j++)
			PriVec[ i * bal->VecSiz + j ] = -1;
	}

	free(DegTab);

	return(BalIdx);
}


/*----------------------------------------------------------*/
/* Release four OpenCL buffers associated with a ball		*/
/*----------------------------------------------------------*/

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


/*----------------------------------------------------------*/
/* Send all four data fields to the GPU						*/
/*----------------------------------------------------------*/

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


/*----------------------------------------------------------*/
/* Read and compile an OpenCL source code					*/
/*----------------------------------------------------------*/

int GmlNewKernel(char *KernelSource, char *PrcNam)
{
	char *buffer, *StrTab[1];
	int err, idx = ++gml.NmbKrn;
	GmlKrnSct *krn = &gml.krn[ idx ];
	size_t len, LenTab[1];

	if(idx > GmlMaxKrn)
		return(0);

	StrTab[0] = KernelSource;
	LenTab[0] = strlen(KernelSource)-1;

	/* Compile source code */

	if(!(krn->program = clCreateProgramWithSource(gml.context, 1, (const char **)StrTab, (const size_t *)LenTab, &err)))
		return(0);

	if(clBuildProgram(krn->program, 0, NULL, \
		"-cl-single-precision-constant -cl-mad-enable", NULL, NULL) != CL_SUCCESS)
	{
		clGetProgramBuildInfo(krn->program, gml.device_id[ gml.CurDev ], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

		if(!(buffer = malloc(len)))
			return(0);

		clGetProgramBuildInfo(krn->program, gml.device_id[ gml.CurDev ], CL_PROGRAM_BUILD_LOG, len, buffer, &len);
		printf("%s\n", buffer);
		free(buffer);
		return(0);
	}

	if(!(krn->kernel = clCreateKernel(krn->program, PrcNam, &err)) || (err != CL_SUCCESS))
		return(0);

/*	clGetProgramInfo(krn->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &len, NULL);
		printf("\ncompilation of %s : %ld bytes\n",PrcNam,len);*/

	return(idx);
}


/*----------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel				*/
/*----------------------------------------------------------*/

double GmlLaunchKernel(int idx, int TruSiz, int NmbDat, ...)
{
	int i, DatTab[ GmlMaxDat ];
	size_t GloSiz, LocSiz, RetSiz = 0;
	va_list VarArg;
	GmlDatSct *dat;
	GmlKrnSct *krn = &gml.krn[ idx ];
	cl_event event;
	cl_ulong start, end;

	if( (idx < 1) || (idx > gml.NmbKrn) || !krn->kernel )
		return(-1);

	/* First send the parameters to the GPU memmory */

	if(!GmlUploadData(gml.ParIdx))
		return(-2);

	/* Build arguments list */

	va_start(VarArg, NmbDat);

	for(i=0;i<NmbDat;i++)
		DatTab[i] = va_arg(VarArg, int);

	va_end(VarArg);

	for(i=0;i<NmbDat;i++)
	{
		dat = &gml.dat[ DatTab[i] ];

		if( (DatTab[i] < 1) || (DatTab[i] > GmlMaxDat) || !dat->GpuMem \
		|| (clSetKernelArg(krn->kernel, i, sizeof(cl_mem), &dat->GpuMem) != CL_SUCCESS) )
		{
			printf("i=%d, DatTab[i]=%d, GpuMem=%p\n",i,DatTab[i],dat->GpuMem);
			return(-3);
		}
	}

	if(clSetKernelArg(krn->kernel, NmbDat, sizeof(cl_mem), &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
		return(-4);

	if(clSetKernelArg(krn->kernel, NmbDat+1, sizeof(int), &TruSiz) != CL_SUCCESS)
		return(-5);

	/* Fit data loop size to the GPU kernel size */

	if(clGetKernelWorkGroupInfo(krn->kernel, gml.device_id[ gml.CurDev ], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), \
								&LocSiz, &RetSiz) != CL_SUCCESS)
	{
		return(-6);
	}

	gml.CurLocSiz = LocSiz;
	GloSiz = TruSiz / LocSiz;
	GloSiz *= LocSiz;

	if(GloSiz < TruSiz)
		GloSiz += LocSiz;

	/* Launch GPU code */

	clFinish(gml.queue);

	if(clEnqueueNDRangeKernel(gml.queue, krn->kernel, 1, NULL, &GloSiz, &LocSiz, 0, NULL, &event))
		return(-7);

	clFinish(gml.queue);

	if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) != CL_SUCCESS)
		return(-8);

	if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) != CL_SUCCESS)
		return(-9);

	/* Finaly, get back the parameters from the GPU memmory */

	if(!GmlDownloadData(gml.ParIdx))
		return(-10);

	return((double)(end - start) / 1e9);
}


/*----------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel				*/
/*----------------------------------------------------------*/

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

	if( (KrnIdx1 < 1) || (KrnIdx1 > gml.NmbKrn) || !krn1->kernel \
	||	(KrnIdx2 < 1) || (KrnIdx2 > gml.NmbKrn) || !krn2->kernel )
	{
		return(-1);
	}

	if( (BalIdx < 1) || (BalIdx > GmlMaxBal) || !bal->NmbPri )
		return(-2);

	for(i=0;i<NmbDat;i++)
		if( (DatIdxTab[i] < 1) || (DatIdxTab[i] > GmlMaxDat) || !DatTab[i]->GpuMem )
			return(-3);

	/* First send the parameters to the GPU memmory */

	if(!GmlUploadData(gml.ParIdx))
		return(-4);


	/*-----------------------------------*/
	/* Build first kernel arguments list */
	/*-----------------------------------*/

	if((clSetKernelArg(krn1->kernel, 0, sizeof(cl_mem), &gml.dat[ bal->PriDegIdx ].GpuMem) != CL_SUCCESS) \
	|| (clSetKernelArg(krn1->kernel, 1, sizeof(cl_mem), &gml.dat[ bal->PriVecIdx ].GpuMem) != CL_SUCCESS) )
	{
		return(-5);
	}

	for(i=0;i<NmbDat;i++)
		if(clSetKernelArg(krn1->kernel, i+2, sizeof(cl_mem), &DatTab[i]->GpuMem) != CL_SUCCESS)
			return(-6);

	if(clSetKernelArg(krn1->kernel, NmbDat+2, sizeof(cl_mem), &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
		return(-7);

	if(clSetKernelArg(krn1->kernel, NmbDat+3, sizeof(int), &bal->NmbPri) != CL_SUCCESS)
		return(-8);

	/* Fit data loop size to the GPU kernel size */

	if(clGetKernelWorkGroupInfo(krn1->kernel, gml.device_id[ gml.CurDev ], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), \
								&LocSiz, &RetSiz) != CL_SUCCESS)
	{
		return(-9);
	}

	GloSiz = bal->NmbPri / LocSiz;
	GloSiz *= LocSiz;

	if(GloSiz < bal->NmbPri)
		GloSiz += LocSiz;

	/* Launch GPU code */

	clFinish(gml.queue);

	if(clEnqueueNDRangeKernel(gml.queue, krn1->kernel, 1, NULL, &GloSiz, &LocSiz, 0, NULL, &event))
		return(-10);

	clFinish(gml.queue);

	if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) != CL_SUCCESS)
		return(-11);

	if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) != CL_SUCCESS)
		return(-12);

	tim1 = end - start;

	if(!bal->NmbExtPri)
		return(tim1 / 1e9);


	/*------------------------------------*/
	/* Build second kernel arguments list */
	/*------------------------------------*/

	if((clSetKernelArg(krn2->kernel, 0, sizeof(cl_mem), &gml.dat[ bal->PriExtIdx ].GpuMem) != CL_SUCCESS) \
	|| (clSetKernelArg(krn2->kernel, 1, sizeof(cl_mem), &gml.dat[ bal->ExtDatIdx ].GpuMem) != CL_SUCCESS) )
	{
		return(-13);
	}

	for(i=0;i<NmbDat;i++)
		if(clSetKernelArg(krn2->kernel, i+2, sizeof(cl_mem), &DatTab[i]->GpuMem) != CL_SUCCESS)
			return(-14);

	if(clSetKernelArg(krn2->kernel, NmbDat+2, sizeof(cl_mem), &gml.dat[ gml.ParIdx ].GpuMem) != CL_SUCCESS)
		return(-15);

	if(clSetKernelArg(krn2->kernel, NmbDat+3, sizeof(int), &bal->NmbExtPri) != CL_SUCCESS)
		return(-16);

	/* Fit data loop size to the GPU kernel size */

	if(clGetKernelWorkGroupInfo(krn2->kernel, gml.device_id[ gml.CurDev ], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), \
								&LocSiz, &RetSiz) != CL_SUCCESS)
	{
		return(-17);
	}

	GloSiz = bal->NmbExtPri / LocSiz;
	GloSiz *= LocSiz;

	if(GloSiz < bal->NmbExtPri)
		GloSiz += LocSiz;

	/* Launch GPU code */

	clFinish(gml.queue);

	if(clEnqueueNDRangeKernel(gml.queue, krn2->kernel, 1, NULL, &GloSiz, &LocSiz, 0, NULL, &event))
		return(-18);

	clFinish(gml.queue);

	if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) != CL_SUCCESS)
		return(-19);

	if(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) != CL_SUCCESS)
		return(-20);

	tim2 = end - start;

	/* Finaly, get back the parameters from the GPU memmory */

	if(!GmlDownloadData(gml.ParIdx))
		return(-21);

	return((tim1 + tim2) / 1e9);
}


/*----------------------------------------------------------*/
/* Compute various reduction functions: min,max,L1,L2 norms	*/
/*----------------------------------------------------------*/

double GmlReduceVector(int DatIdx, int opp, double *res)
{
	int i;
	float *vec;
	double tim;
	GmlDatSct *dat = &gml.dat[ DatIdx ], *dat2;

	/* Check indices */

	if( (DatIdx < 1) || (DatIdx > GmlMaxDat) || !dat->GpuMem || (opp < GmlMin) || (opp > GmlMax) )
		return(-1);

	/* Allocate an output vector the size of the input vector */

	if(!dat->RedVecIdx)
	 	if(!(dat->RedVecIdx = GmlNewData(GmlRawData, 1, dat->siz, GmlOutput)))
			return(-2);
			
	/* Launch the right reduction kernel according to the requested opperation */

	tim = GmlLaunchKernel(gml.RedKrnIdx[ opp ], dat->siz/sizeof(float), 2, DatIdx, dat->RedVecIdx);

	if(tim < 0)	
		return(tim);

	/* Trim the size of the output vector down to the number of OpenCL groups used by the kernel 
		and download this amount of data */

	dat2 = &gml.dat[ dat->RedVecIdx ];
	dat2->siz = dat->siz / gml.CurLocSiz;
	GmlDownloadData(dat->RedVecIdx);
	vec = (float *)dat2->CpuMem;

	/* Perform the last reduction step with the CPU */

	switch(opp)
	{
		case GmlMin :
		{
			*res = 1e37;
			for(i=0;i<dat2->siz/sizeof(float);i++)
				*res = min(*res, vec[i]);
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
				*res = max(*res, vec[i]);
		}break;
	}

	return(tim);
}


/*----------------------------------------------------------*/
/* Return memory currently allocated on the GPU				*/
/*----------------------------------------------------------*/

size_t GmlGetMemoryUsage()
{
	return(gml.MemSiz);
}


/*----------------------------------------------------------*/
/* Return memory currently transfered to or from the GPU	*/
/*----------------------------------------------------------*/

size_t GmlGetMemoryTransfer()
{
	return(gml.MovSiz);
}


/*----------------------------------------------------------*/
/* Return public parameter structure						*/
/*----------------------------------------------------------*/

GmlParSct *GmlGetParameters()
{
	return(gml.dat[ gml.ParIdx ].CpuMem);
}
