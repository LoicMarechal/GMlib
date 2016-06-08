

/*----------------------------------------------------------*/
/*															*/
/*					GPU Meshing Library 1.10				*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Easy mesh programing with OpenCl	*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		jul 02 2010							*/
/*	Last modification:	dec 01 2011							*/
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
#include <stdarg.h>
#include "gmlib1.h"


/*----------------------------------------------------------*/
/* Library internal data structures							*/
/*----------------------------------------------------------*/

typedef struct
{
	int idx, typ;
	size_t siz;
	cl_mem GpuMem;
	void *CpuMem;
}GmlDatSct;

typedef struct
{
	int idx, siz, DatTab[ GmlMaxDat ];
	cl_kernel kernel;
	cl_program program; 
}GmlKrnSct;

typedef struct
{
	int NmbKrn;
	size_t MemSiz;
	GmlDatSct dat[ GmlMaxDat+1 ];
	GmlKrnSct krn[ GmlMaxKrn+1 ];
	cl_device_id device_id;
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

int InitGPU(int mod)
{
	int err;
	cl_platform_id platforms[10];
	cl_uint num_platforms;

	if(mod == GmlCpu)
		mod = CL_DEVICE_TYPE_CPU;
	else if(mod == GmlGpu)
		mod = CL_DEVICE_TYPE_GPU;
	else
		return(0);

	memset(&gml, 0, sizeof(GmlSct));

	if(clGetPlatformIDs(10, platforms, &num_platforms) != CL_SUCCESS)
		return(0);

	if(clGetDeviceIDs(platforms[0], mod, 1, &gml.device_id, NULL) != CL_SUCCESS)
		return(0);

	if(!(gml.context = clCreateContext(0, 1, &gml.device_id, NULL, NULL, &err)))
		return(0);

	if(!(gml.queue = clCreateCommandQueue(gml.context, gml.device_id, CL_QUEUE_PROFILING_ENABLE, &err)))
		return(0);

	return(1);
}


/*----------------------------------------------------------*/
/* Free OpenCL buffers and close the library				*/
/*----------------------------------------------------------*/

void StopGPU()
{
	int i;

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
/* Allocate an OpenCL buffer plus 10% more for resizing		*/
/*----------------------------------------------------------*/

int NewData(size_t siz, int typ, void **CpuMem)
{
	int idx;
	GmlDatSct *dat;

	for(idx=1;idx<=GmlMaxDat;idx++)
		if(!gml.dat[ idx ].GpuMem)
			break;

	if(idx > GmlMaxDat)
		return(0);

	dat = &gml.dat[ idx ];
	dat->siz = siz;
	dat->typ = typ;
	dat->GpuMem = dat->CpuMem = NULL;

	if(typ == GmlInput)
		dat->GpuMem = clCreateBuffer(gml.context, CL_MEM_READ_ONLY, dat->siz, NULL, NULL);
	else if(typ == GmlOutput)
		dat->GpuMem = clCreateBuffer(gml.context, CL_MEM_WRITE_ONLY, dat->siz, NULL, NULL);
	else if((typ == GmlInout) || (typ == GmlInternal))
		dat->GpuMem = clCreateBuffer(gml.context, CL_MEM_READ_WRITE, dat->siz, NULL, NULL);

	if(!dat->GpuMem)
	{
		printf("Cannot allocate %ld MB on the GPU\n", siz / 1048576);
		return(0);
	}

	if(typ != GmlInternal)
	{
		dat->CpuMem = *CpuMem = malloc(siz);

		if(!dat->CpuMem)
			return(0);
	}

	gml.MemSiz += siz;

	return(idx);
}


/*----------------------------------------------------------*/
/* Release an OpenCL buffer 								*/
/*----------------------------------------------------------*/

int FreeData(int idx)
{
	if( (idx >= 1) && (idx <= GmlMaxDat) && gml.dat[ idx ].GpuMem )
	{
		if(clReleaseMemObject(gml.dat[ idx ].GpuMem) != CL_SUCCESS)
			return(0);

		gml.dat[ idx ].GpuMem = NULL;
		gml.MemSiz -= gml.dat[ idx ].siz;

		if(gml.dat[ idx ].CpuMem)
			free(gml.dat[ idx ].CpuMem);

		return(1);
	}
	else
		return(0);
}


/*----------------------------------------------------------*/
/* Copy user's data into an OpenCL buffer 					*/
/*----------------------------------------------------------*/

int SetData(int idx)
{
	GmlDatSct *dat = &gml.dat[ idx ];

	if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || (dat->typ == GmlOutput) )
		return(0);

	if(clEnqueueWriteBuffer(gml.queue, dat->GpuMem, CL_FALSE, 0, dat->siz, dat->CpuMem, 0, NULL,NULL) != CL_SUCCESS)
		return(0);
	else
		return(dat->siz);
}


/*----------------------------------------------------------*/
/* Copy an OpenCL buffer into user's data					*/
/*----------------------------------------------------------*/

int GetData(int idx)
{
	GmlDatSct *dat = &gml.dat[ idx ];

	if( (idx < 1) || (idx > GmlMaxDat) || !dat->GpuMem || (dat->typ == GmlInput) )
		return(0);

	if(clEnqueueReadBuffer(gml.queue, dat->GpuMem, CL_TRUE, 0, dat->siz, dat->CpuMem, 0, NULL, NULL ) != CL_SUCCESS)
		return(0);
	else
		return(dat->siz);
}


/*----------------------------------------------------------*/
/* Read and compile an OpenCL source code					*/
/*----------------------------------------------------------*/

int NewGPUCode(char *KernelSource, char *PrcNam)
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
		clGetProgramBuildInfo(krn->program, gml.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

		if(!(buffer = malloc(len)))
			return(0);

		clGetProgramBuildInfo(krn->program, gml.device_id, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
		printf("%s\n", buffer);
		free(buffer);
		return(0);
	}

	if(!(krn->kernel = clCreateKernel(krn->program, PrcNam, &err)) || (err != CL_SUCCESS))
		return(0);

/*	clGetProgramInfo(krn->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &len, NULL);
	printf("compilation done : %ld bytes\n",len);*/

	return(idx);
}


/*----------------------------------------------------------*/
/* Select arguments and launch an OpenCL kernel				*/
/*----------------------------------------------------------*/

double LaunchGPUCode(int idx, int TruSiz, int NmbDat, ...)
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
			return(-1);
		}
	}

	if(clSetKernelArg(krn->kernel, NmbDat, sizeof(int), &TruSiz) != CL_SUCCESS)
		return(-1);

	/* Fit data loop size to the GPU kernel size */

	if(clGetKernelWorkGroupInfo(krn->kernel, gml.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), \
								&LocSiz, &RetSiz) != CL_SUCCESS)
	{
		return(-1);
	}

	GloSiz = TruSiz / LocSiz;
	GloSiz *= LocSiz;

	if(GloSiz < TruSiz)
		GloSiz += LocSiz;

	/* Launch GPU code */

	clFinish(gml.queue);

	if(clEnqueueNDRangeKernel(gml.queue, krn->kernel, 1, NULL, &GloSiz, &LocSiz, 0, NULL, &event))
		return(-1);

	clFinish(gml.queue);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

	return((double)(end - start) / 1e9);
}

/*----------------------------------------------------------*/
/* Return memory currently allocated on the GPU				*/
/*----------------------------------------------------------*/

size_t GetGPUMemoryUsage()
{
	return(gml.MemSiz);
}


/*----------------------------------------------------------*/
/* Build and store vertices ball in vectors					*/
/*----------------------------------------------------------*/

int BuildVerticesBall(GmlBall *bal)
{
	int i, j, VerIdx, ExtIdx, (*DegTab)[2];
	puts("1");
	bal->VecSiz = (bal->EleSiz * bal->NmbEle) / bal->NmbVer;
	bal->NmbExtVer = bal->NmbExtDat = 0;

	if(bal->VecSiz < 2)
		bal->VecSiz = 2;
	else if(bal->VecSiz < 4)
		bal->VecSiz = 4;
	else if(bal->VecSiz < 8)
		bal->VecSiz = 8;
	else
		bal->VecSiz = 16;

	puts("2");
	DegTab = calloc(bal->NmbVer, 2 * sizeof(int));
	bal->VerVec = malloc(bal->NmbVer * bal->VecSiz * sizeof(int));

	for(i=0;i<bal->NmbEle;i++)
		for(j=0;j<bal->EleSiz;j++)
			DegTab[ bal->EleTab[ i * bal->EleSiz + j ] - 1 ][0]++;

		puts("3");
	for(i=0;i<bal->NmbVer;i++)
		if(DegTab[i][0] > bal->VecSiz)
		{
			bal->NmbExtVer++;
			bal->NmbExtDat += DegTab[i][0] - bal->VecSiz;
		}

		puts("4");
	bal->VerHdr = calloc(bal->NmbExtVer, 3 * sizeof(int));
	bal->VerExt = malloc(bal->NmbExtDat * sizeof(int));
	bal->NmbExtVer = bal->NmbExtDat = 0;

	for(i=0;i<bal->NmbVer;i++)
	{
		if(DegTab[i][0] > bal->VecSiz)
		{
			bal->VerHdr[ bal->NmbExtVer ][0] = i;
			bal->VerHdr[ bal->NmbExtVer ][1] = bal->NmbExtDat;
			DegTab[i][1] = bal->NmbExtVer;
			bal->NmbExtVer++;
			bal->NmbExtDat += DegTab[i][0] - bal->VecSiz;
		}

		DegTab[i][0] = 0;
	}

		puts("5");
	for(i=0;i<bal->NmbEle;i++)
		for(j=0;j<bal->EleSiz;j++)
		{
			VerIdx = bal->EleTab[ i * bal->EleSiz + j ] - 1;

			if(DegTab[ VerIdx ][0] < bal->VecSiz)
				bal->VerVec[ VerIdx * bal->VecSiz + DegTab[ VerIdx ][0] ] = (i << 3) | j;
			else
			{
				ExtIdx = DegTab[ VerIdx ][1];
				bal->VerExt[ bal->VerHdr[ ExtIdx ][1] + bal->VerHdr[ ExtIdx ][2] ] = (i << 3) | j;
				bal->VerHdr[ ExtIdx ][2]++;
			}

			DegTab[ VerIdx ][0]++;
		}

	for(i=0;i<bal->NmbVer;i++)
		for(j=DegTab[i][0]; j < bal->VecSiz; j++)
			bal->VerVec[ i * bal->VecSiz + j ] = -1;

	free(DegTab);

	puts("6");

/*	for(i=0;i<bal->NmbVer;i++)
	{
		printf("ver %d : ", i+1);
		for(j=0;j<8;j++)
			if(bal->VerVec[ i * bal->VecSiz + j ] != -1)
				printf("%d, ",bal->VerVec[ i * bal->VecSiz + j ] + 1);
		puts("");
	}

	for(i=0;i<bal->NmbExtVer;i++)
	{
		printf("ver %d : ", bal->VerHdr[i][0]+1);
		for(j=0;j<bal->VerHdr[i][2];j++)
			printf("%d, ", bal->VerExt[ bal->VerHdr[i][1] + j ] + 1);
		puts("");
	}*/
}
