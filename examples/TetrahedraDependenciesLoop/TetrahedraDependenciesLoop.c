

/*----------------------------------------------------------*/
/*															*/
/*					GPU Meshing Library 2.00				*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		loop on elements with indirect		*/
/*						writes to vertices					*/
/*	Author:				Loic MARECHAL						*/
/*	Creation date:		dec 03 2012							*/
/*	Last modification:	jan 20 2014							*/
/*															*/
/*----------------------------------------------------------*/


/*----------------------------------------------------------*/
/* Includes													*/
/*----------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libmesh5.h>
#include <gmlib2.h>
#include "TetrahedraDependenciesLoop.h"


/*----------------------------------------------------------*/
/* Read a mesh, send the data on the GPU, smooth the		*/
/* coordinates and get back the results.					*/
/*----------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
	int iter, i, j, NmbVer, NmbTet, InpMsh, ver=0, dim=0, ref, (*TetTab)[4]=NULL, GpuIdx=0;
	int VerIdx, TetIdx, CalPos, CalCrd1, CalCrd2, BalIdx, PosIdx, NmbItr=100;
	float (*VerTab)[3]=NULL, dummy, FltTab[8], chk=0.;
	double GpuTim, total=0;


	/* If no arguments are give, print the help */

	if(ArgCnt == 1)
	{
		puts("\nTetrahedraDependenciesLoop GPU_index");
		puts(" Choose GPU_index from the following list:");
		GmlListGPU();
		exit(0);
	}
	else
		GpuIdx = atoi(ArgVec[1]);


	/*--------------*/
	/* MESH READING */
	/*--------------*/

	/* Open the mesh */
	if( !(InpMsh = GmfOpenMesh("tetrahedra.meshb", GmfRead, &ver, &dim)) || (ver != GmfFloat) || (dim != 3) )
		return(1);

	/* Read the number of vertices and elements and allocate the memory */
	if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) || !(VerTab = malloc((NmbVer+1) * 3 * sizeof(float))) )
		return(1);

	if( !(NmbTet = GmfStatKwd(InpMsh, GmfTetrahedra)) || !(TetTab = malloc((NmbTet+1) * 4 * sizeof(int))) )
		return(1);

	/* Read the vertices */
	GmfGotoKwd(InpMsh, GmfVertices);
	for(i=1;i<=NmbVer;i++)
		GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

	/* Read the elements */
	GmfGotoKwd(InpMsh, GmfTetrahedra);
	for(i=1;i<=NmbTet;i++)
		GmfGetLin(InpMsh, GmfTetrahedra, &TetTab[i][0], &TetTab[i][1], &TetTab[i][2], &TetTab[i][3], &ref);

	/* And close the mesh */
	GmfCloseMesh(InpMsh);


	/*---------------*/
	/* GPU COMPUTING */
	/*---------------*/

	/* Init the GMLIB and compile the OpenCL source code */
	if(!GmlInit(GpuIdx))
		return(1);

	if(!(CalPos = GmlNewKernel(TetrahedraDependenciesLoop, "TetrahedraScatter")))
		return(1);

	if(!(CalCrd1 = GmlNewKernel(TetrahedraDependenciesLoop, "TetrahedraGather1")))
		return(1);

	if(!(CalCrd2 = GmlNewKernel(TetrahedraDependenciesLoop, "TetrahedraGather2")))
		return(1);

	/* Create a vertices data type and transfer the data to the GPU */
	if(!(VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInout)))
		return(1);

	for(i=1;i<=NmbVer;i++)
		GmlSetVertex(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2]);

	GmlUploadData(VerIdx);

	/* Do the same with the elements */
	if(!(TetIdx = GmlNewData(GmlTetrahedra, NmbTet, 0, GmlInput)))
		return(1);

	for(i=1;i<=NmbTet;i++)
		GmlSetTetrahedron(TetIdx, i-1, TetTab[i][0]-1, TetTab[i][1]-1, TetTab[i][2]-1, TetTab[i][3]-1);

	GmlUploadData(TetIdx);

	/* Create a data type with the list of incident elements to each vertices */
	BalIdx = GmlNewBall(VerIdx, TetIdx);
	GmlUploadBall(BalIdx);

	/* Create a raw datatype to store the elements scatter data. It does not need to be tranfered to the GPU */
	if(!(PosIdx = GmlNewData(GmlRawData, NmbTet, 4*sizeof(cl_float4), GmlOutput)))
		return(1);

	/* Smooth the coordinates a 100 times */
	for(iter=1;iter<=NmbItr;iter++)
	{
		/* SCATTER: Compute the new verties coordinates on the GPU but store them in an element based local buffer */
		GpuTim = GmlLaunchKernel(CalPos, NmbTet, 3, TetIdx, PosIdx, VerIdx);
    
		if(GpuTim < 0)
			return(1);
    
		total += GpuTim;
    
		/* GATHER: compute the average local element coordinates for each vertices */
		GpuTim = GmlLaunchBallKernel(CalCrd1, CalCrd2, BalIdx, 2, PosIdx, VerIdx);
    
		if(GpuTim < 0)
			return(1);
    
		total += GpuTim;
	}

	/* Get the results back and print some stats */
	GmlDownloadData(VerIdx);

	for(i=1;i<=NmbVer;i++)
	{
		GmlGetVertex(VerIdx, i-1, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2]);
		chk += sqrt(VerTab[i][0] * VerTab[i][0] + VerTab[i][1] * VerTab[i][1] + VerTab[i][2] * VerTab[i][2]);
	}

	printf("%d coordinates smoothed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
		NmbVer*NmbItr, total, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbVer );


	/*-----*/
	/* END */
	/*-----*/

	GmlFreeData(VerIdx);
	GmlFreeData(TetIdx);
	GmlFreeData(PosIdx);
	GmlFreeBall(BalIdx);
	GmlStop();

	free(TetTab);
	free(VerTab);

	return(0);
}
