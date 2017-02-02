

/*----------------------------------------------------------*/
/*															*/
/*					GPU Meshing Library 2.00				*/
/*															*/
/*----------------------------------------------------------*/
/*															*/
/*	Description:		Basic loop on elements				*/
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
#include "TetrahedraBasicLoop.h"


/*----------------------------------------------------------*/
/* Read an element mesh, send the data on the GPU, compute	*/
/* the elements middle and get back the results.			*/
/*----------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
	int i, NmbVer, NmbTet, InpMsh, ver=0, dim=0, ref, (*TetTab)[4]=NULL, VerIdx, TetIdx, MidIdx, CalMid, GpuIdx=0;
	float (*VerTab)[3]=NULL, (*MidTab)[4], dummy, chk=0.;
	double GpuTim;


	/* If no arguments are give, print the help */

	if(ArgCnt == 1)
	{
		puts("\nTetrahedraBasicLoop GPU_index");
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

	if(!(CalMid = GmlNewKernel(TetrahedraBasicLoop, "TetrahedraBasic")))
		return(1);

	/* Create a vertices data type and transfer the data to the GPU */
	if(!(VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInput)))
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

	/* Create a raw datatype to store the element middles. It does not need to be tranfered to the GPU */
	if(!(MidIdx = GmlNewData(GmlRawData, NmbTet, sizeof(cl_float4), GmlOutput)))
		return(1);

	if(!(MidTab = malloc((NmbTet+1)*4*sizeof(float))))
		return(1);

	/* Launch the kernel on the GPU */
	GpuTim = GmlLaunchKernel(CalMid, NmbTet, 3, TetIdx, MidIdx, VerIdx);

	if(GpuTim < 0)
		return(1);

	/* Get the results back and print some stats */
	GmlDownloadData(MidIdx);

	for(i=1;i<=NmbTet;i++)
	{
		GmlGetRawData(MidIdx, i-1, MidTab[i]);
		chk += sqrt(MidTab[i][0] * MidTab[i][0] + MidTab[i][1] * MidTab[i][1] + MidTab[i][2] * MidTab[i][2]);
	}

	printf("%d tet centers computed in %g seconds, %ld MB used, %ld MB transfered, checksum = %g\n",
		NmbTet, GpuTim, GmlGetMemoryUsage()/1048576, GmlGetMemoryTransfer()/1048576, chk / NmbTet );


	/*-----*/
	/* END */
	/*-----*/

	GmlFreeData(VerIdx);
	GmlFreeData(TetIdx);
	GmlFreeData(MidIdx);
	GmlStop();

	free(MidTab);
	free(TetTab);
	free(VerTab);

	return(0);
}
