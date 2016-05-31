

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
#include "HexahedraDependenciesLoop.h"


/*----------------------------------------------------------*/
/* Read a mesh, send the data on the GPU, smooth the		*/
/* coordinates and get back the results.					*/
/*----------------------------------------------------------*/

int main(int ArgCnt, char **ArgVec)
{
	int iter, i, j, NmbVer, NmbHex, InpMsh, ver=0, dim=0, ref, (*HexTab)[8]=NULL, GpuIdx=0;
	int VerIdx, HexIdx, CalPos, CalCrd1, CalCrd2, BalIdx, PosIdx, NmbItr=100;
	float (*VerTab)[3]=NULL, dummy, FltTab[8], chk=0.;
	double GpuTim, total=0;


	/* If no arguments are give, print the help */

	if(ArgCnt == 1)
	{
		puts("\nHexahedraDependenciesLoop GPU_index");
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
	if( !(InpMsh = GmfOpenMesh("hexahedra.meshb", GmfRead, &ver, &dim)) || (ver != GmfFloat) || (dim != 3) )
		return(1);

	/* Read the number of vertices and elements and allocate the memory */
	if( !(NmbVer = GmfStatKwd(InpMsh, GmfVertices)) || !(VerTab = malloc((NmbVer+1) * 3 * sizeof(float))) )
		return(1);

	if( !(NmbHex = GmfStatKwd(InpMsh, GmfHexahedra)) || !(HexTab = malloc((NmbHex+1) * 8 * sizeof(int))) )
		return(1);

	/* Read the vertices */
	GmfGotoKwd(InpMsh, GmfVertices);
	for(i=1;i<=NmbVer;i++)
		GmfGetLin(InpMsh, GmfVertices, &VerTab[i][0], &VerTab[i][1], &VerTab[i][2], &ref);

	/* Read the elements */
	GmfGotoKwd(InpMsh, GmfHexahedra);
	for(i=1;i<=NmbHex;i++)
		GmfGetLin(	InpMsh, GmfHexahedra, &HexTab[i][0], &HexTab[i][1], &HexTab[i][2], &HexTab[i][3], \
					&HexTab[i][4], &HexTab[i][5], &HexTab[i][6], &HexTab[i][7], &ref );

	/* And close the mesh */
	GmfCloseMesh(InpMsh);


	/*---------------*/
	/* GPU COMPUTING */
	/*---------------*/

	/* Init the GMLIB and compile the OpenCL source code */
	if(!GmlInit(GpuIdx))
		return(1);

	if(!(CalPos = GmlNewKernel(HexahedraDependenciesLoop, "HexahedraScatter")))
		return(1);

	if(!(CalCrd1 = GmlNewKernel(HexahedraDependenciesLoop, "HexahedraGather1")))
		return(1);

	if(!(CalCrd2 = GmlNewKernel(HexahedraDependenciesLoop, "HexahedraGather2")))
		return(1);

	/* Create a vertices data type and transfer the data to the GPU */
	if(!(VerIdx = GmlNewData(GmlVertices, NmbVer, 0, GmlInout)))
		return(1);

	for(i=1;i<=NmbVer;i++)
		GmlSetVertex(VerIdx, i-1, VerTab[i][0], VerTab[i][1], VerTab[i][2]);

	GmlUploadData(VerIdx);

	/* Do the same with the elements */
	if(!(HexIdx = GmlNewData(GmlHexahedra, NmbHex, 0, GmlInput)))
		return(1);

	for(i=1;i<=NmbHex;i++)
		GmlSetHexahedron(	HexIdx, i-1, HexTab[i][0]-1, HexTab[i][1]-1, HexTab[i][2]-1, HexTab[i][3]-1, \
							HexTab[i][4]-1, HexTab[i][5]-1, HexTab[i][6]-1, HexTab[i][7]-1 );

	GmlUploadData(HexIdx);

	/* Create a data type with the list of incident elements to each vertices */
	BalIdx = GmlNewBall(VerIdx, HexIdx);
	GmlUploadBall(BalIdx);

	/* Create a raw datatype to store the elements scatter data. It does not need to be tranfered to the GPU */
	if(!(PosIdx = GmlNewData(GmlRawData, NmbHex, 8*sizeof(cl_float4), GmlOutput)))
		return(1);

	/* Smooth the coordinates a 100 times */
	for(iter=1;iter<=NmbItr;iter++)
	{
		/* SCATTER: Compute the new verties coordinates on the GPU but store them in an element based local buffer */
		GpuTim = GmlLaunchKernel(CalPos, NmbHex, 3, HexIdx, PosIdx, VerIdx);
    
		if(GpuTim < 0)
			return(1);
    
		total += GpuTim;
    
		/* GATHER: compute the average local element coordinates for each vertices */
		GpuTim = GmlLaunchBallKernel(CalCrd1, CalCrd2, BalIdx, 2, PosIdx, VerIdx);
    
		if(GpuTim < 0)
			return(1);
    
		total += GpuTim;
	}

	/* Get back the results and print some stats */
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
	GmlFreeData(HexIdx);
	GmlFreeData(PosIdx);
	GmlFreeBall(BalIdx);
	GmlStop();

	free(HexTab);
	free(VerTab);

	return(0);
}
