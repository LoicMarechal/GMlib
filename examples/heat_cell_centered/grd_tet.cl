/* Normal vectors. */
float4 n0 = (float4)(0., 0., 0., 0.);
float4 n1 = (float4)(0., 0., 0., 0.);
float4 n2 = (float4)(0., 0., 0., 0.);
float4 n3 = (float4)(0., 0., 0., 0.);

/* Face ordering for tetrahedra {1,2,3,0}, {2,0,3,0}, {3,0,1,0}, {0,2,1,0} */
n0 = 0.5f * cross(VerCrd[3] - VerCrd[1], VerCrd[2] - VerCrd[1]);
n1 = 0.5f * cross(VerCrd[3] - VerCrd[2], VerCrd[0] - VerCrd[2]);
n2 = 0.5f * cross(VerCrd[1] - VerCrd[3], VerCrd[0] - VerCrd[3]);
n3 = 0.5f * cross(VerCrd[1] - VerCrd[0], VerCrd[2] - VerCrd[0]);

/* Flux balance. */
GrdTet = n0 * SolExt[0] + n1 * SolExt[1] + n2 * SolExt[2] + n3 * SolExt[3];