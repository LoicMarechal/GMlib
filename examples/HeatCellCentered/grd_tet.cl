/* Flux balance. */
/* Face ordering for tetrahedra {1,2,3,0}, {2,0,3,0}, {3,0,1,0}, {0,2,1,0} */
GrdTet = SolExt[0] * cross(VerCrd[2] - VerCrd[1], VerCrd[3] - VerCrd[1]) +
         SolExt[1] * cross(VerCrd[0] - VerCrd[2], VerCrd[3] - VerCrd[2]) +
         SolExt[2] * cross(VerCrd[0] - VerCrd[3], VerCrd[1] - VerCrd[3]) +
         SolExt[3] * cross(VerCrd[2] - VerCrd[0], VerCrd[1] - VerCrd[0]);
GrdTet = 0.5f * GrdTet / CalTetVol(VerCrd[0], VerCrd[1], VerCrd[2], VerCrd[3]);
