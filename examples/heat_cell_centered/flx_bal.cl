/* Flux balance. */
/* Face ordering for tetrahedra {1,2,3,0}, {2,0,3,0}, {3,0,1,0}, {0,2,1,0} */
Rhs = dot(GrdExt[0], cross(VerCrd[2] - VerCrd[1], VerCrd[3] - VerCrd[1])) +
      dot(GrdExt[1], cross(VerCrd[0] - VerCrd[2], VerCrd[3] - VerCrd[2])) +
      dot(GrdExt[2], cross(VerCrd[0] - VerCrd[3], VerCrd[1] - VerCrd[3])) +
      dot(GrdExt[3], cross(VerCrd[2] - VerCrd[0], VerCrd[1] - VerCrd[0]));
Rhs = 0.5f * Rhs / CalTetVol(VerCrd[0], VerCrd[1], VerCrd[2], VerCrd[3]);