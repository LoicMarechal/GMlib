int i;

GrdExt = 0.5f * (GrdTet[0] + GrdTet[1]);

/* Boundary conditions treatment. */
for (i = 0; i < 6; i++) {
  if (GmlPar->BoCo[i] == DIRICHLET && GmlPar->BoCo_Ref[i] == TriRef) {
    GrdExt = GrdTet[0];
  }
  if (GmlPar->BoCo[i] == NEUMANN && GmlPar->BoCo_Ref[i] == TriRef) {
    GrdExt = GmlPar->BoCo_Val[i];
  }
}
