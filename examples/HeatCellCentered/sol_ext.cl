int i;

SolExt = 0.5f * (SolTet[0] + SolTet[1]);

/* Boundary conditions treatment. */
for (i = 0; i < 6; i++) {
  if (GmlPar->BoCo[i] == DIRICHLET && GmlPar->BoCo_Ref[i] == TriRef) {
    SolExt = GmlPar->BoCo_Val[i];
  }
  if (GmlPar->BoCo[i] == NEUMANN && GmlPar->BoCo_Ref[i] == TriRef) {
    SolExt = SolTet[0];
  }
}
