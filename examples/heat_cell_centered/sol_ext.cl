int i;

SolExt = 0.5f * (SolTet[0] + SolTet[1]);

/* Boundary conditions treatment. */
for (i = 0; i < 6; i++)
  if (GmlPar->BoCo[i] == DIRICHLET && GmlPar->BoCo_Ref[i] == TriRef)
    SolExt = 1.234;

// if(TriRef == 1) SolExt = 1.;
// if(TriRef == 2) SolExt = 2.;
// if(TriRef == 3) SolExt = 0.;

// if (TriRef > 0) GmlPar->Cnt = GmlPar->Cnt + 1;
