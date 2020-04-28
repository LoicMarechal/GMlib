GrdExt = 0.5f * (GrdTet[0] + GrdTet[1]);

/* Boundary conditions treatment. */
if (TriRef == 1) GrdExt = GrdTet[0];
if (TriRef == 2) GrdExt = GrdTet[0];
if (TriRef == 3) GrdExt = 0.f;