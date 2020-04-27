// SolExt = 0.5f * SolTet[0] + 0.5f * SolTet[1];

// /* Boundary conditions treatment. */
// if (TriRef == 1) SolExt = 1.f;
// if (TriRef == 2) SolExt = 2.f;
// if (TriRef == 3) SolExt = 0.f;

SolExt = (float) TriRef;