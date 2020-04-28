//SolExt = 0.5f * (SolTet[0] + SolTet[1]);

/* Boundary conditions treatment. */

float RefVal[6] = {0.0, 1.0, 2.0, 0.0, 0.0, 0.0};
SolExt = RefVal[ TriRef ];


//float val = 0.;
//if(TriRef == 1) val = 1.;
//if(TriRef == 2) val = 2.;
//SolExt = val;
