int      i;
double4  sol = (double4){0.,0.,0.,0.};

for(i=0;i<DegTab[ cnt ];i++)
   sol += MulMatVec(MatTab[ cnt ][i], RhsTab[ BalTab[ cnt ][i] ]);

SolTab[ cnt ] = sol;
