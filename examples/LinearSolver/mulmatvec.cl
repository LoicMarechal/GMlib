int      deg;
int16    bal;
double4  sol, nul = {0,0,0,0};

deg = DegTab[ cnt ];
bal = BalTab[ cnt ];
sol = nul;

              MulMatVec(MatTab[ cnt ][ 0], sol, RhsTab[ bal.s0 ]);
if (deg >  1) MulMatVec(MatTab[ cnt ][ 1], sol, RhsTab[ bal.s1 ]);
if (deg >  2) MulMatVec(MatTab[ cnt ][ 2], sol, RhsTab[ bal.s2 ]);
if (deg >  3) MulMatVec(MatTab[ cnt ][ 3], sol, RhsTab[ bal.s3 ]);
if (deg >  4) MulMatVec(MatTab[ cnt ][ 4], sol, RhsTab[ bal.s4 ]);
if (deg >  5) MulMatVec(MatTab[ cnt ][ 5], sol, RhsTab[ bal.s5 ]);
if (deg >  6) MulMatVec(MatTab[ cnt ][ 6], sol, RhsTab[ bal.s6 ]);
if (deg >  7) MulMatVec(MatTab[ cnt ][ 7], sol, RhsTab[ bal.s7 ]);
if (deg >  8) MulMatVec(MatTab[ cnt ][ 8], sol, RhsTab[ bal.s8 ]);
if (deg >  9) MulMatVec(MatTab[ cnt ][ 9], sol, RhsTab[ bal.s9 ]);
if (deg > 10) MulMatVec(MatTab[ cnt ][10], sol, RhsTab[ bal.sa ]);
if (deg > 11) MulMatVec(MatTab[ cnt ][11], sol, RhsTab[ bal.sb ]);
if (deg > 12) MulMatVec(MatTab[ cnt ][12], sol, RhsTab[ bal.sc ]);
if (deg > 13) MulMatVec(MatTab[ cnt ][13], sol, RhsTab[ bal.sd ]);
if (deg > 14) MulMatVec(MatTab[ cnt ][14], sol, RhsTab[ bal.se ]);
if (deg > 15) MulMatVec(MatTab[ cnt ][15], sol, RhsTab[ bal.sf ]);

SolTab[ cnt ] = sol;
