int      deg;
int16    bal;
double4  sol = {0,0,0,0}, nul = {0,0,0,0};

deg = DegTab[ cnt ];
bal = BalTab[ cnt ];

MulMatVec(MatTab[ cnt ][ 0], sol, RhsTab[ bal.s0 ]);
MulMatVec(MatTab[ cnt ][ 1], sol, (deg >  1) ? RhsTab[ bal.s1 ] : nul);
MulMatVec(MatTab[ cnt ][ 2], sol, (deg >  2) ? RhsTab[ bal.s2 ] : nul);
MulMatVec(MatTab[ cnt ][ 3], sol, (deg >  3) ? RhsTab[ bal.s3 ] : nul);
MulMatVec(MatTab[ cnt ][ 4], sol, (deg >  4) ? RhsTab[ bal.s4 ] : nul);
MulMatVec(MatTab[ cnt ][ 5], sol, (deg >  5) ? RhsTab[ bal.s5 ] : nul);
MulMatVec(MatTab[ cnt ][ 6], sol, (deg >  6) ? RhsTab[ bal.s6 ] : nul);
MulMatVec(MatTab[ cnt ][ 7], sol, (deg >  7) ? RhsTab[ bal.s7 ] : nul);
MulMatVec(MatTab[ cnt ][ 8], sol, (deg >  8) ? RhsTab[ bal.s8 ] : nul);
MulMatVec(MatTab[ cnt ][ 9], sol, (deg >  9) ? RhsTab[ bal.s9 ] : nul);
MulMatVec(MatTab[ cnt ][10], sol, (deg > 10) ? RhsTab[ bal.sa ] : nul);
MulMatVec(MatTab[ cnt ][11], sol, (deg > 11) ? RhsTab[ bal.sb ] : nul);
MulMatVec(MatTab[ cnt ][12], sol, (deg > 12) ? RhsTab[ bal.sc ] : nul);
MulMatVec(MatTab[ cnt ][13], sol, (deg > 13) ? RhsTab[ bal.sd ] : nul);
MulMatVec(MatTab[ cnt ][14], sol, (deg > 14) ? RhsTab[ bal.se ] : nul);
MulMatVec(MatTab[ cnt ][15], sol, (deg > 15) ? RhsTab[ bal.sf ] : nul);

SolTab[ cnt ] = sol;
