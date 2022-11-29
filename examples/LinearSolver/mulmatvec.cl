int      deg;
int16    bal;
float4   sol;

deg = DegTab[ cnt ];
bal = BalTab[ cnt ];

             sol  = MulMatVec(MatTab[ cnt ][ 0], RhsTab[ bal.s0 ]);
if(deg >  1) sol += MulMatVec(MatTab[ cnt ][ 1], RhsTab[ bal.s1 ]);
if(deg >  2) sol += MulMatVec(MatTab[ cnt ][ 2], RhsTab[ bal.s2 ]);
if(deg >  3) sol += MulMatVec(MatTab[ cnt ][ 3], RhsTab[ bal.s3 ]);
if(deg >  4) sol += MulMatVec(MatTab[ cnt ][ 4], RhsTab[ bal.s4 ]);
if(deg >  5) sol += MulMatVec(MatTab[ cnt ][ 5], RhsTab[ bal.s5 ]);
if(deg >  6) sol += MulMatVec(MatTab[ cnt ][ 6], RhsTab[ bal.s6 ]);
if(deg >  7) sol += MulMatVec(MatTab[ cnt ][ 7], RhsTab[ bal.s7 ]);
if(deg >  8) sol += MulMatVec(MatTab[ cnt ][ 8], RhsTab[ bal.s8 ]);
if(deg >  9) sol += MulMatVec(MatTab[ cnt ][ 9], RhsTab[ bal.s9 ]);
if(deg > 10) sol += MulMatVec(MatTab[ cnt ][10], RhsTab[ bal.sa ]);
if(deg > 11) sol += MulMatVec(MatTab[ cnt ][11], RhsTab[ bal.sb ]);
if(deg > 12) sol += MulMatVec(MatTab[ cnt ][12], RhsTab[ bal.sc ]);
if(deg > 13) sol += MulMatVec(MatTab[ cnt ][13], RhsTab[ bal.sd ]);
if(deg > 14) sol += MulMatVec(MatTab[ cnt ][14], RhsTab[ bal.se ]);
if(deg > 15) sol += MulMatVec(MatTab[ cnt ][15], RhsTab[ bal.sf ]);

SolTab[ cnt ] = sol;
