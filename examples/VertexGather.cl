   int i;
   float4 left  = (float4){0,0,0,0};
   float4 right = (float4){0,0,0,0};
   float4 norm  = (float4){1. / TetMidDeg, 1. / TetMidDeg, 1. / TetMidDeg, 1. / TetMidDeg};

   left.s0 = (float)TetMidDeg;
   right.s0 = (float)VerTetBal[0].s0;

   if(TetMidDeg >= 16)
      right.s1 = (float)VerTetBal[1].s0;

   if(TetMidDeg >= 32)
      right.s2 = (float)VerTetBal[2].s0;

   if(TetMidDeg >= 48)
      right.s3 = (float)VerTetBal[3].s0;

   SolAtVer[0] = left;
   SolAtVer[1] = right;
