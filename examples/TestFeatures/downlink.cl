   float4 left, right;

   left    = SolAtVer[0][0] + SolAtVer[1][0] + SolAtVer[2][0] + SolAtVer[3][0];
   right   = SolAtVer[0][1] + SolAtVer[1][1] + SolAtVer[2][1] + SolAtVer[3][1];

   TetMid  = (VerCrd[0] + VerCrd[1] + VerCrd[2] + VerCrd[3]) * (left + right);
   TetMid  = TetMid * rsqrt(dot(TetMid, TetMid));

   ResVec  = (float)(count.s0) / GmlPar->res;
