   int i;
   float4 left  = (float4)(0,0,0,0);
   float4 right = (float4)(0,0,0,0);
   
   for(i=0;i<TetMidDeg;i++)
   {
      left  += TetMid[i][0];
      right += TetMid[i][1];
   }
   
   left  /= (float4)(TetMidDeg, TetMidDeg, TetMidDeg, TetMidDeg);
   right /= (float4)(TetMidDeg, TetMidDeg, TetMidDeg, TetMidDeg);
   
   SolAtVer[0] = left;
   SolAtVer[1] = right;
