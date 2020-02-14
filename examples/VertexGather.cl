   int i;
   float4 left  = (float4){0,0,0,0};
   float4 right = (float4){0,0,0,0};

   for(i=0;i<TetMidDegMax;i++)
   {
      left  += TetMid[i];
      right += TetMid[i];
   }

   SolAtVer[0] = left  / TetMidDeg;
   SolAtVer[1] = right / TetMidDeg;
