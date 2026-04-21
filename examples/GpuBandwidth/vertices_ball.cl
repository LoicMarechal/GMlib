{
   int i;
   float4 dat = (float4)(0);

   for(i=0;i<TetDatDegMax;i++)
      dat += TetDat[i];

   VerDat = dat;
}
