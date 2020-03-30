{
   int i;
   float4 wei[4], nul = (float4)(0.), crd = nul;

   for(i=0;i<OptCrdDegMax;i++)
   {
      wei[0] = wei[1] = wei[2] = wei[3] = nul;
      wei[ VerTetVoy[i] ] = (float4)(1.);
      crd += wei[0] * OptCrd[i][0] + wei[1] * OptCrd[i][1] + wei[2] * OptCrd[i][2] + wei[3] * OptCrd[i][3];
   }

   crd *= (float4)(1./crd.s3);
   crd.s3 = 0.;
   VerCrd = crd;
}
