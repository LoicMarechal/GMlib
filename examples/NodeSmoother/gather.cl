{
   int i;
   float4 crd = (float4)(0.);

   for(i=0;i<OptCrdDegMax;i++)
   {
      crd += (VerTetVoy[i] == 0) ? OptCrd[i][0] : (float4)(0);
      crd += (VerTetVoy[i] == 1) ? OptCrd[i][1] : (float4)(0);
      crd += (VerTetVoy[i] == 2) ? OptCrd[i][2] : (float4)(0);
      crd += (VerTetVoy[i] == 3) ? OptCrd[i][3] : (float4)(0);
   }

   crd *= (float4)(1./(OptCrdDeg));
   crd = mix(VerCrd, crd, (float)0.2);
   res = dot(VerCrd - crd, VerCrd - crd);
   VerCrd = crd;
}
