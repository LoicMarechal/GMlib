{
   float  len[6], MaxEdg;
   float4 edg[6], nrm, ctr;
   
   edg[0] = VerCrd[0] - VerCrd[1];
   edg[1] = VerCrd[0] - VerCrd[2];
   edg[2] = VerCrd[0] - VerCrd[3];
   edg[3] = VerCrd[1] - VerCrd[2];
   edg[4] = VerCrd[1] - VerCrd[3];
   edg[5] = VerCrd[2] - VerCrd[3];
   
   len[0] = dot(edg[0], edg[0]);
   len[1] = dot(edg[1], edg[1]);
   len[2] = dot(edg[2], edg[2]);
   len[3] = dot(edg[3], edg[3]);
   len[4] = dot(edg[4], edg[4]);
   len[5] = dot(edg[5], edg[5]);

   MaxEdg = max(len[3], len[4]);
   MaxEdg = max(MaxEdg, len[5]);
   MaxEdg = 0.816496 * sqrt(MaxEdg);
   ctr = (VerCrd[1] + VerCrd[2] + VerCrd[3]) * (float4)(1./3.);
   nrm = fast_normalize(cross(edg[3], edg[5])) * (float4)(MaxEdg);
   OptCrd[0] = ctr + nrm;

   MaxEdg = max(len[1], len[2]);
   MaxEdg = max(MaxEdg, len[5]);
   MaxEdg = 0.816496 * sqrt(MaxEdg);
   ctr = (VerCrd[0] + VerCrd[2] + VerCrd[3]) * (float4)(1./3.);
   nrm = fast_normalize(cross(edg[1], edg[5])) * (float4)(MaxEdg);
   OptCrd[1] = ctr + nrm;

   MaxEdg = max(len[0], len[2]);
   MaxEdg = max(MaxEdg, len[4]);
   MaxEdg = 0.816496 * sqrt(MaxEdg);
   ctr = (VerCrd[1] + VerCrd[0] + VerCrd[3]) * (float4)(1./3.);
   nrm = fast_normalize(cross(edg[2], edg[4])) * (float4)(MaxEdg);
   OptCrd[2] = ctr + nrm;

   MaxEdg = max(len[0], len[1]);
   MaxEdg = max(MaxEdg, len[3]);
   MaxEdg = 0.816496 * sqrt(MaxEdg);
   ctr = (VerCrd[1] + VerCrd[2] + VerCrd[0]) * (float4)(1./3.);
   nrm = fast_normalize(cross(edg[3], edg[1])) * (float4)(MaxEdg);
   OptCrd[3] = ctr + nrm;
}
