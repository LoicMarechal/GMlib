{
   float len, srf, vol;

   len =          dot(VerCrd[0] - VerCrd[1], VerCrd[0] - VerCrd[1]);
   len = max(len, dot(VerCrd[0] - VerCrd[2], VerCrd[0] - VerCrd[2]));
   len = max(len, dot(VerCrd[0] - VerCrd[3], VerCrd[0] - VerCrd[3]));
   len = max(len, dot(VerCrd[1] - VerCrd[2], VerCrd[1] - VerCrd[2]));
   len = max(len, dot(VerCrd[1] - VerCrd[3], VerCrd[1] - VerCrd[3]));
   len = max(len, dot(VerCrd[2] - VerCrd[3], VerCrd[2] - VerCrd[3]));

   srf = CalTriSrf(VerCrd[0], VerCrd[1], VerCrd[2])
       + CalTriSrf(VerCrd[0], VerCrd[1], VerCrd[3]);
       + CalTriSrf(VerCrd[1], VerCrd[2], VerCrd[3]);
       + CalTriSrf(VerCrd[2], VerCrd[0], VerCrd[3]);

   vol = CalTetVol(VerCrd[0], VerCrd[1], VerCrd[2], VerCrd[3]);

   qal = 7.348469 * vol / (srf * sqrt(len));
}
