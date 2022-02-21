{
   int i, j, k;
   float len, MaxLen, srf, volP1, volP2, MinJac, MaxJac, N, s[64];
   float4 n, u[4], v[4], w[4];

   // Search for the longest P2 edge length
   len = CalLen(VerCrd[0], MidCrd[0]) + CalLen( MidCrd[0], VerCrd[1]);
   MaxLen = len;
   len = CalLen(VerCrd[1], MidCrd[1]) + CalLen( MidCrd[1], VerCrd[2]);
   MaxLen = max(MaxLen, len);
   len = CalLen(VerCrd[2], MidCrd[2]) + CalLen( MidCrd[2], VerCrd[0]);
   MaxLen = max(MaxLen, len);
   len = CalLen(VerCrd[3], MidCrd[3]) + CalLen( MidCrd[3], VerCrd[0]);
   MaxLen = max(MaxLen, len);
   len = CalLen(VerCrd[3], MidCrd[4]) + CalLen( MidCrd[4], VerCrd[1]);
   MaxLen = max(MaxLen, len);
   len = CalLen(VerCrd[3], MidCrd[5]) + CalLen( MidCrd[5], VerCrd[2]);
   MaxLen = max(MaxLen, len);

   // Compute the tet's P2 faces area
   srf = CalSrf(VerCrd[0], MidCrd[0], MidCrd[3])
       + CalSrf(MidCrd[0], MidCrd[4], MidCrd[3])
       + CalSrf(MidCrd[0], VerCrd[1], MidCrd[4])
       + CalSrf(MidCrd[3], MidCrd[4], VerCrd[3])
       + CalSrf(VerCrd[1], MidCrd[1], MidCrd[4])
       + CalSrf(MidCrd[1], MidCrd[5], MidCrd[4])
       + CalSrf(MidCrd[1], VerCrd[2], MidCrd[5])
       + CalSrf(MidCrd[4], MidCrd[5], VerCrd[3])
       + CalSrf(VerCrd[0], MidCrd[2], MidCrd[0])
       + CalSrf(MidCrd[2], MidCrd[1], MidCrd[0])
       + CalSrf(MidCrd[2], VerCrd[2], MidCrd[1])
       + CalSrf(MidCrd[1], VerCrd[1], MidCrd[0])
       + CalSrf(VerCrd[0], MidCrd[3], MidCrd[2])
       + CalSrf(MidCrd[2], MidCrd[3], MidCrd[5])
       + CalSrf(VerCrd[2], MidCrd[2], MidCrd[5])
       + CalSrf(VerCrd[3], MidCrd[5], MidCrd[3]);

   // Compute the P1 and P2 volumes
   volP1 = CalVol(VerCrd[0], VerCrd[1], VerCrd[2], VerCrd[3]);

   volP2 = CalVol(VerCrd[0], MidCrd[0], MidCrd[2], MidCrd[3])
         + CalVol(MidCrd[1], MidCrd[2], MidCrd[0], MidCrd[3])
         + CalVol(MidCrd[3], MidCrd[5], MidCrd[4], MidCrd[1])
         + CalVol(MidCrd[3], MidCrd[0], MidCrd[1], MidCrd[4])
         + CalVol(MidCrd[3], MidCrd[1], MidCrd[2], MidCrd[5])
         + CalVol(MidCrd[3], MidCrd[4], MidCrd[5], VerCrd[3])
         + CalVol(MidCrd[2], MidCrd[1], VerCrd[2], MidCrd[5])
         + CalVol(MidCrd[0], VerCrd[1], MidCrd[1], MidCrd[4]);

   // Convert mid points to control points
   MidCrd[0] = 2.f * MidCrd[0] -0.5f * (VerCrd[0] + VerCrd[1]);
   MidCrd[1] = 2.f * MidCrd[1] -0.5f * (VerCrd[1] + VerCrd[2]);
   MidCrd[2] = 2.f * MidCrd[2] -0.5f * (VerCrd[2] + VerCrd[0]);
   MidCrd[3] = 2.f * MidCrd[3] -0.5f * (VerCrd[0] + VerCrd[3]);
   MidCrd[4] = 2.f * MidCrd[4] -0.5f * (VerCrd[1] + VerCrd[3]);
   MidCrd[5] = 2.f * MidCrd[5] -0.5f * (VerCrd[2] + VerCrd[3]);

   // Set the 3 sets of four vectors per direction
   u[0] = MidCrd[4] - MidCrd[3];
   u[1] = MidCrd[1] - MidCrd[2];
   u[2] = VerCrd[1] - MidCrd[0];
   u[3] = MidCrd[0] - VerCrd[0];

   v[0] = MidCrd[5] - MidCrd[3];
   v[1] = VerCrd[2] - MidCrd[2];
   v[2] = MidCrd[1] - MidCrd[0];
   v[3] = MidCrd[2] - VerCrd[0];

   w[0] = VerCrd[3] - MidCrd[3];
   w[1] = MidCrd[5] - MidCrd[2];
   w[2] = MidCrd[4] - MidCrd[0];
   w[3] = MidCrd[3] - VerCrd[0];

   // Compute the 64 sub volumes
   n = cross(u[0], v[0]);
   s[ 0] = dot(n, w[0]);
   s[ 1] = dot(n, w[1]);
   s[ 2] = dot(n, w[2]);
   s[ 3] = dot(n, w[3]);

   n = cross(u[0], v[1]);
   s[ 4] = dot(n, w[0]);
   s[ 5] = dot(n, w[1]);
   s[ 6] = dot(n, w[2]);
   s[ 7] = dot(n, w[3]);

   n = cross(u[0], v[2]);
   s[ 8] = dot(n, w[0]);
   s[ 9] = dot(n, w[1]);
   s[10] = dot(n, w[2]);
   s[11] = dot(n, w[3]);

   n = cross(u[0], v[3]);
   s[12] = dot(n, w[0]);
   s[13] = dot(n, w[1]);
   s[14] = dot(n, w[2]);
   s[15] = dot(n, w[3]);

   n = cross(u[1], v[0]);
   s[16] = dot(n, w[0]);
   s[17] = dot(n, w[1]);
   s[18] = dot(n, w[2]);
   s[19] = dot(n, w[3]);

   n = cross(u[1], v[1]);
   s[20] = dot(n, w[0]);
   s[21] = dot(n, w[1]);
   s[22] = dot(n, w[2]);
   s[23] = dot(n, w[3]);

   n = cross(u[1], v[2]);
   s[24] = dot(n, w[0]);
   s[25] = dot(n, w[1]);
   s[26] = dot(n, w[2]);
   s[27] = dot(n, w[3]);

   n = cross(u[1], v[3]);
   s[28] = dot(n, w[0]);
   s[29] = dot(n, w[1]);
   s[30] = dot(n, w[2]);
   s[31] = dot(n, w[3]);

   n = cross(u[2], v[0]);
   s[32] = dot(n, w[0]);
   s[33] = dot(n, w[1]);
   s[34] = dot(n, w[2]);
   s[35] = dot(n, w[3]);

   n = cross(u[2], v[1]);
   s[36] = dot(n, w[0]);
   s[37] = dot(n, w[1]);
   s[38] = dot(n, w[2]);
   s[39] = dot(n, w[3]);

   n = cross(u[2], v[2]);
   s[40] = dot(n, w[0]);
   s[41] = dot(n, w[1]);
   s[42] = dot(n, w[2]);
   s[43] = dot(n, w[3]);

   n = cross(u[2], v[3]);
   s[44] = dot(n, w[0]);
   s[45] = dot(n, w[1]);
   s[46] = dot(n, w[2]);
   s[47] = dot(n, w[3]);

   n = cross(u[3], v[0]);
   s[48] = dot(n, w[0]);
   s[49] = dot(n, w[1]);
   s[50] = dot(n, w[2]);
   s[51] = dot(n, w[3]);

   n = cross(u[3], v[1]);
   s[52] = dot(n, w[0]);
   s[53] = dot(n, w[1]);
   s[54] = dot(n, w[2]);
   s[55] = dot(n, w[3]);

   n = cross(u[3], v[2]);
   s[56] = dot(n, w[0]);
   s[57] = dot(n, w[1]);
   s[58] = dot(n, w[2]);
   s[59] = dot(n, w[3]);

   n = cross(u[3], v[3]);
   s[60] = dot(n, w[0]);
   s[61] = dot(n, w[1]);
   s[62] = dot(n, w[2]);
   s[63] = dot(n, w[3]);

   // Compute the 20 coefficients and get their min and max values
   N = 8.f * s[ 0];
   MinJac = MaxJac = N;

   N = 8.f * s[21];
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = 8.f * s[42];
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = 8.f * s[63];
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[ 1] + s[ 4] + s[16]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[ 5] + s[17] + s[20]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[ 2] + s[ 8] + s[32]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[22] + s[25] + s[37]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[10] + s[34] + s[40]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[26] + s[38] + s[41]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[ 3] + s[12] + s[48]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[23] + s[29] + s[53]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[43] + s[46] + s[58]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[15] + s[51] + s[60]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[31] + s[55] + s[61]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (8.f/3.f) * (s[47] + s[59] + s[62]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (4.f/3.f) * (s[18] + s[24] + s[33] + s[36] + s[ 6] + s[ 9]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (4.f/3.f) * (s[13] + s[19] + s[28] + s[49] + s[52] + s[ 7]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (4.f/3.f) * (s[11] + s[14] + s[35] + s[44] + s[50] + s[56]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   N = (4.f/3.f) * (s[27] + s[30] + s[39] + s[45] + s[54] + s[57]);
   MinJac = min(MinJac, N);
   MaxJac = max(MaxJac, N);

   // Compute the quality criterion
   qal = (MinJac / MaxJac)
         * ((2.f * sqrt(6.f) * volP2) / (MaxLen * srf))
         * (min(volP1, volP2) / max(volP1, volP2));
}
