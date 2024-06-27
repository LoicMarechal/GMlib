#ifdef REAL32
#define fpn    float
#define fpn2   float2
#define fpn4   float4
#define fpn8   float8
#define fpn16  float16
#else
#define fpn    double
#define fpn2   double2
#define fpn4   double4
#define fpn8   double8
#define fpn16  double16
#endif


#if BLKSIZ == 4

fpn4 MulMatVec(fpn16 a, fpn4 x)
{
   fpn4 b;

   b.s0 = a.s0 * x.s0 + a.s1 * x.s1 + a.s2 * x.s2 + a.s3 * x.s3;
   b.s1 = a.s4 * x.s0 + a.s5 * x.s1 + a.s6 * x.s2 + a.s7 * x.s3;
   b.s2 = a.s8 * x.s0 + a.s9 * x.s1 + a.sa * x.s2 + a.sb * x.s3;
   b.s3 = a.sc * x.s0 + a.sd * x.s1 + a.se * x.s2 + a.sf * x.s3;

   return(b);
}

__kernel void MulMatVecSlc16( __global int   *D,
                              __global int16 *C,
                              __global fpn16 (*A)[16],
                              __global fpn4  *B,
                              __global fpn4  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn4  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l];

              b  = MulMatVec(A[l][ 0], X[ c.s0 ]);
   if(d >  1) b += MulMatVec(A[l][ 1], X[ c.s1 ]);
   if(d >  2) b += MulMatVec(A[l][ 2], X[ c.s2 ]);
   if(d >  3) b += MulMatVec(A[l][ 3], X[ c.s3 ]);
   if(d >  4) b += MulMatVec(A[l][ 4], X[ c.s4 ]);
   if(d >  5) b += MulMatVec(A[l][ 5], X[ c.s5 ]);
   if(d >  6) b += MulMatVec(A[l][ 6], X[ c.s6 ]);
   if(d >  7) b += MulMatVec(A[l][ 7], X[ c.s7 ]);
   if(d >  8) b += MulMatVec(A[l][ 8], X[ c.s8 ]);
   if(d >  9) b += MulMatVec(A[l][ 9], X[ c.s9 ]);
   if(d > 10) b += MulMatVec(A[l][10], X[ c.sa ]);
   if(d > 11) b += MulMatVec(A[l][11], X[ c.sb ]);
   if(d > 12) b += MulMatVec(A[l][12], X[ c.sc ]);
   if(d > 13) b += MulMatVec(A[l][13], X[ c.sd ]);
   if(d > 14) b += MulMatVec(A[l][14], X[ c.se ]);
   if(d > 15) b += MulMatVec(A[l][15], X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc32( __global int   *D,
                              __global int16 (*C)[2],
                              __global fpn16 (*A)[32],
                              __global fpn4  *B,
                              __global fpn4  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn4  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b = MulMatVec(A[l][ 0], X[ c.s0 ])
     + MulMatVec(A[l][ 1], X[ c.s1 ])
     + MulMatVec(A[l][ 2], X[ c.s2 ])
     + MulMatVec(A[l][ 3], X[ c.s3 ])
     + MulMatVec(A[l][ 4], X[ c.s4 ])
     + MulMatVec(A[l][ 5], X[ c.s5 ])
     + MulMatVec(A[l][ 6], X[ c.s6 ])
     + MulMatVec(A[l][ 7], X[ c.s7 ])
     + MulMatVec(A[l][ 8], X[ c.s8 ])
     + MulMatVec(A[l][ 9], X[ c.s9 ])
     + MulMatVec(A[l][10], X[ c.sa ])
     + MulMatVec(A[l][11], X[ c.sb ])
     + MulMatVec(A[l][12], X[ c.sc ])
     + MulMatVec(A[l][13], X[ c.sd ])
     + MulMatVec(A[l][14], X[ c.se ])
     + MulMatVec(A[l][15], X[ c.sf ]);

   c = C[l][1];

              b += MulMatVec(A[l][16], X[ c.s0 ]);
   if(d > 17) b += MulMatVec(A[l][17], X[ c.s1 ]);
   if(d > 18) b += MulMatVec(A[l][18], X[ c.s2 ]);
   if(d > 19) b += MulMatVec(A[l][19], X[ c.s3 ]);
   if(d > 20) b += MulMatVec(A[l][20], X[ c.s4 ]);
   if(d > 21) b += MulMatVec(A[l][21], X[ c.s5 ]);
   if(d > 22) b += MulMatVec(A[l][22], X[ c.s6 ]);
   if(d > 23) b += MulMatVec(A[l][23], X[ c.s7 ]);
   if(d > 24) b += MulMatVec(A[l][24], X[ c.s8 ]);
   if(d > 25) b += MulMatVec(A[l][25], X[ c.s9 ]);
   if(d > 26) b += MulMatVec(A[l][26], X[ c.sa ]);
   if(d > 27) b += MulMatVec(A[l][27], X[ c.sb ]);
   if(d > 28) b += MulMatVec(A[l][28], X[ c.sc ]);
   if(d > 29) b += MulMatVec(A[l][29], X[ c.sd ]);
   if(d > 30) b += MulMatVec(A[l][30], X[ c.se ]);
   if(d > 31) b += MulMatVec(A[l][31], X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc64( __global int   *D,
                              __global int16 (*C)[4],
                              __global fpn16 (*A)[64],
                              __global fpn4  *B,
                              __global fpn4  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn4  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b = MulMatVec(A[l][ 0], X[ c.s0 ])
     + MulMatVec(A[l][ 1], X[ c.s1 ])
     + MulMatVec(A[l][ 2], X[ c.s2 ])
     + MulMatVec(A[l][ 3], X[ c.s3 ])
     + MulMatVec(A[l][ 4], X[ c.s4 ])
     + MulMatVec(A[l][ 5], X[ c.s5 ])
     + MulMatVec(A[l][ 6], X[ c.s6 ])
     + MulMatVec(A[l][ 7], X[ c.s7 ])
     + MulMatVec(A[l][ 8], X[ c.s8 ])
     + MulMatVec(A[l][ 9], X[ c.s9 ])
     + MulMatVec(A[l][10], X[ c.sa ])
     + MulMatVec(A[l][11], X[ c.sb ])
     + MulMatVec(A[l][12], X[ c.sc ])
     + MulMatVec(A[l][13], X[ c.sd ])
     + MulMatVec(A[l][14], X[ c.se ])
     + MulMatVec(A[l][15], X[ c.sf ]);

   c = C[l][1];

   b += MulMatVec(A[l][16], X[ c.s0 ])
      + MulMatVec(A[l][17], X[ c.s1 ])
      + MulMatVec(A[l][18], X[ c.s2 ])
      + MulMatVec(A[l][19], X[ c.s3 ])
      + MulMatVec(A[l][20], X[ c.s4 ])
      + MulMatVec(A[l][21], X[ c.s5 ])
      + MulMatVec(A[l][22], X[ c.s6 ])
      + MulMatVec(A[l][23], X[ c.s7 ])
      + MulMatVec(A[l][24], X[ c.s8 ])
      + MulMatVec(A[l][25], X[ c.s9 ])
      + MulMatVec(A[l][26], X[ c.sa ])
      + MulMatVec(A[l][27], X[ c.sb ])
      + MulMatVec(A[l][28], X[ c.sc ])
      + MulMatVec(A[l][29], X[ c.sd ])
      + MulMatVec(A[l][30], X[ c.se ])
      + MulMatVec(A[l][31], X[ c.sf ]);

   c = C[l][2];

              b += MulMatVec(A[l][32], X[ c.s0 ]);
   if(d > 33) b += MulMatVec(A[l][33], X[ c.s1 ]);
   if(d > 34) b += MulMatVec(A[l][34], X[ c.s2 ]);
   if(d > 35) b += MulMatVec(A[l][35], X[ c.s3 ]);
   if(d > 36) b += MulMatVec(A[l][36], X[ c.s4 ]);
   if(d > 37) b += MulMatVec(A[l][37], X[ c.s5 ]);
   if(d > 38) b += MulMatVec(A[l][38], X[ c.s6 ]);
   if(d > 39) b += MulMatVec(A[l][39], X[ c.s7 ]);
   if(d > 40) b += MulMatVec(A[l][40], X[ c.s8 ]);
   if(d > 41) b += MulMatVec(A[l][41], X[ c.s9 ]);
   if(d > 42) b += MulMatVec(A[l][42], X[ c.sa ]);
   if(d > 43) b += MulMatVec(A[l][43], X[ c.sb ]);
   if(d > 44) b += MulMatVec(A[l][44], X[ c.sc ]);
   if(d > 45) b += MulMatVec(A[l][45], X[ c.sd ]);
   if(d > 46) b += MulMatVec(A[l][46], X[ c.se ]);
   if(d > 47) b += MulMatVec(A[l][47], X[ c.sf ]);

   c = C[l][3];

   if(d > 48) b += MulMatVec(A[l][48], X[ c.s0 ]);
   if(d > 49) b += MulMatVec(A[l][49], X[ c.s1 ]);
   if(d > 50) b += MulMatVec(A[l][50], X[ c.s2 ]);
   if(d > 51) b += MulMatVec(A[l][51], X[ c.s3 ]);
   if(d > 52) b += MulMatVec(A[l][52], X[ c.s4 ]);
   if(d > 53) b += MulMatVec(A[l][53], X[ c.s5 ]);
   if(d > 54) b += MulMatVec(A[l][54], X[ c.s6 ]);
   if(d > 55) b += MulMatVec(A[l][55], X[ c.s7 ]);
   if(d > 56) b += MulMatVec(A[l][56], X[ c.s8 ]);
   if(d > 57) b += MulMatVec(A[l][57], X[ c.s9 ]);
   if(d > 58) b += MulMatVec(A[l][58], X[ c.sa ]);
   if(d > 59) b += MulMatVec(A[l][59], X[ c.sb ]);
   if(d > 60) b += MulMatVec(A[l][60], X[ c.sc ]);
   if(d > 61) b += MulMatVec(A[l][61], X[ c.sd ]);
   if(d > 62) b += MulMatVec(A[l][62], X[ c.se ]);
   if(d > 63) b += MulMatVec(A[l][63], X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc128(__global int   *D,
                              __global int16 (*C)[8],
                              __global fpn16 (*A)[128],
                              __global fpn4  *B,
                              __global fpn4  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn4  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b = MulMatVec(A[l][ 0], X[ c.s0 ])
     + MulMatVec(A[l][ 1], X[ c.s1 ])
     + MulMatVec(A[l][ 2], X[ c.s2 ])
     + MulMatVec(A[l][ 3], X[ c.s3 ])
     + MulMatVec(A[l][ 4], X[ c.s4 ])
     + MulMatVec(A[l][ 5], X[ c.s5 ])
     + MulMatVec(A[l][ 6], X[ c.s6 ])
     + MulMatVec(A[l][ 7], X[ c.s7 ])
     + MulMatVec(A[l][ 8], X[ c.s8 ])
     + MulMatVec(A[l][ 9], X[ c.s9 ])
     + MulMatVec(A[l][10], X[ c.sa ])
     + MulMatVec(A[l][11], X[ c.sb ])
     + MulMatVec(A[l][12], X[ c.sc ])
     + MulMatVec(A[l][13], X[ c.sd ])
     + MulMatVec(A[l][14], X[ c.se ])
     + MulMatVec(A[l][15], X[ c.sf ]);

   c = C[l][1];

   b += MulMatVec(A[l][16], X[ c.s0 ])
      + MulMatVec(A[l][17], X[ c.s1 ])
      + MulMatVec(A[l][18], X[ c.s2 ])
      + MulMatVec(A[l][19], X[ c.s3 ])
      + MulMatVec(A[l][20], X[ c.s4 ])
      + MulMatVec(A[l][21], X[ c.s5 ])
      + MulMatVec(A[l][22], X[ c.s6 ])
      + MulMatVec(A[l][23], X[ c.s7 ])
      + MulMatVec(A[l][24], X[ c.s8 ])
      + MulMatVec(A[l][25], X[ c.s9 ])
      + MulMatVec(A[l][26], X[ c.sa ])
      + MulMatVec(A[l][27], X[ c.sb ])
      + MulMatVec(A[l][28], X[ c.sc ])
      + MulMatVec(A[l][29], X[ c.sd ])
      + MulMatVec(A[l][30], X[ c.se ])
      + MulMatVec(A[l][31], X[ c.sf ]);

   c = C[l][2];

   b += MulMatVec(A[l][32], X[ c.s0 ])
     +  MulMatVec(A[l][33], X[ c.s1 ])
     +  MulMatVec(A[l][34], X[ c.s2 ])
     +  MulMatVec(A[l][35], X[ c.s3 ])
     +  MulMatVec(A[l][36], X[ c.s4 ])
     +  MulMatVec(A[l][37], X[ c.s5 ])
     +  MulMatVec(A[l][38], X[ c.s6 ])
     +  MulMatVec(A[l][39], X[ c.s7 ])
     +  MulMatVec(A[l][40], X[ c.s8 ])
     +  MulMatVec(A[l][41], X[ c.s9 ])
     +  MulMatVec(A[l][42], X[ c.sa ])
     +  MulMatVec(A[l][43], X[ c.sb ])
     +  MulMatVec(A[l][44], X[ c.sc ])
     +  MulMatVec(A[l][45], X[ c.sd ])
     +  MulMatVec(A[l][46], X[ c.se ])
     +  MulMatVec(A[l][47], X[ c.sf ]);

   c = C[l][3];

   b += MulMatVec(A[l][48], X[ c.s0 ])
      + MulMatVec(A[l][49], X[ c.s1 ])
      + MulMatVec(A[l][50], X[ c.s2 ])
      + MulMatVec(A[l][51], X[ c.s3 ])
      + MulMatVec(A[l][52], X[ c.s4 ])
      + MulMatVec(A[l][53], X[ c.s5 ])
      + MulMatVec(A[l][54], X[ c.s6 ])
      + MulMatVec(A[l][55], X[ c.s7 ])
      + MulMatVec(A[l][56], X[ c.s8 ])
      + MulMatVec(A[l][57], X[ c.s9 ])
      + MulMatVec(A[l][58], X[ c.sa ])
      + MulMatVec(A[l][59], X[ c.sb ])
      + MulMatVec(A[l][60], X[ c.sc ])
      + MulMatVec(A[l][61], X[ c.sd ])
      + MulMatVec(A[l][62], X[ c.se ])
      + MulMatVec(A[l][63], X[ c.sf ]);

   c = C[l][4];

              b += MulMatVec(A[l][64], X[ c.s0 ]);
   if(d > 65) b += MulMatVec(A[l][65], X[ c.s1 ]);
   if(d > 66) b += MulMatVec(A[l][66], X[ c.s2 ]);
   if(d > 67) b += MulMatVec(A[l][67], X[ c.s3 ]);
   if(d > 68) b += MulMatVec(A[l][68], X[ c.s4 ]);
   if(d > 69) b += MulMatVec(A[l][69], X[ c.s5 ]);
   if(d > 70) b += MulMatVec(A[l][70], X[ c.s6 ]);
   if(d > 71) b += MulMatVec(A[l][71], X[ c.s7 ]);
   if(d > 72) b += MulMatVec(A[l][72], X[ c.s8 ]);
   if(d > 73) b += MulMatVec(A[l][73], X[ c.s9 ]);
   if(d > 74) b += MulMatVec(A[l][74], X[ c.sa ]);
   if(d > 75) b += MulMatVec(A[l][75], X[ c.sb ]);
   if(d > 76) b += MulMatVec(A[l][76], X[ c.sc ]);
   if(d > 77) b += MulMatVec(A[l][77], X[ c.sd ]);
   if(d > 78) b += MulMatVec(A[l][78], X[ c.se ]);
   if(d > 79) b += MulMatVec(A[l][79], X[ c.sf ]);

   c = C[l][5];

   if(d > 80) b += MulMatVec(A[l][80], X[ c.s0 ]);
   if(d > 81) b += MulMatVec(A[l][81], X[ c.s1 ]);
   if(d > 82) b += MulMatVec(A[l][82], X[ c.s2 ]);
   if(d > 83) b += MulMatVec(A[l][83], X[ c.s3 ]);
   if(d > 84) b += MulMatVec(A[l][84], X[ c.s4 ]);
   if(d > 85) b += MulMatVec(A[l][85], X[ c.s5 ]);
   if(d > 86) b += MulMatVec(A[l][86], X[ c.s6 ]);
   if(d > 87) b += MulMatVec(A[l][87], X[ c.s7 ]);
   if(d > 88) b += MulMatVec(A[l][88], X[ c.s8 ]);
   if(d > 89) b += MulMatVec(A[l][89], X[ c.s9 ]);
   if(d > 90) b += MulMatVec(A[l][90], X[ c.sa ]);
   if(d > 91) b += MulMatVec(A[l][91], X[ c.sb ]);
   if(d > 92) b += MulMatVec(A[l][92], X[ c.sc ]);
   if(d > 93) b += MulMatVec(A[l][93], X[ c.sd ]);
   if(d > 94) b += MulMatVec(A[l][94], X[ c.se ]);
   if(d > 95) b += MulMatVec(A[l][95], X[ c.sf ]);

   c = C[l][6];

   if(d >  96) b += MulMatVec(A[l][ 96], X[ c.s0 ]);
   if(d >  97) b += MulMatVec(A[l][ 97], X[ c.s1 ]);
   if(d >  98) b += MulMatVec(A[l][ 98], X[ c.s2 ]);
   if(d >  99) b += MulMatVec(A[l][ 99], X[ c.s3 ]);
   if(d > 100) b += MulMatVec(A[l][100], X[ c.s4 ]);
   if(d > 101) b += MulMatVec(A[l][101], X[ c.s5 ]);
   if(d > 102) b += MulMatVec(A[l][102], X[ c.s6 ]);
   if(d > 103) b += MulMatVec(A[l][103], X[ c.s7 ]);
   if(d > 104) b += MulMatVec(A[l][104], X[ c.s8 ]);
   if(d > 105) b += MulMatVec(A[l][105], X[ c.s9 ]);
   if(d > 106) b += MulMatVec(A[l][106], X[ c.sa ]);
   if(d > 107) b += MulMatVec(A[l][107], X[ c.sb ]);
   if(d > 108) b += MulMatVec(A[l][108], X[ c.sc ]);
   if(d > 109) b += MulMatVec(A[l][109], X[ c.sd ]);
   if(d > 110) b += MulMatVec(A[l][110], X[ c.se ]);
   if(d > 111) b += MulMatVec(A[l][111], X[ c.sf ]);

   c = C[l][7];

   if(d > 112) b += MulMatVec(A[l][112], X[ c.s0 ]);
   if(d > 113) b += MulMatVec(A[l][113], X[ c.s1 ]);
   if(d > 114) b += MulMatVec(A[l][114], X[ c.s2 ]);
   if(d > 115) b += MulMatVec(A[l][115], X[ c.s3 ]);
   if(d > 116) b += MulMatVec(A[l][116], X[ c.s4 ]);
   if(d > 117) b += MulMatVec(A[l][117], X[ c.s5 ]);
   if(d > 118) b += MulMatVec(A[l][118], X[ c.s6 ]);
   if(d > 119) b += MulMatVec(A[l][119], X[ c.s7 ]);
   if(d > 120) b += MulMatVec(A[l][120], X[ c.s8 ]);
   if(d > 121) b += MulMatVec(A[l][121], X[ c.s9 ]);
   if(d > 122) b += MulMatVec(A[l][122], X[ c.sa ]);
   if(d > 123) b += MulMatVec(A[l][123], X[ c.sb ]);
   if(d > 124) b += MulMatVec(A[l][124], X[ c.sc ]);
   if(d > 125) b += MulMatVec(A[l][125], X[ c.sd ]);
   if(d > 126) b += MulMatVec(A[l][126], X[ c.se ]);
   if(d > 127) b += MulMatVec(A[l][127], X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc256(__global int   *D,
                              __global int16 (*C)[16],
                              __global fpn16 (*A)[256],
                              __global fpn4  *B,
                              __global fpn4  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn4  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b = MulMatVec(A[l][ 0], X[ c.s0 ])
     + MulMatVec(A[l][ 1], X[ c.s1 ])
     + MulMatVec(A[l][ 2], X[ c.s2 ])
     + MulMatVec(A[l][ 3], X[ c.s3 ])
     + MulMatVec(A[l][ 4], X[ c.s4 ])
     + MulMatVec(A[l][ 5], X[ c.s5 ])
     + MulMatVec(A[l][ 6], X[ c.s6 ])
     + MulMatVec(A[l][ 7], X[ c.s7 ])
     + MulMatVec(A[l][ 8], X[ c.s8 ])
     + MulMatVec(A[l][ 9], X[ c.s9 ])
     + MulMatVec(A[l][10], X[ c.sa ])
     + MulMatVec(A[l][11], X[ c.sb ])
     + MulMatVec(A[l][12], X[ c.sc ])
     + MulMatVec(A[l][13], X[ c.sd ])
     + MulMatVec(A[l][14], X[ c.se ])
     + MulMatVec(A[l][15], X[ c.sf ]);

   c = C[l][1];

   b += MulMatVec(A[l][16], X[ c.s0 ])
      + MulMatVec(A[l][17], X[ c.s1 ])
      + MulMatVec(A[l][18], X[ c.s2 ])
      + MulMatVec(A[l][19], X[ c.s3 ])
      + MulMatVec(A[l][20], X[ c.s4 ])
      + MulMatVec(A[l][21], X[ c.s5 ])
      + MulMatVec(A[l][22], X[ c.s6 ])
      + MulMatVec(A[l][23], X[ c.s7 ])
      + MulMatVec(A[l][24], X[ c.s8 ])
      + MulMatVec(A[l][25], X[ c.s9 ])
      + MulMatVec(A[l][26], X[ c.sa ])
      + MulMatVec(A[l][27], X[ c.sb ])
      + MulMatVec(A[l][28], X[ c.sc ])
      + MulMatVec(A[l][29], X[ c.sd ])
      + MulMatVec(A[l][30], X[ c.se ])
      + MulMatVec(A[l][31], X[ c.sf ]);

   c = C[l][2];

   b += MulMatVec(A[l][32], X[ c.s0 ])
      + MulMatVec(A[l][33], X[ c.s1 ])
      + MulMatVec(A[l][34], X[ c.s2 ])
      + MulMatVec(A[l][35], X[ c.s3 ])
      + MulMatVec(A[l][36], X[ c.s4 ])
      + MulMatVec(A[l][37], X[ c.s5 ])
      + MulMatVec(A[l][38], X[ c.s6 ])
      + MulMatVec(A[l][39], X[ c.s7 ])
      + MulMatVec(A[l][40], X[ c.s8 ])
      + MulMatVec(A[l][41], X[ c.s9 ])
      + MulMatVec(A[l][42], X[ c.sa ])
      + MulMatVec(A[l][43], X[ c.sb ])
      + MulMatVec(A[l][44], X[ c.sc ])
      + MulMatVec(A[l][45], X[ c.sd ])
      + MulMatVec(A[l][46], X[ c.se ])
      + MulMatVec(A[l][47], X[ c.sf ]);

   c = C[l][3];

   b += MulMatVec(A[l][48], X[ c.s0 ])
      + MulMatVec(A[l][49], X[ c.s1 ])
      + MulMatVec(A[l][50], X[ c.s2 ])
      + MulMatVec(A[l][51], X[ c.s3 ])
      + MulMatVec(A[l][52], X[ c.s4 ])
      + MulMatVec(A[l][53], X[ c.s5 ])
      + MulMatVec(A[l][54], X[ c.s6 ])
      + MulMatVec(A[l][55], X[ c.s7 ])
      + MulMatVec(A[l][56], X[ c.s8 ])
      + MulMatVec(A[l][57], X[ c.s9 ])
      + MulMatVec(A[l][58], X[ c.sa ])
      + MulMatVec(A[l][59], X[ c.sb ])
      + MulMatVec(A[l][60], X[ c.sc ])
      + MulMatVec(A[l][61], X[ c.sd ])
      + MulMatVec(A[l][62], X[ c.se ])
      + MulMatVec(A[l][63], X[ c.sf ]);

   c = C[l][4];

   b += MulMatVec(A[l][64], X[ c.s0 ])
      + MulMatVec(A[l][65], X[ c.s1 ])
      + MulMatVec(A[l][66], X[ c.s2 ])
      + MulMatVec(A[l][67], X[ c.s3 ])
      + MulMatVec(A[l][68], X[ c.s4 ])
      + MulMatVec(A[l][69], X[ c.s5 ])
      + MulMatVec(A[l][70], X[ c.s6 ])
      + MulMatVec(A[l][71], X[ c.s7 ])
      + MulMatVec(A[l][72], X[ c.s8 ])
      + MulMatVec(A[l][73], X[ c.s9 ])
      + MulMatVec(A[l][74], X[ c.sa ])
      + MulMatVec(A[l][75], X[ c.sb ])
      + MulMatVec(A[l][76], X[ c.sc ])
      + MulMatVec(A[l][77], X[ c.sd ])
      + MulMatVec(A[l][78], X[ c.se ])
      + MulMatVec(A[l][79], X[ c.sf ]);

   c = C[l][5];

   b += MulMatVec(A[l][80], X[ c.s0 ])
      + MulMatVec(A[l][81], X[ c.s1 ])
      + MulMatVec(A[l][82], X[ c.s2 ])
      + MulMatVec(A[l][83], X[ c.s3 ])
      + MulMatVec(A[l][84], X[ c.s4 ])
      + MulMatVec(A[l][85], X[ c.s5 ])
      + MulMatVec(A[l][86], X[ c.s6 ])
      + MulMatVec(A[l][87], X[ c.s7 ])
      + MulMatVec(A[l][88], X[ c.s8 ])
      + MulMatVec(A[l][89], X[ c.s9 ])
      + MulMatVec(A[l][90], X[ c.sa ])
      + MulMatVec(A[l][91], X[ c.sb ])
      + MulMatVec(A[l][92], X[ c.sc ])
      + MulMatVec(A[l][93], X[ c.sd ])
      + MulMatVec(A[l][94], X[ c.se ])
      + MulMatVec(A[l][95], X[ c.sf ]);

   c = C[l][6];

   b += MulMatVec(A[l][ 96], X[ c.s0 ])
      + MulMatVec(A[l][ 97], X[ c.s1 ])
      + MulMatVec(A[l][ 98], X[ c.s2 ])
      + MulMatVec(A[l][ 99], X[ c.s3 ])
      + MulMatVec(A[l][100], X[ c.s4 ])
      + MulMatVec(A[l][101], X[ c.s5 ])
      + MulMatVec(A[l][102], X[ c.s6 ])
      + MulMatVec(A[l][103], X[ c.s7 ])
      + MulMatVec(A[l][104], X[ c.s8 ])
      + MulMatVec(A[l][105], X[ c.s9 ])
      + MulMatVec(A[l][106], X[ c.sa ])
      + MulMatVec(A[l][107], X[ c.sb ])
      + MulMatVec(A[l][108], X[ c.sc ])
      + MulMatVec(A[l][109], X[ c.sd ])
      + MulMatVec(A[l][110], X[ c.se ])
      + MulMatVec(A[l][111], X[ c.sf ]);

   c = C[l][7];

   b += MulMatVec(A[l][112], X[ c.s0 ])
      + MulMatVec(A[l][113], X[ c.s1 ])
      + MulMatVec(A[l][114], X[ c.s2 ])
      + MulMatVec(A[l][115], X[ c.s3 ])
      + MulMatVec(A[l][116], X[ c.s4 ])
      + MulMatVec(A[l][117], X[ c.s5 ])
      + MulMatVec(A[l][118], X[ c.s6 ])
      + MulMatVec(A[l][119], X[ c.s7 ])
      + MulMatVec(A[l][120], X[ c.s8 ])
      + MulMatVec(A[l][121], X[ c.s9 ])
      + MulMatVec(A[l][122], X[ c.sa ])
      + MulMatVec(A[l][123], X[ c.sb ])
      + MulMatVec(A[l][124], X[ c.sc ])
      + MulMatVec(A[l][125], X[ c.sd ])
      + MulMatVec(A[l][126], X[ c.se ])
      + MulMatVec(A[l][127], X[ c.sf ]);

   c = C[l][8];

               b += MulMatVec(A[l][128], X[ c.s0 ]);
   if(d > 129) b += MulMatVec(A[l][129], X[ c.s1 ]);
   if(d > 130) b += MulMatVec(A[l][130], X[ c.s2 ]);
   if(d > 131) b += MulMatVec(A[l][131], X[ c.s3 ]);
   if(d > 132) b += MulMatVec(A[l][132], X[ c.s4 ]);
   if(d > 133) b += MulMatVec(A[l][133], X[ c.s5 ]);
   if(d > 134) b += MulMatVec(A[l][134], X[ c.s6 ]);
   if(d > 135) b += MulMatVec(A[l][135], X[ c.s7 ]);
   if(d > 136) b += MulMatVec(A[l][136], X[ c.s8 ]);
   if(d > 137) b += MulMatVec(A[l][137], X[ c.s9 ]);
   if(d > 138) b += MulMatVec(A[l][138], X[ c.sa ]);
   if(d > 139) b += MulMatVec(A[l][139], X[ c.sb ]);
   if(d > 140) b += MulMatVec(A[l][140], X[ c.sc ]);
   if(d > 141) b += MulMatVec(A[l][141], X[ c.sd ]);
   if(d > 142) b += MulMatVec(A[l][142], X[ c.se ]);
   if(d > 143) b += MulMatVec(A[l][143], X[ c.sf ]);

   c = C[l][9];

   if(d > 144) b += MulMatVec(A[l][144], X[ c.s0 ]);
   if(d > 145) b += MulMatVec(A[l][145], X[ c.s1 ]);
   if(d > 146) b += MulMatVec(A[l][146], X[ c.s2 ]);
   if(d > 147) b += MulMatVec(A[l][147], X[ c.s3 ]);
   if(d > 148) b += MulMatVec(A[l][148], X[ c.s4 ]);
   if(d > 149) b += MulMatVec(A[l][149], X[ c.s5 ]);
   if(d > 150) b += MulMatVec(A[l][150], X[ c.s6 ]);
   if(d > 151) b += MulMatVec(A[l][151], X[ c.s7 ]);
   if(d > 152) b += MulMatVec(A[l][152], X[ c.s8 ]);
   if(d > 153) b += MulMatVec(A[l][153], X[ c.s9 ]);
   if(d > 154) b += MulMatVec(A[l][154], X[ c.sa ]);
   if(d > 155) b += MulMatVec(A[l][155], X[ c.sb ]);
   if(d > 156) b += MulMatVec(A[l][156], X[ c.sc ]);
   if(d > 157) b += MulMatVec(A[l][157], X[ c.sd ]);
   if(d > 158) b += MulMatVec(A[l][158], X[ c.se ]);
   if(d > 159) b += MulMatVec(A[l][159], X[ c.sf ]);

   c = C[l][10];

   if(d > 160) b += MulMatVec(A[l][160], X[ c.s0 ]);
   if(d > 161) b += MulMatVec(A[l][161], X[ c.s1 ]);
   if(d > 162) b += MulMatVec(A[l][162], X[ c.s2 ]);
   if(d > 163) b += MulMatVec(A[l][163], X[ c.s3 ]);
   if(d > 164) b += MulMatVec(A[l][164], X[ c.s4 ]);
   if(d > 165) b += MulMatVec(A[l][165], X[ c.s5 ]);
   if(d > 166) b += MulMatVec(A[l][166], X[ c.s6 ]);
   if(d > 167) b += MulMatVec(A[l][167], X[ c.s7 ]);
   if(d > 168) b += MulMatVec(A[l][168], X[ c.s8 ]);
   if(d > 169) b += MulMatVec(A[l][169], X[ c.s9 ]);
   if(d > 170) b += MulMatVec(A[l][170], X[ c.sa ]);
   if(d > 171) b += MulMatVec(A[l][171], X[ c.sb ]);
   if(d > 172) b += MulMatVec(A[l][172], X[ c.sc ]);
   if(d > 173) b += MulMatVec(A[l][173], X[ c.sd ]);
   if(d > 174) b += MulMatVec(A[l][174], X[ c.se ]);
   if(d > 175) b += MulMatVec(A[l][175], X[ c.sf ]);

   c = C[l][11];

   if(d > 176) b += MulMatVec(A[l][176], X[ c.s0 ]);
   if(d > 177) b += MulMatVec(A[l][177], X[ c.s1 ]);
   if(d > 178) b += MulMatVec(A[l][178], X[ c.s2 ]);
   if(d > 179) b += MulMatVec(A[l][179], X[ c.s3 ]);
   if(d > 180) b += MulMatVec(A[l][180], X[ c.s4 ]);
   if(d > 181) b += MulMatVec(A[l][181], X[ c.s5 ]);
   if(d > 182) b += MulMatVec(A[l][182], X[ c.s6 ]);
   if(d > 183) b += MulMatVec(A[l][183], X[ c.s7 ]);
   if(d > 184) b += MulMatVec(A[l][184], X[ c.s8 ]);
   if(d > 185) b += MulMatVec(A[l][185], X[ c.s9 ]);
   if(d > 186) b += MulMatVec(A[l][186], X[ c.sa ]);
   if(d > 187) b += MulMatVec(A[l][187], X[ c.sb ]);
   if(d > 188) b += MulMatVec(A[l][188], X[ c.sc ]);
   if(d > 189) b += MulMatVec(A[l][189], X[ c.sd ]);
   if(d > 190) b += MulMatVec(A[l][190], X[ c.se ]);
   if(d > 191) b += MulMatVec(A[l][191], X[ c.sf ]);

   c = C[l][12];

   if(d > 192) b += MulMatVec(A[l][192], X[ c.s0 ]);
   if(d > 193) b += MulMatVec(A[l][193], X[ c.s1 ]);
   if(d > 194) b += MulMatVec(A[l][194], X[ c.s2 ]);
   if(d > 195) b += MulMatVec(A[l][195], X[ c.s3 ]);
   if(d > 196) b += MulMatVec(A[l][196], X[ c.s4 ]);
   if(d > 197) b += MulMatVec(A[l][197], X[ c.s5 ]);
   if(d > 198) b += MulMatVec(A[l][198], X[ c.s6 ]);
   if(d > 199) b += MulMatVec(A[l][199], X[ c.s7 ]);
   if(d > 200) b += MulMatVec(A[l][200], X[ c.s8 ]);
   if(d > 201) b += MulMatVec(A[l][201], X[ c.s9 ]);
   if(d > 202) b += MulMatVec(A[l][202], X[ c.sa ]);
   if(d > 203) b += MulMatVec(A[l][203], X[ c.sb ]);
   if(d > 204) b += MulMatVec(A[l][204], X[ c.sc ]);
   if(d > 205) b += MulMatVec(A[l][205], X[ c.sd ]);
   if(d > 206) b += MulMatVec(A[l][206], X[ c.se ]);
   if(d > 207) b += MulMatVec(A[l][207], X[ c.sf ]);

   c = C[l][13];

   if(d > 208) b += MulMatVec(A[l][208], X[ c.s0 ]);
   if(d > 209) b += MulMatVec(A[l][209], X[ c.s1 ]);
   if(d > 210) b += MulMatVec(A[l][210], X[ c.s2 ]);
   if(d > 211) b += MulMatVec(A[l][211], X[ c.s3 ]);
   if(d > 212) b += MulMatVec(A[l][212], X[ c.s4 ]);
   if(d > 213) b += MulMatVec(A[l][213], X[ c.s5 ]);
   if(d > 214) b += MulMatVec(A[l][214], X[ c.s6 ]);
   if(d > 215) b += MulMatVec(A[l][215], X[ c.s7 ]);
   if(d > 216) b += MulMatVec(A[l][216], X[ c.s8 ]);
   if(d > 217) b += MulMatVec(A[l][217], X[ c.s9 ]);
   if(d > 218) b += MulMatVec(A[l][218], X[ c.sa ]);
   if(d > 219) b += MulMatVec(A[l][219], X[ c.sb ]);
   if(d > 220) b += MulMatVec(A[l][220], X[ c.sc ]);
   if(d > 221) b += MulMatVec(A[l][221], X[ c.sd ]);
   if(d > 222) b += MulMatVec(A[l][222], X[ c.se ]);
   if(d > 223) b += MulMatVec(A[l][223], X[ c.sf ]);

   c = C[l][14];

   if(d > 224) b += MulMatVec(A[l][224], X[ c.s0 ]);
   if(d > 225) b += MulMatVec(A[l][225], X[ c.s1 ]);
   if(d > 226) b += MulMatVec(A[l][226], X[ c.s2 ]);
   if(d > 227) b += MulMatVec(A[l][227], X[ c.s3 ]);
   if(d > 228) b += MulMatVec(A[l][228], X[ c.s4 ]);
   if(d > 229) b += MulMatVec(A[l][229], X[ c.s5 ]);
   if(d > 230) b += MulMatVec(A[l][230], X[ c.s6 ]);
   if(d > 231) b += MulMatVec(A[l][231], X[ c.s7 ]);
   if(d > 232) b += MulMatVec(A[l][232], X[ c.s8 ]);
   if(d > 233) b += MulMatVec(A[l][233], X[ c.s9 ]);
   if(d > 234) b += MulMatVec(A[l][234], X[ c.sa ]);
   if(d > 235) b += MulMatVec(A[l][235], X[ c.sb ]);
   if(d > 236) b += MulMatVec(A[l][236], X[ c.sc ]);
   if(d > 237) b += MulMatVec(A[l][237], X[ c.sd ]);
   if(d > 238) b += MulMatVec(A[l][238], X[ c.se ]);
   if(d > 239) b += MulMatVec(A[l][239], X[ c.sf ]);

   c = C[l][15];

   if(d > 240) b += MulMatVec(A[l][240], X[ c.s0 ]);
   if(d > 241) b += MulMatVec(A[l][241], X[ c.s1 ]);
   if(d > 242) b += MulMatVec(A[l][242], X[ c.s2 ]);
   if(d > 243) b += MulMatVec(A[l][243], X[ c.s3 ]);
   if(d > 244) b += MulMatVec(A[l][244], X[ c.s4 ]);
   if(d > 245) b += MulMatVec(A[l][245], X[ c.s5 ]);
   if(d > 246) b += MulMatVec(A[l][246], X[ c.s6 ]);
   if(d > 247) b += MulMatVec(A[l][247], X[ c.s7 ]);
   if(d > 248) b += MulMatVec(A[l][248], X[ c.s8 ]);
   if(d > 249) b += MulMatVec(A[l][249], X[ c.s9 ]);
   if(d > 250) b += MulMatVec(A[l][250], X[ c.sa ]);
   if(d > 251) b += MulMatVec(A[l][251], X[ c.sb ]);
   if(d > 252) b += MulMatVec(A[l][252], X[ c.sc ]);
   if(d > 253) b += MulMatVec(A[l][253], X[ c.sd ]);
   if(d > 254) b += MulMatVec(A[l][254], X[ c.se ]);
   if(d > 255) b += MulMatVec(A[l][255], X[ c.sf ]);

   B[l+N.s1] = b;
}

#elif BLKSIZ == 5

fpn8 MulMatVec01(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s0 * x.s0 + ma.s1 * x.s1 + ma.s2 * x.s2 + ma.s3 * x.s3 + ma.s4 * x.s4;
   b.s1 = ma.s5 * x.s0 + ma.s6 * x.s1 + ma.s7 * x.s2 + ma.s8 * x.s3 + ma.s9 * x.s4;
   b.s2 = ma.sa * x.s0 + ma.sb * x.s1 + ma.sc * x.s2 + ma.sd * x.s3 + ma.se * x.s4;
   b.s3 = ma.sf * x.s0 + mb.s0 * x.s1 + mb.s1 * x.s2 + mb.s2 * x.s3 + mb.s3 * x.s4;
   b.s4 = mb.s4 * x.s0 + mb.s5 * x.s1 + mb.s6 * x.s2 + mb.s7 * x.s3 + mb.s8 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec02(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s9 * x.s0 + ma.sa * x.s1 + ma.sb * x.s2 + ma.sc * x.s3 + ma.sd * x.s4;
   b.s1 = ma.se * x.s0 + ma.sf * x.s1 + mb.s0 * x.s2 + mb.s1 * x.s3 + mb.s2 * x.s4;
   b.s2 = mb.s3 * x.s0 + mb.s4 * x.s1 + mb.s5 * x.s2 + mb.s6 * x.s3 + mb.s7 * x.s4;
   b.s3 = mb.s8 * x.s0 + mb.s9 * x.s1 + mb.sa * x.s2 + mb.sb * x.s3 + mb.sc * x.s4;
   b.s4 = mb.sd * x.s0 + mb.se * x.s1 + mb.sf * x.s2 + mc.s0 * x.s3 + mc.s1 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec03(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s2 * x.s0 + ma.s3 * x.s1 + ma.s4 * x.s2 + ma.s5 * x.s3 + ma.s6 * x.s4;
   b.s1 = ma.s7 * x.s0 + ma.s8 * x.s1 + ma.s9 * x.s2 + ma.sa * x.s3 + ma.sb * x.s4;
   b.s2 = ma.sc * x.s0 + ma.sd * x.s1 + ma.se * x.s2 + ma.sf * x.s3 + mb.s0 * x.s4;
   b.s3 = mb.s1 * x.s0 + mb.s2 * x.s1 + mb.s3 * x.s2 + mb.s4 * x.s3 + mb.s5 * x.s4;
   b.s4 = mb.s6 * x.s0 + mb.s7 * x.s1 + mb.s8 * x.s2 + mb.s9 * x.s3 + mb.sa * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec04(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.sb * x.s0 + ma.sc * x.s1 + ma.sd * x.s2 + ma.se * x.s3 + ma.sf * x.s4;
   b.s1 = mb.s0 * x.s0 + mb.s1 * x.s1 + mb.s2 * x.s2 + mb.s3 * x.s3 + mb.s4 * x.s4;
   b.s2 = mb.s5 * x.s0 + mb.s6 * x.s1 + mb.s7 * x.s2 + mb.s9 * x.s3 + mb.s9 * x.s4;
   b.s3 = mb.sa * x.s0 + mb.sb * x.s1 + mb.sc * x.s2 + mb.sd * x.s3 + mb.se * x.s4;
   b.s4 = mb.sf * x.s0 + mc.s0 * x.s1 + mc.s1 * x.s2 + mc.s2 * x.s3 + mc.s3 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec05(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s4 * x.s0 + ma.s5 * x.s1 + ma.s6 * x.s2 + ma.s7 * x.s3 + ma.s8 * x.s4;
   b.s1 = ma.s9 * x.s0 + ma.sa * x.s1 + ma.sb * x.s2 + ma.sc * x.s3 + ma.sd * x.s4;
   b.s2 = ma.se * x.s0 + ma.sf * x.s1 + mb.s0 * x.s2 + mb.s1 * x.s3 + mb.s2 * x.s4;
   b.s3 = mb.s3 * x.s0 + mb.s4 * x.s1 + mb.s5 * x.s2 + mb.s6 * x.s3 + mb.s7 * x.s4;
   b.s4 = mb.s8 * x.s0 + mb.s9 * x.s1 + mb.sa * x.s2 + mb.sb * x.s3 + mb.sc * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec06(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.sd * x.s0 + ma.se * x.s1 + ma.sf * x.s2 + mb.s0 * x.s3 + mb.s1 * x.s4;
   b.s1 = mb.s2 * x.s0 + mb.s3 * x.s1 + mb.s4 * x.s2 + mb.s5 * x.s3 + mb.s6 * x.s4;
   b.s2 = mb.s7 * x.s0 + mb.s8 * x.s1 + mb.s9 * x.s2 + mb.sa * x.s3 + mb.sb * x.s4;
   b.s3 = mb.sc * x.s0 + mb.sd * x.s1 + mb.se * x.s2 + mb.sf * x.s3 + mc.s0 * x.s4;
   b.s4 = mc.s1 * x.s0 + mc.s2 * x.s1 + mc.s3 * x.s2 + mc.s4 * x.s3 + mc.s5 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec07(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s6 * x.s0 + ma.s7 * x.s1 + ma.s8 * x.s2 + ma.s9 * x.s3 + ma.sa * x.s4;
   b.s1 = ma.sb * x.s0 + ma.sc * x.s1 + ma.sd * x.s2 + ma.se * x.s3 + ma.sf * x.s4;
   b.s2 = mb.s0 * x.s0 + mb.s1 * x.s1 + mb.s2 * x.s2 + mb.s3 * x.s3 + mb.s4 * x.s4;
   b.s3 = mb.s5 * x.s0 + mb.s6 * x.s1 + mb.s7 * x.s2 + mb.s8 * x.s3 + mb.s9 * x.s4;
   b.s4 = mb.sa * x.s0 + mb.sb * x.s1 + mb.sc * x.s2 + mb.sd * x.s3 + mb.se * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec08(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.sf * x.s0 + mb.s0 * x.s1 + mb.s1 * x.s2 + mb.s2 * x.s3 + mb.s3 * x.s4;
   b.s1 = mb.s4 * x.s0 + mb.s5 * x.s1 + mb.s6 * x.s2 + mb.s7 * x.s3 + mb.s8 * x.s4;
   b.s2 = mb.s9 * x.s0 + mb.sa * x.s1 + mb.sb * x.s2 + mb.sc * x.s3 + mb.sd * x.s4;
   b.s3 = mb.se * x.s0 + mb.sf * x.s1 + mc.s0 * x.s2 + mc.s1 * x.s3 + mc.s2 * x.s4;
   b.s4 = mc.s3 * x.s0 + mc.s4 * x.s1 + mc.s5 * x.s2 + mc.s6 * x.s3 + mc.s7 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec09(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s8 * x.s0 + ma.s9 * x.s1 + ma.sa * x.s2 + ma.sb * x.s3 + ma.sc * x.s4;
   b.s1 = ma.sd * x.s0 + ma.se * x.s1 + ma.sf * x.s2 + mb.s0 * x.s3 + mb.s1 * x.s4;
   b.s2 = mb.s2 * x.s0 + mb.s3 * x.s1 + mb.s4 * x.s2 + mb.s5 * x.s3 + mb.s6 * x.s4;
   b.s3 = mb.s7 * x.s0 + mb.s8 * x.s1 + mb.s9 * x.s2 + mb.sa * x.s3 + mb.sb * x.s4;
   b.s4 = mb.sc * x.s0 + mb.sd * x.s1 + mb.se * x.s2 + mb.sf * x.s3 + mc.s0 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec10(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s1 * x.s0 + ma.s2 * x.s1 + ma.s3 * x.s2 + ma.s4 * x.s3 + ma.s5 * x.s4;
   b.s1 = ma.s6 * x.s0 + ma.s7 * x.s1 + ma.s8 * x.s2 + ma.s9 * x.s3 + ma.sa * x.s4;
   b.s2 = ma.sb * x.s0 + ma.sc * x.s1 + ma.sd * x.s2 + ma.se * x.s3 + ma.sf * x.s4;
   b.s3 = mb.s0 * x.s0 + mb.s1 * x.s1 + mb.s2 * x.s2 + mb.s3 * x.s3 + mb.s4 * x.s4;
   b.s4 = mb.s5 * x.s0 + mb.s6 * x.s1 + mb.s7 * x.s2 + mb.s8 * x.s3 + mb.s9 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec11(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.sa * x.s0 + ma.sb * x.s1 + ma.sc * x.s2 + ma.sd * x.s3 + ma.se * x.s4;
   b.s1 = ma.sf * x.s0 + mb.s0 * x.s1 + mb.s1 * x.s2 + mb.s2 * x.s3 + mb.s3 * x.s4;
   b.s2 = mb.s4 * x.s0 + mb.s5 * x.s1 + mb.s6 * x.s2 + mb.s7 * x.s3 + mb.s8 * x.s4;
   b.s3 = mb.s9 * x.s0 + mb.sa * x.s1 + mb.sb * x.s2 + mb.sc * x.s3 + mb.sd * x.s4;
   b.s4 = mb.se * x.s0 + mb.sf * x.s1 + mc.s0 * x.s2 + mc.s1 * x.s3 + mc.s2 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec12(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s3 * x.s0 + ma.s4 * x.s1 + ma.s5 * x.s2 + ma.s6 * x.s3 + ma.s7 * x.s4;
   b.s1 = ma.s8 * x.s0 + ma.s9 * x.s1 + ma.sa * x.s2 + ma.sb * x.s3 + ma.sc * x.s4;
   b.s2 = ma.sd * x.s0 + ma.se * x.s1 + ma.sf * x.s2 + mb.s0 * x.s3 + mb.s1 * x.s4;
   b.s3 = mb.s2 * x.s0 + mb.s3 * x.s1 + mb.s4 * x.s2 + mb.s5 * x.s3 + mb.s6 * x.s4;
   b.s4 = mb.s7 * x.s0 + mb.s8 * x.s1 + mb.s9 * x.s2 + mb.sa * x.s3 + mb.sb * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec13(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.sc * x.s0 + ma.sd * x.s1 + ma.se * x.s2 + ma.sf * x.s3 + mb.s0 * x.s4;
   b.s1 = mb.s1 * x.s0 + mb.s2 * x.s1 + mb.s3 * x.s2 + mb.s4 * x.s3 + mb.s5 * x.s4;
   b.s2 = mb.s6 * x.s0 + mb.s7 * x.s1 + mb.s8 * x.s2 + mb.s9 * x.s3 + mb.sa * x.s4;
   b.s3 = mb.sb * x.s0 + mb.sc * x.s1 + mb.sd * x.s2 + mb.se * x.s3 + mb.sf * x.s4;
   b.s4 = mc.s0 * x.s0 + mc.s1 * x.s1 + mc.s2 * x.s2 + mc.s3 * x.s3 + mc.s4 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec14(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s5 * x.s0 + ma.s6 * x.s1 + ma.s7 * x.s2 + ma.s8 * x.s3 + ma.s9 * x.s4;
   b.s1 = ma.sa * x.s0 + ma.sb * x.s1 + ma.sc * x.s2 + ma.sd * x.s3 + ma.se * x.s4;
   b.s2 = ma.sf * x.s0 + mb.s0 * x.s1 + mb.s1 * x.s2 + mb.s2 * x.s3 + mb.s3 * x.s4;
   b.s3 = mb.s4 * x.s0 + mb.s5 * x.s1 + mb.s6 * x.s2 + mb.s7 * x.s3 + mb.s8 * x.s4;
   b.s4 = mb.s9 * x.s0 + mb.sa * x.s1 + mb.sb * x.s2 + mb.sc * x.s3 + mb.sd * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec15(fpn16 ma, fpn16 mb, fpn16 mc, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.se * x.s0 + ma.sf * x.s1 + mb.s0 * x.s2 + mb.s1 * x.s3 + mb.s2 * x.s4;
   b.s1 = mb.s3 * x.s0 + mb.s4 * x.s1 + mb.s5 * x.s2 + mb.s6 * x.s3 + mb.s7 * x.s4;
   b.s2 = mb.s8 * x.s0 + mb.s9 * x.s1 + mb.sa * x.s2 + mb.sb * x.s3 + mb.sc * x.s4;
   b.s3 = mb.sd * x.s0 + mb.se * x.s1 + mb.sf * x.s2 + mc.s0 * x.s3 + mc.s1 * x.s4;
   b.s4 = mc.s2 * x.s0 + mc.s3 * x.s1 + mc.s4 * x.s2 + mc.s5 * x.s3 + mc.s6 * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

fpn8 MulMatVec16(fpn16 ma, fpn16 mb, fpn8 x)
{
   fpn8 b;

   b.s0 = ma.s7 * x.s0 + ma.s8 * x.s1 + ma.s9 * x.s2 + ma.sa * x.s3 + ma.sb * x.s4;
   b.s1 = ma.sc * x.s0 + ma.sd * x.s1 + ma.se * x.s2 + ma.sf * x.s3 + mb.s0 * x.s4;
   b.s2 = mb.s1 * x.s0 + mb.s2 * x.s1 + mb.s3 * x.s2 + mb.s4 * x.s3 + mb.s5 * x.s4;
   b.s3 = mb.s6 * x.s0 + mb.s7 * x.s1 + mb.s8 * x.s2 + mb.s9 * x.s3 + mb.sa * x.s4;
   b.s4 = mb.sb * x.s0 + mb.sc * x.s1 + mb.sd * x.s2 + mb.se * x.s3 + mb.sf * x.s4;
   b.s5 = b.s6 = b.s7 = 0.;

   return(b);
}

__kernel void MulMatVecSlc16( __global int   *D,
                              __global int16 *C,
                              __global fpn16 (*A)[25],
                              __global fpn8  *B,
                              __global fpn8  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn8  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l];

               b  = MulMatVec01(A[l][ 0], A[l][ 1],           X[ c.s0 ]);
   if(d >  1)  b += MulMatVec02(A[l][ 1], A[l][ 2], A[l][ 3], X[ c.s1 ]);
   if(d >  2)  b += MulMatVec03(A[l][ 3], A[l][ 4],           X[ c.s2 ]);
   if(d >  3)  b += MulMatVec04(A[l][ 4], A[l][ 5], A[l][ 6], X[ c.s3 ]);
   if(d >  4)  b += MulMatVec05(A[l][ 6], A[l][ 7],           X[ c.s4 ]);
   if(d >  5)  b += MulMatVec06(A[l][ 7], A[l][ 8], A[l][ 9], X[ c.s5 ]);
   if(d >  6)  b += MulMatVec07(A[l][ 9], A[l][10],           X[ c.s6 ]);
   if(d >  7)  b += MulMatVec08(A[l][10], A[l][11], A[l][12], X[ c.s7 ]);
   if(d >  8)  b += MulMatVec09(A[l][12], A[l][13], A[l][14], X[ c.s8 ]);
   if(d >  9)  b += MulMatVec10(A[l][14], A[l][15],           X[ c.s9 ]);
   if(d > 10)  b += MulMatVec11(A[l][15], A[l][16], A[l][17], X[ c.sa ]);
   if(d > 11)  b += MulMatVec12(A[l][17], A[l][18],           X[ c.sb ]);
   if(d > 12)  b += MulMatVec13(A[l][18], A[l][19], A[l][20], X[ c.sc ]);
   if(d > 13)  b += MulMatVec14(A[l][20], A[l][21],           X[ c.sd ]);
   if(d > 14)  b += MulMatVec15(A[l][21], A[l][22], A[l][23], X[ c.se ]);
   if(d > 15)  b += MulMatVec16(A[l][24], A[l][24],           X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc32( __global int   *D,
                              __global int16 (*C)[2],
                              __global fpn16 (*A)[50],
                              __global fpn8  *B,
                              __global fpn8  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn8  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b  = MulMatVec01(A[l][ 0], A[l][ 1],           X[ c.s0 ]);
      + MulMatVec02(A[l][ 1], A[l][ 2], A[l][ 3], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3], A[l][ 4],           X[ c.s2 ]);
      + MulMatVec04(A[l][ 4], A[l][ 5], A[l][ 6], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6], A[l][ 7],           X[ c.s4 ]);
      + MulMatVec06(A[l][ 7], A[l][ 8], A[l][ 9], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9], A[l][10],           X[ c.s6 ]);
      + MulMatVec08(A[l][10], A[l][11], A[l][12], X[ c.s7 ]);
      + MulMatVec09(A[l][12], A[l][13], A[l][14], X[ c.s8 ]);
      + MulMatVec10(A[l][14], A[l][15],           X[ c.s9 ]);
      + MulMatVec11(A[l][15], A[l][16], A[l][17], X[ c.sa ]);
      + MulMatVec12(A[l][17], A[l][18],           X[ c.sb ]);
      + MulMatVec13(A[l][18], A[l][19], A[l][20], X[ c.sc ]);
      + MulMatVec14(A[l][20], A[l][21],           X[ c.sd ]);
      + MulMatVec15(A[l][21], A[l][22], A[l][23], X[ c.se ]);
      + MulMatVec16(A[l][24], A[l][24],           X[ c.sf ]);

   c = C[l][1];

               b += MulMatVec01(A[l][ 0+25], A[l][ 1+25],              X[ c.s0 ]);
   if(d > 17)  b += MulMatVec02(A[l][ 1+25], A[l][ 2+25], A[l][ 3+25], X[ c.s1 ]);
   if(d > 18)  b += MulMatVec03(A[l][ 3+25], A[l][ 4+25],              X[ c.s2 ]);
   if(d > 19)  b += MulMatVec04(A[l][ 4+25], A[l][ 5+25], A[l][ 6+25], X[ c.s3 ]);
   if(d > 20)  b += MulMatVec05(A[l][ 6+25], A[l][ 7+25],              X[ c.s4 ]);
   if(d > 21)  b += MulMatVec06(A[l][ 7+25], A[l][ 8+25], A[l][ 9+25], X[ c.s5 ]);
   if(d > 22)  b += MulMatVec07(A[l][ 9+25], A[l][10+25],              X[ c.s6 ]);
   if(d > 23)  b += MulMatVec08(A[l][10+25], A[l][11+25], A[l][12+25], X[ c.s7 ]);
   if(d > 24)  b += MulMatVec09(A[l][12+25], A[l][13+25], A[l][14+25], X[ c.s8 ]);
   if(d > 25)  b += MulMatVec10(A[l][14+25], A[l][15+25],              X[ c.s9 ]);
   if(d > 26)  b += MulMatVec11(A[l][15+25], A[l][16+25], A[l][17+25], X[ c.sa ]);
   if(d > 27)  b += MulMatVec12(A[l][17+25], A[l][18+25],              X[ c.sb ]);
   if(d > 28)  b += MulMatVec13(A[l][18+25], A[l][19+25], A[l][20+25], X[ c.sc ]);
   if(d > 29)  b += MulMatVec14(A[l][20+25], A[l][21+25],              X[ c.sd ]);
   if(d > 30)  b += MulMatVec15(A[l][21+25], A[l][22+25], A[l][23+25], X[ c.se ]);
   if(d > 31)  b += MulMatVec16(A[l][24+25], A[l][24+25],              X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc64( __global int   *D,
                              __global int16 (*C)[4],
                              __global fpn16 (*A)[100],
                              __global fpn8  *B,
                              __global fpn8  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn8  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b  = MulMatVec01(A[l][ 0], A[l][ 1],           X[ c.s0 ]);
      + MulMatVec02(A[l][ 1], A[l][ 2], A[l][ 3], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3], A[l][ 4],           X[ c.s2 ]);
      + MulMatVec04(A[l][ 4], A[l][ 5], A[l][ 6], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6], A[l][ 7],           X[ c.s4 ]);
      + MulMatVec06(A[l][ 7], A[l][ 8], A[l][ 9], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9], A[l][10],           X[ c.s6 ]);
      + MulMatVec08(A[l][10], A[l][11], A[l][12], X[ c.s7 ]);
      + MulMatVec09(A[l][12], A[l][13], A[l][14], X[ c.s8 ]);
      + MulMatVec10(A[l][14], A[l][15],           X[ c.s9 ]);
      + MulMatVec11(A[l][15], A[l][16], A[l][17], X[ c.sa ]);
      + MulMatVec12(A[l][17], A[l][18],           X[ c.sb ]);
      + MulMatVec13(A[l][18], A[l][19], A[l][20], X[ c.sc ]);
      + MulMatVec14(A[l][20], A[l][21],           X[ c.sd ]);
      + MulMatVec15(A[l][21], A[l][22], A[l][23], X[ c.se ]);
      + MulMatVec16(A[l][24], A[l][24],           X[ c.sf ]);

   c = C[l][1];

   b += MulMatVec01(A[l][ 0+25], A[l][ 1+25],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+25], A[l][ 2+25], A[l][ 3+25], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+25], A[l][ 4+25],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+25], A[l][ 5+25], A[l][ 6+25], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+25], A[l][ 7+25],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+25], A[l][ 8+25], A[l][ 9+25], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+25], A[l][10+25],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+25], A[l][11+25], A[l][12+25], X[ c.s7 ]);
      + MulMatVec09(A[l][12+25], A[l][13+25], A[l][14+25], X[ c.s8 ]);
      + MulMatVec10(A[l][14+25], A[l][15+25],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+25], A[l][16+25], A[l][17+25], X[ c.sa ]);
      + MulMatVec12(A[l][17+25], A[l][18+25],              X[ c.sb ]);
      + MulMatVec13(A[l][18+25], A[l][19+25], A[l][20+25], X[ c.sc ]);
      + MulMatVec14(A[l][20+25], A[l][21+25],              X[ c.sd ]);
      + MulMatVec15(A[l][21+25], A[l][22+25], A[l][23+25], X[ c.se ]);
      + MulMatVec16(A[l][24+25], A[l][24+25],              X[ c.sf ]);

      c = C[l][2];

               b += MulMatVec01(A[l][ 0+50], A[l][ 1+50],              X[ c.s0 ]);
   if(d > 33)  b += MulMatVec02(A[l][ 1+50], A[l][ 2+50], A[l][ 3+50], X[ c.s1 ]);
   if(d > 34)  b += MulMatVec03(A[l][ 3+50], A[l][ 4+50],              X[ c.s2 ]);
   if(d > 35)  b += MulMatVec04(A[l][ 4+50], A[l][ 5+50], A[l][ 6+50], X[ c.s3 ]);
   if(d > 36)  b += MulMatVec05(A[l][ 6+50], A[l][ 7+50],              X[ c.s4 ]);
   if(d > 37)  b += MulMatVec06(A[l][ 7+50], A[l][ 8+50], A[l][ 9+50], X[ c.s5 ]);
   if(d > 38)  b += MulMatVec07(A[l][ 9+50], A[l][10+50],              X[ c.s6 ]);
   if(d > 39)  b += MulMatVec08(A[l][10+50], A[l][11+50], A[l][12+50], X[ c.s7 ]);
   if(d > 40)  b += MulMatVec09(A[l][12+50], A[l][13+50], A[l][14+50], X[ c.s8 ]);
   if(d > 41)  b += MulMatVec10(A[l][14+50], A[l][15+50],              X[ c.s9 ]);
   if(d > 42)  b += MulMatVec11(A[l][15+50], A[l][16+50], A[l][17+50], X[ c.sa ]);
   if(d > 43)  b += MulMatVec12(A[l][17+50], A[l][18+50],              X[ c.sb ]);
   if(d > 44)  b += MulMatVec13(A[l][18+50], A[l][19+50], A[l][20+50], X[ c.sc ]);
   if(d > 45)  b += MulMatVec14(A[l][20+50], A[l][21+50],              X[ c.sd ]);
   if(d > 46)  b += MulMatVec15(A[l][21+50], A[l][22+50], A[l][23+50], X[ c.se ]);
   if(d > 47)  b += MulMatVec16(A[l][24+50], A[l][24+50],              X[ c.sf ]);

   c = C[l][3];

   if(d > 48)  b += MulMatVec01(A[l][ 0+75], A[l][ 1+75],              X[ c.s0 ]);
   if(d > 49)  b += MulMatVec02(A[l][ 1+75], A[l][ 2+75], A[l][ 3+75], X[ c.s1 ]);
   if(d > 50)  b += MulMatVec03(A[l][ 3+75], A[l][ 4+75],              X[ c.s2 ]);
   if(d > 51)  b += MulMatVec04(A[l][ 4+75], A[l][ 5+75], A[l][ 6+75], X[ c.s3 ]);
   if(d > 52)  b += MulMatVec05(A[l][ 6+75], A[l][ 7+75],              X[ c.s4 ]);
   if(d > 53)  b += MulMatVec06(A[l][ 7+75], A[l][ 8+75], A[l][ 9+75], X[ c.s5 ]);
   if(d > 54)  b += MulMatVec07(A[l][ 9+75], A[l][10+75],              X[ c.s6 ]);
   if(d > 55)  b += MulMatVec08(A[l][10+75], A[l][11+75], A[l][12+75], X[ c.s7 ]);
   if(d > 56)  b += MulMatVec09(A[l][12+75], A[l][13+75], A[l][14+75], X[ c.s8 ]);
   if(d > 57)  b += MulMatVec10(A[l][14+75], A[l][15+75],              X[ c.s9 ]);
   if(d > 58)  b += MulMatVec11(A[l][15+75], A[l][16+75], A[l][17+75], X[ c.sa ]);
   if(d > 59)  b += MulMatVec12(A[l][17+75], A[l][18+75],              X[ c.sb ]);
   if(d > 60)  b += MulMatVec13(A[l][18+75], A[l][19+75], A[l][20+75], X[ c.sc ]);
   if(d > 61)  b += MulMatVec14(A[l][20+75], A[l][21+75],              X[ c.sd ]);
   if(d > 62)  b += MulMatVec15(A[l][21+75], A[l][22+75], A[l][23+75], X[ c.se ]);
   if(d > 63)  b += MulMatVec16(A[l][24+75], A[l][24+75],              X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc128(__global int   *D,
                              __global int16 (*C)[8],
                              __global fpn16 (*A)[200],
                              __global fpn8  *B,
                              __global fpn8  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn8  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b  = MulMatVec01(A[l][ 0], A[l][ 1],           X[ c.s0 ]);
      + MulMatVec02(A[l][ 1], A[l][ 2], A[l][ 3], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3], A[l][ 4],           X[ c.s2 ]);
      + MulMatVec04(A[l][ 4], A[l][ 5], A[l][ 6], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6], A[l][ 7],           X[ c.s4 ]);
      + MulMatVec06(A[l][ 7], A[l][ 8], A[l][ 9], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9], A[l][10],           X[ c.s6 ]);
      + MulMatVec08(A[l][10], A[l][11], A[l][12], X[ c.s7 ]);
      + MulMatVec09(A[l][12], A[l][13], A[l][14], X[ c.s8 ]);
      + MulMatVec10(A[l][14], A[l][15],           X[ c.s9 ]);
      + MulMatVec11(A[l][15], A[l][16], A[l][17], X[ c.sa ]);
      + MulMatVec12(A[l][17], A[l][18],           X[ c.sb ]);
      + MulMatVec13(A[l][18], A[l][19], A[l][20], X[ c.sc ]);
      + MulMatVec14(A[l][20], A[l][21],           X[ c.sd ]);
      + MulMatVec15(A[l][21], A[l][22], A[l][23], X[ c.se ]);
      + MulMatVec16(A[l][24], A[l][24],           X[ c.sf ]);

   c = C[l][1];

   b += MulMatVec01(A[l][ 0+25], A[l][ 1+25],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+25], A[l][ 2+25], A[l][ 3+25], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+25], A[l][ 4+25],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+25], A[l][ 5+25], A[l][ 6+25], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+25], A[l][ 7+25],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+25], A[l][ 8+25], A[l][ 9+25], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+25], A[l][10+25],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+25], A[l][11+25], A[l][12+25], X[ c.s7 ]);
      + MulMatVec09(A[l][12+25], A[l][13+25], A[l][14+25], X[ c.s8 ]);
      + MulMatVec10(A[l][14+25], A[l][15+25],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+25], A[l][16+25], A[l][17+25], X[ c.sa ]);
      + MulMatVec12(A[l][17+25], A[l][18+25],              X[ c.sb ]);
      + MulMatVec13(A[l][18+25], A[l][19+25], A[l][20+25], X[ c.sc ]);
      + MulMatVec14(A[l][20+25], A[l][21+25],              X[ c.sd ]);
      + MulMatVec15(A[l][21+25], A[l][22+25], A[l][23+25], X[ c.se ]);
      + MulMatVec16(A[l][24+25], A[l][24+25],              X[ c.sf ]);

   c = C[l][2];

   b += MulMatVec01(A[l][ 0+50], A[l][ 1+50],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+50], A[l][ 2+50], A[l][ 3+50], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+50], A[l][ 4+50],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+50], A[l][ 5+50], A[l][ 6+50], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+50], A[l][ 7+50],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+50], A[l][ 8+50], A[l][ 9+50], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+50], A[l][10+50],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+50], A[l][11+50], A[l][12+50], X[ c.s7 ]);
      + MulMatVec09(A[l][12+50], A[l][13+50], A[l][14+50], X[ c.s8 ]);
      + MulMatVec10(A[l][14+50], A[l][15+50],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+50], A[l][16+50], A[l][17+50], X[ c.sa ]);
      + MulMatVec12(A[l][17+50], A[l][18+50],              X[ c.sb ]);
      + MulMatVec13(A[l][18+50], A[l][19+50], A[l][20+50], X[ c.sc ]);
      + MulMatVec14(A[l][20+50], A[l][21+50],              X[ c.sd ]);
      + MulMatVec15(A[l][21+50], A[l][22+50], A[l][23+50], X[ c.se ]);
      + MulMatVec16(A[l][24+50], A[l][24+50],              X[ c.sf ]);

   c = C[l][3];

   b += MulMatVec01(A[l][ 0+75], A[l][ 1+75],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+75], A[l][ 2+75], A[l][ 3+75], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+75], A[l][ 4+75],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+75], A[l][ 5+75], A[l][ 6+75], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+75], A[l][ 7+75],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+75], A[l][ 8+75], A[l][ 9+75], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+75], A[l][10+75],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+75], A[l][11+75], A[l][12+75], X[ c.s7 ]);
      + MulMatVec09(A[l][12+75], A[l][13+75], A[l][14+75], X[ c.s8 ]);
      + MulMatVec10(A[l][14+75], A[l][15+75],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+75], A[l][16+75], A[l][17+75], X[ c.sa ]);
      + MulMatVec12(A[l][17+75], A[l][18+75],              X[ c.sb ]);
      + MulMatVec13(A[l][18+75], A[l][19+75], A[l][20+75], X[ c.sc ]);
      + MulMatVec14(A[l][20+75], A[l][21+75],              X[ c.sd ]);
      + MulMatVec15(A[l][21+75], A[l][22+75], A[l][23+75], X[ c.se ]);
      + MulMatVec16(A[l][24+75], A[l][24+75],              X[ c.sf ]);

   c = C[l][4];

               b += MulMatVec01(A[l][ 0+100], A[l][ 1+100],               X[ c.s0 ]);
   if(d > 65)  b += MulMatVec02(A[l][ 1+100], A[l][ 2+100], A[l][ 3+100], X[ c.s1 ]);
   if(d > 66)  b += MulMatVec03(A[l][ 3+100], A[l][ 4+100],               X[ c.s2 ]);
   if(d > 67)  b += MulMatVec04(A[l][ 4+100], A[l][ 5+100], A[l][ 6+100], X[ c.s3 ]);
   if(d > 68)  b += MulMatVec05(A[l][ 6+100], A[l][ 7+100],               X[ c.s4 ]);
   if(d > 69)  b += MulMatVec06(A[l][ 7+100], A[l][ 8+100], A[l][ 9+100], X[ c.s5 ]);
   if(d > 70)  b += MulMatVec07(A[l][ 9+100], A[l][10+100],               X[ c.s6 ]);
   if(d > 71)  b += MulMatVec08(A[l][10+100], A[l][11+100], A[l][12+100], X[ c.s7 ]);
   if(d > 72)  b += MulMatVec09(A[l][12+100], A[l][13+100], A[l][14+100], X[ c.s8 ]);
   if(d > 73)  b += MulMatVec10(A[l][14+100], A[l][15+100],               X[ c.s9 ]);
   if(d > 74)  b += MulMatVec11(A[l][15+100], A[l][16+100], A[l][17+100], X[ c.sa ]);
   if(d > 75)  b += MulMatVec12(A[l][17+100], A[l][18+100],               X[ c.sb ]);
   if(d > 76)  b += MulMatVec13(A[l][18+100], A[l][19+100], A[l][20+100], X[ c.sc ]);
   if(d > 77)  b += MulMatVec14(A[l][20+100], A[l][21+100],               X[ c.sd ]);
   if(d > 78)  b += MulMatVec15(A[l][21+100], A[l][22+100], A[l][23+100], X[ c.se ]);
   if(d > 79)  b += MulMatVec16(A[l][24+100], A[l][24+100],               X[ c.sf ]);

   c = C[l][5];

   if(d > 80)  b += MulMatVec01(A[l][ 0+125], A[l][ 1+125],               X[ c.s0 ]);
   if(d > 81)  b += MulMatVec02(A[l][ 1+125], A[l][ 2+125], A[l][ 3+125], X[ c.s1 ]);
   if(d > 82)  b += MulMatVec03(A[l][ 3+125], A[l][ 4+125],               X[ c.s2 ]);
   if(d > 83)  b += MulMatVec04(A[l][ 4+125], A[l][ 5+125], A[l][ 6+125], X[ c.s3 ]);
   if(d > 84)  b += MulMatVec05(A[l][ 6+125], A[l][ 7+125],               X[ c.s4 ]);
   if(d > 85)  b += MulMatVec06(A[l][ 7+125], A[l][ 8+125], A[l][ 9+125], X[ c.s5 ]);
   if(d > 86)  b += MulMatVec07(A[l][ 9+125], A[l][10+125],               X[ c.s6 ]);
   if(d > 87)  b += MulMatVec08(A[l][10+125], A[l][11+125], A[l][12+125], X[ c.s7 ]);
   if(d > 88)  b += MulMatVec09(A[l][12+125], A[l][13+125], A[l][14+125], X[ c.s8 ]);
   if(d > 89)  b += MulMatVec10(A[l][14+125], A[l][15+125],               X[ c.s9 ]);
   if(d > 90)  b += MulMatVec11(A[l][15+125], A[l][16+125], A[l][17+125], X[ c.sa ]);
   if(d > 91)  b += MulMatVec12(A[l][17+125], A[l][18+125],               X[ c.sb ]);
   if(d > 92)  b += MulMatVec13(A[l][18+125], A[l][19+125], A[l][20+125], X[ c.sc ]);
   if(d > 93)  b += MulMatVec14(A[l][20+125], A[l][21+125],               X[ c.sd ]);
   if(d > 94)  b += MulMatVec15(A[l][21+125], A[l][22+125], A[l][23+125], X[ c.se ]);
   if(d > 95)  b += MulMatVec16(A[l][24+125], A[l][24+125],               X[ c.sf ]);

   c = C[l][6];

   if(d >  96) b += MulMatVec01(A[l][ 0+150], A[l][ 1+150],               X[ c.s0 ]);
   if(d >  97) b += MulMatVec02(A[l][ 1+150], A[l][ 2+150], A[l][ 3+150], X[ c.s1 ]);
   if(d >  98) b += MulMatVec03(A[l][ 3+150], A[l][ 4+150],               X[ c.s2 ]);
   if(d >  99) b += MulMatVec04(A[l][ 4+150], A[l][ 5+150], A[l][ 6+150], X[ c.s3 ]);
   if(d > 100) b += MulMatVec05(A[l][ 6+150], A[l][ 7+150],               X[ c.s4 ]);
   if(d > 101) b += MulMatVec06(A[l][ 7+150], A[l][ 8+150], A[l][ 9+150], X[ c.s5 ]);
   if(d > 102) b += MulMatVec07(A[l][ 9+150], A[l][10+150],               X[ c.s6 ]);
   if(d > 103) b += MulMatVec08(A[l][10+150], A[l][11+150], A[l][12+150], X[ c.s7 ]);
   if(d > 104) b += MulMatVec09(A[l][12+150], A[l][13+150], A[l][14+150], X[ c.s8 ]);
   if(d > 105) b += MulMatVec10(A[l][14+150], A[l][15+150],               X[ c.s9 ]);
   if(d > 106) b += MulMatVec11(A[l][15+150], A[l][16+150], A[l][17+150], X[ c.sa ]);
   if(d > 107) b += MulMatVec12(A[l][17+150], A[l][18+150],               X[ c.sb ]);
   if(d > 108) b += MulMatVec13(A[l][18+150], A[l][19+150], A[l][20+150], X[ c.sc ]);
   if(d > 109) b += MulMatVec14(A[l][20+150], A[l][21+150],               X[ c.sd ]);
   if(d > 110) b += MulMatVec15(A[l][21+150], A[l][22+150], A[l][23+150], X[ c.se ]);
   if(d > 111) b += MulMatVec16(A[l][24+150], A[l][24+150],               X[ c.sf ]);

   c = C[l][7];

   if(d > 112) b += MulMatVec01(A[l][ 0+175], A[l][ 1+175],               X[ c.s0 ]);
   if(d > 113) b += MulMatVec02(A[l][ 1+175], A[l][ 2+175], A[l][ 3+175], X[ c.s1 ]);
   if(d > 114) b += MulMatVec03(A[l][ 3+175], A[l][ 4+175],               X[ c.s2 ]);
   if(d > 115) b += MulMatVec04(A[l][ 4+175], A[l][ 5+175], A[l][ 6+175], X[ c.s3 ]);
   if(d > 116) b += MulMatVec05(A[l][ 6+175], A[l][ 7+175],               X[ c.s4 ]);
   if(d > 117) b += MulMatVec06(A[l][ 7+175], A[l][ 8+175], A[l][ 9+175], X[ c.s5 ]);
   if(d > 118) b += MulMatVec07(A[l][ 9+175], A[l][10+175],               X[ c.s6 ]);
   if(d > 119) b += MulMatVec08(A[l][10+175], A[l][11+175], A[l][12+175], X[ c.s7 ]);
   if(d > 120) b += MulMatVec09(A[l][12+175], A[l][13+175], A[l][14+175], X[ c.s8 ]);
   if(d > 121) b += MulMatVec10(A[l][14+175], A[l][15+175],               X[ c.s9 ]);
   if(d > 122) b += MulMatVec11(A[l][15+175], A[l][16+175], A[l][17+175], X[ c.sa ]);
   if(d > 123) b += MulMatVec12(A[l][17+175], A[l][18+175],               X[ c.sb ]);
   if(d > 124) b += MulMatVec13(A[l][18+175], A[l][19+175], A[l][20+175], X[ c.sc ]);
   if(d > 125) b += MulMatVec14(A[l][20+175], A[l][21+175],               X[ c.sd ]);
   if(d > 126) b += MulMatVec15(A[l][21+175], A[l][22+175], A[l][23+175], X[ c.se ]);
   if(d > 127) b += MulMatVec16(A[l][24+175], A[l][24+175],               X[ c.sf ]);

   B[l+N.s1] = b;
}

__kernel void MulMatVecSlc256(__global int   *D,
                              __global int16 (*C)[16],
                              __global fpn16 (*A)[400],
                              __global fpn8  *B,
                              __global fpn8  *X,
                              __global void  *par,
                              const int2     N )
{
   int   d, l;
   int16 c;
   fpn8  b;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   c = C[l][0];

   b  = MulMatVec01(A[l][ 0], A[l][ 1],           X[ c.s0 ]);
      + MulMatVec02(A[l][ 1], A[l][ 2], A[l][ 3], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3], A[l][ 4],           X[ c.s2 ]);
      + MulMatVec04(A[l][ 4], A[l][ 5], A[l][ 6], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6], A[l][ 7],           X[ c.s4 ]);
      + MulMatVec06(A[l][ 7], A[l][ 8], A[l][ 9], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9], A[l][10],           X[ c.s6 ]);
      + MulMatVec08(A[l][10], A[l][11], A[l][12], X[ c.s7 ]);
      + MulMatVec09(A[l][12], A[l][13], A[l][14], X[ c.s8 ]);
      + MulMatVec10(A[l][14], A[l][15],           X[ c.s9 ]);
      + MulMatVec11(A[l][15], A[l][16], A[l][17], X[ c.sa ]);
      + MulMatVec12(A[l][17], A[l][18],           X[ c.sb ]);
      + MulMatVec13(A[l][18], A[l][19], A[l][20], X[ c.sc ]);
      + MulMatVec14(A[l][20], A[l][21],           X[ c.sd ]);
      + MulMatVec15(A[l][21], A[l][22], A[l][23], X[ c.se ]);
      + MulMatVec16(A[l][24], A[l][24],           X[ c.sf ]);

   c = C[l][1];

   b += MulMatVec01(A[l][ 0+25], A[l][ 1+25],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+25], A[l][ 2+25], A[l][ 3+25], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+25], A[l][ 4+25],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+25], A[l][ 5+25], A[l][ 6+25], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+25], A[l][ 7+25],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+25], A[l][ 8+25], A[l][ 9+25], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+25], A[l][10+25],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+25], A[l][11+25], A[l][12+25], X[ c.s7 ]);
      + MulMatVec09(A[l][12+25], A[l][13+25], A[l][14+25], X[ c.s8 ]);
      + MulMatVec10(A[l][14+25], A[l][15+25],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+25], A[l][16+25], A[l][17+25], X[ c.sa ]);
      + MulMatVec12(A[l][17+25], A[l][18+25],              X[ c.sb ]);
      + MulMatVec13(A[l][18+25], A[l][19+25], A[l][20+25], X[ c.sc ]);
      + MulMatVec14(A[l][20+25], A[l][21+25],              X[ c.sd ]);
      + MulMatVec15(A[l][21+25], A[l][22+25], A[l][23+25], X[ c.se ]);
      + MulMatVec16(A[l][24+25], A[l][24+25],              X[ c.sf ]);

   c = C[l][2];

   b += MulMatVec01(A[l][ 0+50], A[l][ 1+50],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+50], A[l][ 2+50], A[l][ 3+50], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+50], A[l][ 4+50],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+50], A[l][ 5+50], A[l][ 6+50], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+50], A[l][ 7+50],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+50], A[l][ 8+50], A[l][ 9+50], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+50], A[l][10+50],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+50], A[l][11+50], A[l][12+50], X[ c.s7 ]);
      + MulMatVec09(A[l][12+50], A[l][13+50], A[l][14+50], X[ c.s8 ]);
      + MulMatVec10(A[l][14+50], A[l][15+50],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+50], A[l][16+50], A[l][17+50], X[ c.sa ]);
      + MulMatVec12(A[l][17+50], A[l][18+50],              X[ c.sb ]);
      + MulMatVec13(A[l][18+50], A[l][19+50], A[l][20+50], X[ c.sc ]);
      + MulMatVec14(A[l][20+50], A[l][21+50],              X[ c.sd ]);
      + MulMatVec15(A[l][21+50], A[l][22+50], A[l][23+50], X[ c.se ]);
      + MulMatVec16(A[l][24+50], A[l][24+50],              X[ c.sf ]);

   c = C[l][3];

   b += MulMatVec01(A[l][ 0+75], A[l][ 1+75],              X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+75], A[l][ 2+75], A[l][ 3+75], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+75], A[l][ 4+75],              X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+75], A[l][ 5+75], A[l][ 6+75], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+75], A[l][ 7+75],              X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+75], A[l][ 8+75], A[l][ 9+75], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+75], A[l][10+75],              X[ c.s6 ]);
      + MulMatVec08(A[l][10+75], A[l][11+75], A[l][12+75], X[ c.s7 ]);
      + MulMatVec09(A[l][12+75], A[l][13+75], A[l][14+75], X[ c.s8 ]);
      + MulMatVec10(A[l][14+75], A[l][15+75],              X[ c.s9 ]);
      + MulMatVec11(A[l][15+75], A[l][16+75], A[l][17+75], X[ c.sa ]);
      + MulMatVec12(A[l][17+75], A[l][18+75],              X[ c.sb ]);
      + MulMatVec13(A[l][18+75], A[l][19+75], A[l][20+75], X[ c.sc ]);
      + MulMatVec14(A[l][20+75], A[l][21+75],              X[ c.sd ]);
      + MulMatVec15(A[l][21+75], A[l][22+75], A[l][23+75], X[ c.se ]);
      + MulMatVec16(A[l][24+75], A[l][24+75],              X[ c.sf ]);

   c = C[l][4];

   b += MulMatVec01(A[l][ 0+100], A[l][ 1+100],               X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+100], A[l][ 2+100], A[l][ 3+100], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+100], A[l][ 4+100],               X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+100], A[l][ 5+100], A[l][ 6+100], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+100], A[l][ 7+100],               X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+100], A[l][ 8+100], A[l][ 9+100], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+100], A[l][10+100],               X[ c.s6 ]);
      + MulMatVec08(A[l][10+100], A[l][11+100], A[l][12+100], X[ c.s7 ]);
      + MulMatVec09(A[l][12+100], A[l][13+100], A[l][14+100], X[ c.s8 ]);
      + MulMatVec10(A[l][14+100], A[l][15+100],               X[ c.s9 ]);
      + MulMatVec11(A[l][15+100], A[l][16+100], A[l][17+100], X[ c.sa ]);
      + MulMatVec12(A[l][17+100], A[l][18+100],               X[ c.sb ]);
      + MulMatVec13(A[l][18+100], A[l][19+100], A[l][20+100], X[ c.sc ]);
      + MulMatVec14(A[l][20+100], A[l][21+100],               X[ c.sd ]);
      + MulMatVec15(A[l][21+100], A[l][22+100], A[l][23+100], X[ c.se ]);
      + MulMatVec16(A[l][24+100], A[l][24+100],               X[ c.sf ]);

   c = C[l][5];

   b += MulMatVec01(A[l][ 0+125], A[l][ 1+125],               X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+125], A[l][ 2+125], A[l][ 3+125], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+125], A[l][ 4+125],               X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+125], A[l][ 5+125], A[l][ 6+125], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+125], A[l][ 7+125],               X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+125], A[l][ 8+125], A[l][ 9+125], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+125], A[l][10+125],               X[ c.s6 ]);
      + MulMatVec08(A[l][10+125], A[l][11+125], A[l][12+125], X[ c.s7 ]);
      + MulMatVec09(A[l][12+125], A[l][13+125], A[l][14+125], X[ c.s8 ]);
      + MulMatVec10(A[l][14+125], A[l][15+125],               X[ c.s9 ]);
      + MulMatVec11(A[l][15+125], A[l][16+125], A[l][17+125], X[ c.sa ]);
      + MulMatVec12(A[l][17+125], A[l][18+125],               X[ c.sb ]);
      + MulMatVec13(A[l][18+125], A[l][19+125], A[l][20+125], X[ c.sc ]);
      + MulMatVec14(A[l][20+125], A[l][21+125],               X[ c.sd ]);
      + MulMatVec15(A[l][21+125], A[l][22+125], A[l][23+125], X[ c.se ]);
      + MulMatVec16(A[l][24+125], A[l][24+125],               X[ c.sf ]);

   c = C[l][6];

   b += MulMatVec01(A[l][ 0+150], A[l][ 1+150],               X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+150], A[l][ 2+150], A[l][ 3+150], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+150], A[l][ 4+150],               X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+150], A[l][ 5+150], A[l][ 6+150], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+150], A[l][ 7+150],               X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+150], A[l][ 8+150], A[l][ 9+150], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+150], A[l][10+150],               X[ c.s6 ]);
      + MulMatVec08(A[l][10+150], A[l][11+150], A[l][12+150], X[ c.s7 ]);
      + MulMatVec09(A[l][12+150], A[l][13+150], A[l][14+150], X[ c.s8 ]);
      + MulMatVec10(A[l][14+150], A[l][15+150],               X[ c.s9 ]);
      + MulMatVec11(A[l][15+150], A[l][16+150], A[l][17+150], X[ c.sa ]);
      + MulMatVec12(A[l][17+150], A[l][18+150],               X[ c.sb ]);
      + MulMatVec13(A[l][18+150], A[l][19+150], A[l][20+150], X[ c.sc ]);
      + MulMatVec14(A[l][20+150], A[l][21+150],               X[ c.sd ]);
      + MulMatVec15(A[l][21+150], A[l][22+150], A[l][23+150], X[ c.se ]);
      + MulMatVec16(A[l][24+150], A[l][24+150],               X[ c.sf ]);

   c = C[l][7];

   b += MulMatVec01(A[l][ 0+175], A[l][ 1+175],               X[ c.s0 ]);
      + MulMatVec02(A[l][ 1+175], A[l][ 2+175], A[l][ 3+175], X[ c.s1 ]);
      + MulMatVec03(A[l][ 3+175], A[l][ 4+175],               X[ c.s2 ]);
      + MulMatVec04(A[l][ 4+175], A[l][ 5+175], A[l][ 6+175], X[ c.s3 ]);
      + MulMatVec05(A[l][ 6+175], A[l][ 7+175],               X[ c.s4 ]);
      + MulMatVec06(A[l][ 7+175], A[l][ 8+175], A[l][ 9+175], X[ c.s5 ]);
      + MulMatVec07(A[l][ 9+175], A[l][10+175],               X[ c.s6 ]);
      + MulMatVec08(A[l][10+175], A[l][11+175], A[l][12+175], X[ c.s7 ]);
      + MulMatVec09(A[l][12+175], A[l][13+175], A[l][14+175], X[ c.s8 ]);
      + MulMatVec10(A[l][14+175], A[l][15+175],               X[ c.s9 ]);
      + MulMatVec11(A[l][15+175], A[l][16+175], A[l][17+175], X[ c.sa ]);
      + MulMatVec12(A[l][17+175], A[l][18+175],               X[ c.sb ]);
      + MulMatVec13(A[l][18+175], A[l][19+175], A[l][20+175], X[ c.sc ]);
      + MulMatVec14(A[l][20+175], A[l][21+175],               X[ c.sd ]);
      + MulMatVec15(A[l][21+175], A[l][22+175], A[l][23+175], X[ c.se ]);
      + MulMatVec16(A[l][24+175], A[l][24+175],               X[ c.sf ]);

   c = C[l][8];

               b += MulMatVec01(A[l][ 0+200], A[l][ 1+200],               X[ c.s0 ]);
   if(d > 129) b += MulMatVec02(A[l][ 1+200], A[l][ 2+200], A[l][ 3+200], X[ c.s1 ]);
   if(d > 130) b += MulMatVec03(A[l][ 3+200], A[l][ 4+200],               X[ c.s2 ]);
   if(d > 131) b += MulMatVec04(A[l][ 4+200], A[l][ 5+200], A[l][ 6+200], X[ c.s3 ]);
   if(d > 132) b += MulMatVec05(A[l][ 6+200], A[l][ 7+200],               X[ c.s4 ]);
   if(d > 133) b += MulMatVec06(A[l][ 7+200], A[l][ 8+200], A[l][ 9+200], X[ c.s5 ]);
   if(d > 134) b += MulMatVec07(A[l][ 9+200], A[l][10+200],               X[ c.s6 ]);
   if(d > 135) b += MulMatVec08(A[l][10+200], A[l][11+200], A[l][12+200], X[ c.s7 ]);
   if(d > 136) b += MulMatVec09(A[l][12+200], A[l][13+200], A[l][14+200], X[ c.s8 ]);
   if(d > 137) b += MulMatVec10(A[l][14+200], A[l][15+200],               X[ c.s9 ]);
   if(d > 138) b += MulMatVec11(A[l][15+200], A[l][16+200], A[l][17+200], X[ c.sa ]);
   if(d > 139) b += MulMatVec12(A[l][17+200], A[l][18+200],               X[ c.sb ]);
   if(d > 140) b += MulMatVec13(A[l][18+200], A[l][19+200], A[l][20+200], X[ c.sc ]);
   if(d > 141) b += MulMatVec14(A[l][20+200], A[l][21+200],               X[ c.sd ]);
   if(d > 142) b += MulMatVec15(A[l][21+200], A[l][22+200], A[l][23+200], X[ c.se ]);
   if(d > 143) b += MulMatVec16(A[l][24+200], A[l][24+200],               X[ c.sf ]);

   c = C[l][9];

   if(d > 144) b += MulMatVec01(A[l][ 0+225], A[l][ 1+225],               X[ c.s0 ]);
   if(d > 145) b += MulMatVec02(A[l][ 1+225], A[l][ 2+225], A[l][ 3+225], X[ c.s1 ]);
   if(d > 146) b += MulMatVec03(A[l][ 3+225], A[l][ 4+225],               X[ c.s2 ]);
   if(d > 147) b += MulMatVec04(A[l][ 4+225], A[l][ 5+225], A[l][ 6+225], X[ c.s3 ]);
   if(d > 148) b += MulMatVec05(A[l][ 6+225], A[l][ 7+225],               X[ c.s4 ]);
   if(d > 149) b += MulMatVec06(A[l][ 7+225], A[l][ 8+225], A[l][ 9+225], X[ c.s5 ]);
   if(d > 150) b += MulMatVec07(A[l][ 9+225], A[l][10+225],               X[ c.s6 ]);
   if(d > 151) b += MulMatVec08(A[l][10+225], A[l][11+225], A[l][12+225], X[ c.s7 ]);
   if(d > 152) b += MulMatVec09(A[l][12+225], A[l][13+225], A[l][14+225], X[ c.s8 ]);
   if(d > 153) b += MulMatVec10(A[l][14+225], A[l][15+225],               X[ c.s9 ]);
   if(d > 154) b += MulMatVec11(A[l][15+225], A[l][16+225], A[l][17+225], X[ c.sa ]);
   if(d > 155) b += MulMatVec12(A[l][17+225], A[l][18+225],               X[ c.sb ]);
   if(d > 156) b += MulMatVec13(A[l][18+225], A[l][19+225], A[l][20+225], X[ c.sc ]);
   if(d > 157) b += MulMatVec14(A[l][20+225], A[l][21+225],               X[ c.sd ]);
   if(d > 158) b += MulMatVec15(A[l][21+225], A[l][22+225], A[l][23+225], X[ c.se ]);
   if(d > 159) b += MulMatVec16(A[l][24+225], A[l][24+225],               X[ c.sf ]);

   c = C[l][10];

   if(d > 160) b += MulMatVec01(A[l][ 0+250], A[l][ 1+250],               X[ c.s0 ]);
   if(d > 161) b += MulMatVec02(A[l][ 1+250], A[l][ 2+250], A[l][ 3+250], X[ c.s1 ]);
   if(d > 162) b += MulMatVec03(A[l][ 3+250], A[l][ 4+250],               X[ c.s2 ]);
   if(d > 163) b += MulMatVec04(A[l][ 4+250], A[l][ 5+250], A[l][ 6+250], X[ c.s3 ]);
   if(d > 164) b += MulMatVec05(A[l][ 6+250], A[l][ 7+250],               X[ c.s4 ]);
   if(d > 165) b += MulMatVec06(A[l][ 7+250], A[l][ 8+250], A[l][ 9+250], X[ c.s5 ]);
   if(d > 166) b += MulMatVec07(A[l][ 9+250], A[l][10+250],               X[ c.s6 ]);
   if(d > 167) b += MulMatVec08(A[l][10+250], A[l][11+250], A[l][12+250], X[ c.s7 ]);
   if(d > 168) b += MulMatVec09(A[l][12+250], A[l][13+250], A[l][14+250], X[ c.s8 ]);
   if(d > 169) b += MulMatVec10(A[l][14+250], A[l][15+250],               X[ c.s9 ]);
   if(d > 170) b += MulMatVec11(A[l][15+250], A[l][16+250], A[l][17+250], X[ c.sa ]);
   if(d > 171) b += MulMatVec12(A[l][17+250], A[l][18+250],               X[ c.sb ]);
   if(d > 172) b += MulMatVec13(A[l][18+250], A[l][19+250], A[l][20+250], X[ c.sc ]);
   if(d > 173) b += MulMatVec14(A[l][20+250], A[l][21+250],               X[ c.sd ]);
   if(d > 174) b += MulMatVec15(A[l][21+250], A[l][22+250], A[l][23+250], X[ c.se ]);
   if(d > 175) b += MulMatVec16(A[l][24+250], A[l][24+250],               X[ c.sf ]);

   c = C[l][11];

   if(d > 176) b += MulMatVec01(A[l][ 0+275], A[l][ 1+275],               X[ c.s0 ]);
   if(d > 177) b += MulMatVec02(A[l][ 1+275], A[l][ 2+275], A[l][ 3+275], X[ c.s1 ]);
   if(d > 178) b += MulMatVec03(A[l][ 3+275], A[l][ 4+275],               X[ c.s2 ]);
   if(d > 179) b += MulMatVec04(A[l][ 4+275], A[l][ 5+275], A[l][ 6+275], X[ c.s3 ]);
   if(d > 180) b += MulMatVec05(A[l][ 6+275], A[l][ 7+275],               X[ c.s4 ]);
   if(d > 181) b += MulMatVec06(A[l][ 7+275], A[l][ 8+275], A[l][ 9+275], X[ c.s5 ]);
   if(d > 182) b += MulMatVec07(A[l][ 9+275], A[l][10+275],               X[ c.s6 ]);
   if(d > 183) b += MulMatVec08(A[l][10+275], A[l][11+275], A[l][12+275], X[ c.s7 ]);
   if(d > 184) b += MulMatVec09(A[l][12+275], A[l][13+275], A[l][14+275], X[ c.s8 ]);
   if(d > 185) b += MulMatVec10(A[l][14+275], A[l][15+275],               X[ c.s9 ]);
   if(d > 186) b += MulMatVec11(A[l][15+275], A[l][16+275], A[l][17+275], X[ c.sa ]);
   if(d > 187) b += MulMatVec12(A[l][17+275], A[l][18+275],               X[ c.sb ]);
   if(d > 188) b += MulMatVec13(A[l][18+275], A[l][19+275], A[l][20+275], X[ c.sc ]);
   if(d > 189) b += MulMatVec14(A[l][20+275], A[l][21+275],               X[ c.sd ]);
   if(d > 190) b += MulMatVec15(A[l][21+275], A[l][22+275], A[l][23+275], X[ c.se ]);
   if(d > 191) b += MulMatVec16(A[l][24+275], A[l][24+275],               X[ c.sf ]);

   c = C[l][12];

   if(d > 192) b += MulMatVec01(A[l][ 0+300], A[l][ 1+300],               X[ c.s0 ]);
   if(d > 193) b += MulMatVec02(A[l][ 1+300], A[l][ 2+300], A[l][ 3+300], X[ c.s1 ]);
   if(d > 194) b += MulMatVec03(A[l][ 3+300], A[l][ 4+300],               X[ c.s2 ]);
   if(d > 195) b += MulMatVec04(A[l][ 4+300], A[l][ 5+300], A[l][ 6+300], X[ c.s3 ]);
   if(d > 196) b += MulMatVec05(A[l][ 6+300], A[l][ 7+300],               X[ c.s4 ]);
   if(d > 197) b += MulMatVec06(A[l][ 7+300], A[l][ 8+300], A[l][ 9+300], X[ c.s5 ]);
   if(d > 198) b += MulMatVec07(A[l][ 9+300], A[l][10+300],               X[ c.s6 ]);
   if(d > 199) b += MulMatVec08(A[l][10+300], A[l][11+300], A[l][12+300], X[ c.s7 ]);
   if(d > 200) b += MulMatVec09(A[l][12+300], A[l][13+300], A[l][14+300], X[ c.s8 ]);
   if(d > 201) b += MulMatVec10(A[l][14+300], A[l][15+300],               X[ c.s9 ]);
   if(d > 202) b += MulMatVec11(A[l][15+300], A[l][16+300], A[l][17+300], X[ c.sa ]);
   if(d > 203) b += MulMatVec12(A[l][17+300], A[l][18+300],               X[ c.sb ]);
   if(d > 204) b += MulMatVec13(A[l][18+300], A[l][19+300], A[l][20+300], X[ c.sc ]);
   if(d > 205) b += MulMatVec14(A[l][20+300], A[l][21+300],               X[ c.sd ]);
   if(d > 206) b += MulMatVec15(A[l][21+300], A[l][22+300], A[l][23+300], X[ c.se ]);
   if(d > 207) b += MulMatVec16(A[l][24+300], A[l][24+300],               X[ c.sf ]);

   c = C[l][13];

   if(d > 208) b += MulMatVec01(A[l][ 0+325], A[l][ 1+325],               X[ c.s0 ]);
   if(d > 209) b += MulMatVec02(A[l][ 1+325], A[l][ 2+325], A[l][ 3+325], X[ c.s1 ]);
   if(d > 210) b += MulMatVec03(A[l][ 3+325], A[l][ 4+325],               X[ c.s2 ]);
   if(d > 211) b += MulMatVec04(A[l][ 4+325], A[l][ 5+325], A[l][ 6+325], X[ c.s3 ]);
   if(d > 212) b += MulMatVec05(A[l][ 6+325], A[l][ 7+325],               X[ c.s4 ]);
   if(d > 213) b += MulMatVec06(A[l][ 7+325], A[l][ 8+325], A[l][ 9+325], X[ c.s5 ]);
   if(d > 214) b += MulMatVec07(A[l][ 9+325], A[l][10+325],               X[ c.s6 ]);
   if(d > 215) b += MulMatVec08(A[l][10+325], A[l][11+325], A[l][12+325], X[ c.s7 ]);
   if(d > 216) b += MulMatVec09(A[l][12+325], A[l][13+325], A[l][14+325], X[ c.s8 ]);
   if(d > 217) b += MulMatVec10(A[l][14+325], A[l][15+325],               X[ c.s9 ]);
   if(d > 218) b += MulMatVec11(A[l][15+325], A[l][16+325], A[l][17+325], X[ c.sa ]);
   if(d > 219) b += MulMatVec12(A[l][17+325], A[l][18+325],               X[ c.sb ]);
   if(d > 220) b += MulMatVec13(A[l][18+325], A[l][19+325], A[l][20+325], X[ c.sc ]);
   if(d > 221) b += MulMatVec14(A[l][20+325], A[l][21+325],               X[ c.sd ]);
   if(d > 222) b += MulMatVec15(A[l][21+325], A[l][22+325], A[l][23+325], X[ c.se ]);
   if(d > 223) b += MulMatVec16(A[l][24+325], A[l][24+325],               X[ c.sf ]);

   c = C[l][14];

   if(d > 224) b += MulMatVec01(A[l][ 0+350], A[l][ 1+350],               X[ c.s0 ]);
   if(d > 225) b += MulMatVec02(A[l][ 1+350], A[l][ 2+350], A[l][ 3+350], X[ c.s1 ]);
   if(d > 226) b += MulMatVec03(A[l][ 3+350], A[l][ 4+350],               X[ c.s2 ]);
   if(d > 227) b += MulMatVec04(A[l][ 4+350], A[l][ 5+350], A[l][ 6+350], X[ c.s3 ]);
   if(d > 228) b += MulMatVec05(A[l][ 6+350], A[l][ 7+350],               X[ c.s4 ]);
   if(d > 229) b += MulMatVec06(A[l][ 7+350], A[l][ 8+350], A[l][ 9+350], X[ c.s5 ]);
   if(d > 230) b += MulMatVec07(A[l][ 9+350], A[l][10+350],               X[ c.s6 ]);
   if(d > 231) b += MulMatVec08(A[l][10+350], A[l][11+350], A[l][12+350], X[ c.s7 ]);
   if(d > 232) b += MulMatVec09(A[l][12+350], A[l][13+350], A[l][14+350], X[ c.s8 ]);
   if(d > 233) b += MulMatVec10(A[l][14+350], A[l][15+350],               X[ c.s9 ]);
   if(d > 234) b += MulMatVec11(A[l][15+350], A[l][16+350], A[l][17+350], X[ c.sa ]);
   if(d > 235) b += MulMatVec12(A[l][17+350], A[l][18+350],               X[ c.sb ]);
   if(d > 236) b += MulMatVec13(A[l][18+350], A[l][19+350], A[l][20+350], X[ c.sc ]);
   if(d > 237) b += MulMatVec14(A[l][20+350], A[l][21+350],               X[ c.sd ]);
   if(d > 238) b += MulMatVec15(A[l][21+350], A[l][22+350], A[l][23+350], X[ c.se ]);
   if(d > 239) b += MulMatVec16(A[l][24+350], A[l][24+350],               X[ c.sf ]);

   c = C[l][15];

   if(d > 240) b += MulMatVec01(A[l][ 0+375], A[l][ 1+375],               X[ c.s0 ]);
   if(d > 241) b += MulMatVec02(A[l][ 1+375], A[l][ 2+375], A[l][ 3+375], X[ c.s1 ]);
   if(d > 242) b += MulMatVec03(A[l][ 3+375], A[l][ 4+375],               X[ c.s2 ]);
   if(d > 243) b += MulMatVec04(A[l][ 4+375], A[l][ 5+375], A[l][ 6+375], X[ c.s3 ]);
   if(d > 244) b += MulMatVec05(A[l][ 6+375], A[l][ 7+375],               X[ c.s4 ]);
   if(d > 245) b += MulMatVec06(A[l][ 7+375], A[l][ 8+375], A[l][ 9+375], X[ c.s5 ]);
   if(d > 246) b += MulMatVec07(A[l][ 9+375], A[l][10+375],               X[ c.s6 ]);
   if(d > 247) b += MulMatVec08(A[l][10+375], A[l][11+375], A[l][12+375], X[ c.s7 ]);
   if(d > 248) b += MulMatVec09(A[l][12+375], A[l][13+375], A[l][14+375], X[ c.s8 ]);
   if(d > 249) b += MulMatVec10(A[l][14+375], A[l][15+375],               X[ c.s9 ]);
   if(d > 250) b += MulMatVec11(A[l][15+375], A[l][16+375], A[l][17+375], X[ c.sa ]);
   if(d > 251) b += MulMatVec12(A[l][17+375], A[l][18+375],               X[ c.sb ]);
   if(d > 252) b += MulMatVec13(A[l][18+375], A[l][19+375], A[l][20+375], X[ c.sc ]);
   if(d > 253) b += MulMatVec14(A[l][20+375], A[l][21+375],               X[ c.sd ]);
   if(d > 254) b += MulMatVec15(A[l][21+375], A[l][22+375], A[l][23+375], X[ c.se ]);
   if(d > 255) b += MulMatVec16(A[l][24+375], A[l][24+375],               X[ c.sf ]);

   B[l+N.s1] = b;
}


#elif BLKSIZ == 7

fpn8 MulMatVec(fpn16 aa, fpn16 ab, fpn16 ac, fpn ad, fpn4 xa, fpn2 xb, fpn xc)
{
   fpn8 b;

   b.s0 = aa.s0 * xa.s0 + aa.s1 * xa.s1 + aa.s2 * xa.s2 + aa.s3 * xa.s3 + aa.s4 * xb.s0 + aa.s5 * xb.s1 + aa.s6 * xc;
   b.s1 = aa.s7 * xa.s0 + aa.s8 * xa.s1 + aa.s9 * xa.s2 + aa.sa * xa.s3 + aa.sb * xb.s0 + aa.sc * xb.s1 + aa.sd * xc;
   b.s2 = aa.se * xa.s0 + aa.sf * xa.s1 + ab.s0 * xa.s2 + ab.s1 * xa.s3 + ab.s2 * xb.s0 + ab.s3 * xb.s1 + ab.s4 * xc;
   b.s3 = ab.s5 * xa.s0 + ab.s6 * xa.s1 + ab.s7 * xa.s2 + ab.s8 * xa.s3 + ab.s9 * xb.s0 + ab.sa * xb.s1 + ab.sb * xc;
   b.s4 = ab.sc * xa.s0 + ab.sd * xa.s1 + ab.se * xa.s2 + ab.sf * xa.s3 + ac.s0 * xb.s0 + ac.s1 * xb.s1 + ac.s2 * xc;
   b.s5 = ac.s3 * xa.s0 + ac.s4 * xa.s1 + ac.s5 * xa.s2 + ac.s6 * xa.s3 + ac.s7 * xb.s0 + ac.s8 * xb.s1 + ac.s9 * xc;
   b.s6 = ac.sa * xa.s0 + ac.sb * xa.s1 + ac.sc * xa.s2 + ac.sd * xa.s3 + ac.se * xb.s0 + ac.sf * xb.s1 + ad    * xc;
   b.s7 = 0.;

   return(b);
}

#endif
