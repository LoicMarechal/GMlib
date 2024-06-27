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

__kernel void MultDiaglMatVec(__global fpn16 *D,
                              __global fpn4 *U,
                              __global void *par,
                              const int2 N)
{
   int l;
   fpn4 u, v;
   fpn16 d;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   d = D[l];
   u = U[l];

   v.s0 = d.s0 * u.s0 + d.s1 * u.s1 + d.s2 * u.s2 + d.s3 * u.s3;
   v.s1 = d.s4 * u.s0 + d.s5 * u.s1 + d.s6 * u.s2 + d.s7 * u.s3;
   v.s2 = d.s8 * u.s0 + d.s9 * u.s1 + d.sa * u.s2 + d.sb * u.s3;
   v.s3 = d.sc * u.s0 + d.sd * u.s1 + d.se * u.s2 + d.sf * u.s3;

   U[l] = v;
}

#else

__kernel void MultDiaglMatVec(__global fpn16 (*D)[2],
                              __global fpn8 *U,
                              __global void *par,
                              const int2 N)
{
   int l;
   fpn8 u, v;
   fpn16 da, db;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   da = D[l][0];
   db = D[l][1];
   u = U[l];

   v.s0 = da.s0 * u.s0 + da.s1 * u.s1 + da.s2 * u.s2 + da.s3 * u.s3 + da.s4 * u.s4;
   v.s1 = da.s5 * u.s0 + da.s6 * u.s1 + da.s7 * u.s2 + da.s8 * u.s3 + da.s9 * u.s4;
   v.s2 = da.sa * u.s0 + da.sb * u.s1 + da.sc * u.s2 + da.sd * u.s3 + da.se * u.s4;
   v.s3 = da.se * u.s0 + db.s0 * u.s1 + db.s1 * u.s2 + db.s2 * u.s3 + db.s3 * u.s4;
   v.s4 = db.s4 * u.s0 + db.s5 * u.s1 + db.s6 * u.s2 + db.s7 * u.s3 + db.s8 * u.s4;
   v.s5 = v.s6 = v.s7 = 0.;

   U[l] = v;
}

#endif
