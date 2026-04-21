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

__kernel void L2Norm(__global fpn4 *U,
                     __global float *V,
                     __global void *par,
                     const int2    N)
{
   int l;
   fpn s;
   fpn4 u;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   u = U[l];
   s = sqrt(u.s0 * u.s0 +  u.s1 * u.s1 +  u.s2 * u.s2 +  u.s3 * u.s3);

   V[l] = (float)s;
}

#elif BLKSIZ == 5

__kernel void L2Norm(__global fpn8 *U,
                     __global float *V,
                     __global void *par,
                     const int2    N)
{
   int l;
   fpn s;
   fpn8 u;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   u = U[l];
   s = sqrt(u.s0 * u.s0 +  u.s1 * u.s1 +  u.s2 * u.s2 +  u.s3 * u.s3 +  u.s4 * u.s4);

   V[l] = (float)s;
}

#else

__kernel void L2Norm(__global fpn8 *U,
                     __global float *V,
                     __global void *par,
                     const int2    N)
{
   int l;
   fpn s;
   fpn8 u;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   u = U[l];
   s = sqrt(u.s0 * u.s0 +  u.s1 * u.s1 +  u.s2 * u.s2 +  u.s3 * u.s3 +  u.s4 * u.s4 +  u.s5 * u.s5 +  u.s6 * u.s6);

   V[l] = (float)s;
}

#endif

