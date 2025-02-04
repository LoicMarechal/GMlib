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

__kernel void AddVec(__global fpn4 *U,
                     __global fpn4 *V,
                     __global fpn4 *W,
                     __global fpn4 *Y,
                     __global void *par,
                     const int2 N)
{
   int l;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   Y[l] = U[l] + V[l] + W[l];
}

#else

__kernel void AddVec(__global fpn8 *U,
                     __global fpn8 *V,
                     __global fpn8 *W,
                     __global fpn8 *Y,
                     __global void *par,
                     const int2 N)
{
   int l;

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   Y[l] = U[l] + V[l] + W[l];
}

#endif
