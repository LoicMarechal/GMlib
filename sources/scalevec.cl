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


typedef struct {
   int   foo;
   float res, scale;
}GmlParSct;


#if BLKSIZ == 4

__kernel void ScaleVec( __global fpn4 *U,
                        __global GmlParSct *par,
                        const int2 N )
{
   int l;
   fpn4 s = (fpn4){par->scale, par->scale, par->scale, par->scale};

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   U[l] *= s;
}

#else

__kernel void ScaleVec( __global fpn8 *U,
                        __global GmlParSct *par,
                        const int2 N )
{
   int l;
   fpn8 s = (fpn8){par->scale, par->scale, par->scale, par->scale,
                   par->scale, par->scale, par->scale, par->scale};

   l = get_global_id(0);

   if(l >= N.s0)
      return;

   U[l] *= s;
}

#endif
