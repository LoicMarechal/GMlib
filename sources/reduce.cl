
#ifndef MAX_WORKGROUP_SIZE
#define MAX_WORKGROUP_SIZE 1024
#endif

__kernel void reduce_min(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   float flt;
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   tmp[l] = (g < cnt) ? inp[g] : 1e37;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      if(i > l)
         tmp[l] = fmin(tmp[l], tmp[ l+i ]);
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
   {
      flt = tmp[0] ? tmp[0] : 1e37;
      out[ get_group_id(0) ] = flt;
   }
}

__kernel void reduce_max(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   float flt;
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   tmp[l] = (g < cnt) ? inp[g] : -1e37;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      if(i > l)
         tmp[l] = fmax(tmp[l], tmp[ l+i ]);
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
   {
      flt = tmp[0] ? tmp[0] : -1e37;
      out[ get_group_id(0) ] = flt;
   }
}

__kernel void reduce_Linf(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   tmp[l] = (g < cnt) ? fabs(inp[g]) : 0;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      if(i > l)
         tmp[l] = fmax(tmp[l], tmp[ l+i ]);
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
      out[ get_group_id(0) ] = tmp[0];
}

__kernel void reduce_sum(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   tmp[l] = (g < cnt) ? inp[g] : 0;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      tmp[l] += (i > l) ? tmp[ l+i ] : 0;
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
      out[ get_group_id(0) ] = tmp[0];
}

__kernel void reduce_L0(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   if(g < cnt && inp[g])
      tmp[l] = 1.;
   else
      tmp[l] = 0.;

   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      tmp[l] += (i > l) ? tmp[ l+i ] : 0;
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
      out[ get_group_id(0) ] = tmp[0];
}

__kernel void reduce_L1(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   tmp[l] = (g < cnt) ? fabs(inp[g]) : 0.;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      tmp[l] += (i > l) ? tmp[ l+i ] : 0;
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
      out[ get_group_id(0) ] = tmp[0];
}

__kernel void reduce_L2(__global float *inp, __global float *out, __global void *par, int cnt)
{
   int i, g=get_global_id(0), l=get_local_id(0);
   __local float tmp[ MAX_WORKGROUP_SIZE ];

   tmp[l] = (g < cnt) ? (inp[g] * inp[g]) : 0.;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(i=get_local_size(0)/2; i>0; i=i>>1)
   {
      tmp[l] += (i > l) ? tmp[ l+i ] : 0;
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(!l)
      out[ get_group_id(0) ] = tmp[0];
}
