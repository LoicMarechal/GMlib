float CalEdgLen(float4, float4);
float CalTriSrf(float4, float4, float4);
float CalQadSrf(float4, float4, float4, float4);
float CalTetVol(float4, float4, float4, float4);

float CalEdgLen(float4 a, float4 b)
{
   return(fast_distance(a,b));
}

float CalTriSrf(float4 a, float4 b, float4 c)
{
   return(.5 * fast_length(cross(c-a, b-a)));
}

float CalQadSrf(float4 a, float4 b, float4 c, float4 d)
{
   return(.25 * (fast_length(cross(b-a, d-a))
               + fast_length(cross(c-b, a-b))
               + fast_length(cross(d-c, b-c))
               + fast_length(cross(a-d, c-d))));
}

float CalTetVol(float4 a, float4 b, float4 c, float4 d)
{
   return(1./6. * dot(cross(b-a, c-a), d-a));
}
