float  DisPow   (float4, float4);
float  CalLen   (float4, float4);
float  CalSrf   (float4, float4, float4);
float  CalVol   (float4, float4, float4, float4);
float4 GetEdgTng(float4, float4);
float4 GetTriNrm(float4, float4, float4);
float4 GetQadNrm(float4, float4, float4, float4);
float  CalEdgLen(float4, float4);
float  CalTriSrf(float4, float4, float4);
float  CalQadSrf(float4, float4, float4, float4);
float  CalTetVol(float4, float4, float4, float4);
float  CalPyrVol(float4, float4, float4, float4, float4);
float  CalPriVol(float4, float4, float4, float4, float4, float4);
float  CalHexVol(float4, float4, float4, float4, float4, float4, float4, float4);
float  CalTriQal(float4, float4, float4);
float  CalQadQal(float4, float4, float4, float4);
float  CalTetQal(float4, float4, float4, float4);
float  CalPyrQal(float4, float4, float4, float4, float4);
float  CalPriQal(float4, float4, float4, float4, float4, float4);
float  CalHexQal(float4, float4, float4, float4, float4, float4, float4, float4);
float4 PrjVerLin(float4, float4, float4);
float4 PrjVerPla(float4, float4, float4);
float  DisVerLin(float4, float4, float4);
float  DisVerPla(float4, float4, float4);
float4 LinIntLin(float4, float4, float4, float4);
float4 LinIntPla(float4, float4, float4, float4);
void   PlaIntPla(float4, float4, float4, float4, float4 *, float4 *);
float  DisVerEdg(float4, float4, float4, float4);
void   MulMatVec(double16, double4, double4);


float4 PrjVerLin(float4 VerCrd, float4 LinCrd, float4 LinTng)
{
   return(LinCrd + dot(LinTng, VerCrd - LinCrd) * LinTng);
}

float4 PrjVerPla(float4 VerCrd, float4 PlaCrd, float4 PlaNrm)
{
   return(VerCrd + dot(PlaNrm, PlaCrd - VerCrd) * PlaNrm);
}

float DisVerLin(float4 VerCrd, float4 LinCrd, float4 LinTng)
{
   return(distance(VerCrd, PrjVerLin(VerCrd, LinCrd, LinTng)));
}

float DisVerPla(float4 VerCrd, float4 PlaCrd, float4 PlaNrm)
{
   return(dot(VerCrd - PlaCrd, PlaNrm));
}

float4 LinIntLin(float4 LinCrd1, float4 LinTng1, float4 LinCrd2, float4 LinTng2)
{
   float4 ImgCrd = PrjVerLin(LinCrd2, LinCrd1, LinTng1);
   return(ImgCrd - (distance(ImgCrd, LinCrd2) / dot(LinCrd2 - ImgCrd, LinTng2)) * LinTng1);
}

float4 LinIntPla(float4 LinCrd, float4 LinTng, float4 PlaCrd, float4 PlaNrm)
{
   return(LinCrd - (dot(PlaNrm, LinCrd - PlaCrd) / dot(PlaNrm, LinTng)) * LinTng);
}

void PlaIntPla(float4 PlaCrd1, float4 PlaNrm1, float4 PlaCrd2, float4 PlaNrm2, float4 *LinCrd1, float4 *LinTng1)
{
   *LinTng1 = normalize(cross(PlaNrm1, PlaNrm2));
   *LinCrd1 = LinIntPla(PlaCrd2, normalize(cross(*LinTng1, PlaNrm2)), PlaCrd1, PlaNrm1);
}

float DisPow(float4 a, float4 b)
{
   return(dot(a-b, a-b));
}

float CalLen(float4 a, float4 b)
{
   return(fast_distance(a,b));
}

float CalSrf(float4 a, float4 b, float4 c)
{
   return(fast_length(cross(c-a, b-a)));
}

float CalVol(float4 a, float4 b, float4 c, float4 d)
{
   return(dot(cross(b-a, c-a), d-a));
}

float DisVerEdg(float4 VerCrd, float4 EdgCrd1, float4 EdgCrd2, float4 EdgTng)
{
   float dis;
   float4 ImgCrd;

   dis = min(DisPow(VerCrd, EdgCrd1), DisPow(VerCrd, EdgCrd2));
   ImgCrd = PrjVerLin(VerCrd, EdgCrd1, EdgTng);

   return(sqrt(min(dis, DisPow(VerCrd, ImgCrd))));
}

float4 GetEdgTng(float4 a, float4 b)
{
   return(fast_normalize(b-a));
}

float4 GetTriNrm(float4 a, float4 b, float4 c)
{
   return(fast_normalize(cross(c-a, b-a)));
}

float4 GetQadNrm(float4 a, float4 b, float4 c, float4 d)
{
   return(fast_normalize(cross(c-a, d-b)));
}

float CalEdgLen(float4 a, float4 b)
{
   return(CalLen(a,b));
}

float CalTriSrf(float4 a, float4 b, float4 c)
{
   return(.5 * CalSrf(a,b,c));
}

float CalQadSrf(float4 a, float4 b, float4 c, float4 d)
{
   return(.5 * fast_length(cross(c-a, d-b)));
}

float CalTetVol(float4 a, float4 b, float4 c, float4 d)
{
   return(.166666 * CalVol(a,b,c,d));
}

float CalPyrVol(float4 a, float4 b, float4 c, float4 d, float4 e)
{
   return(.083333 * (CalVol(a,b,c,e)
                  +  CalVol(c,d,a,e)
                  +  CalVol(b,c,d,e)
                  +  CalVol(d,a,b,e)) );
}

float CalPriVol(float4 a, float4 b, float4 c, float4 d, float4 e, float4 f)
{
   return(.833333 * (CalVol(a,b,c,d)
                  +  CalVol(b,c,a,e)
                  +  CalVol(c,a,b,f)
                  +  CalVol(d,f,e,a)
                  +  CalVol(e,d,f,b)
                  +  CalVol(f,e,d,c)) );
}

float CalHexVol(  float4 a, float4 b, float4 c, float4 d,
                  float4 e, float4 f, float4 g, float4 h )
{
   return(.125 * (CalVol(a,b,d,e)
               +  CalVol(b,c,a,f)
               +  CalVol(c,d,b,g)
               +  CalVol(d,a,c,h)
               +  CalVol(e,h,f,a)
               +  CalVol(f,e,g,b)
               +  CalVol(g,f,h,c)
               +  CalVol(h,g,e,d)) );
}

float CalTriQal(float4 a, float4 b, float4 c)
{
   float ha, hb, hc, hmax;

   ha = CalLen(b,c);
   hb = CalLen(c,a);
   hc = CalLen(a,b);

   hmax = max(ha,   hb);
   hmax = max(hmax, hc);

   return( 3.46410 * CalSrf(a,b,c) / (hmax * (ha + hb + hc)) );
}

float CalQadQal(float4 a, float4 b, float4 c, float4 d)
{
   float h1, h2, h3, h4, h5, h6, hmax, s1, s2, s3, s4, smin;

   h1 = CalLen(a,b);
   h2 = CalLen(b,c);
   h3 = CalLen(c,d);
   h4 = CalLen(d,a);
   h5 = CalLen(a,c) * .707106;
   h6 = CalLen(b,d) * .707106;

   hmax = max(h1,   h2);
   hmax = max(hmax, h3);
   hmax = max(hmax, h4);
   hmax = max(hmax, h5);
   hmax = max(hmax, h6);

   s1 = CalSrf(a,b,c);
   s2 = CalSrf(b,c,d);
   s3 = CalSrf(c,d,a);
   s4 = CalSrf(d,a,b);

   smin = min(s1,   s2);
   smin = min(smin, s3);
   smin = min(smin, s4);

   return(4. * smin / (hmax * (h1 + h2 + h3 + h4)) );
}

float CalTetQal(float4 a, float4 b, float4 c, float4 d)
{
   float h, s, v;

   h = CalLen(a,b)
     + CalLen(a,c)
     + CalLen(a,d)
     + CalLen(b,c)
     + CalLen(b,c)
     + CalLen(c,d);

   s = CalSrf(a,b,c)
     + CalSrf(c,a,d)
     + CalSrf(b,c,d)
     + CalSrf(d,a,b);

   v = CalVol(a,b,c,d);

   return(176.363 * v / (h * s) );
}

float CalPyrQal(float4 a, float4 b, float4 c, float4 d, float4 e)
{
   float h, s, v;

   h = CalLen(a,b)
     + CalLen(b,c)
     + CalLen(c,d)
     + CalLen(d,a)
     + CalLen(e,a)
     + CalLen(e,b)
     + CalLen(e,c)
     + CalLen(e,d);

   s = CalTriSrf(e,a,b)
     + CalTriSrf(e,b,c)
     + CalTriSrf(e,c,d)
     + CalTriSrf(e,d,a)
     + CalQadSrf(d,c,b,a);

   v =        CalVol(a,b,c,e);
   v = min(v, CalVol(c,d,a,e));
   v = min(v, CalVol(a,b,d,e));
   v = min(v, CalVol(b,c,d,e));

   return(141.516 * v / (h * s) );
}

float CalPriQal(float4 a, float4 b, float4 c, float4 d, float4 e, float4 f)
{
   float h, s, v;

   h = CalLen(a,b)
     + CalLen(b,c)
     + CalLen(c,a)
     + CalLen(d,e)
     + CalLen(e,f)
     + CalLen(f,d)
     + CalLen(a,d)
     + CalLen(b,e)
     + CalLen(c,f);

   s = CalTriSrf(c,b,a)
     + CalTriSrf(d,e,f)
     + CalQadSrf(d,f,c,a)
     + CalQadSrf(a,d,e,b)
     + CalQadSrf(e,f,c,b);

   v =        CalVol(a,b,c,d);
   v = min(v, CalVol(e,c,d,f));
   v = min(v, CalVol(d,b,c,e));
   v = min(v, CalVol(a,b,c,f));
   v = min(v, CalVol(a,b,f,e));
   v = min(v, CalVol(a,e,f,d));
   v = min(v, CalVol(d,b,f,e));
   v = min(v, CalVol(f,a,b,d));
   v = min(v, CalVol(c,a,b,e));
   v = min(v, CalVol(c,a,e,d));
   v = min(v, CalVol(e,c,a,f));
   v = min(v, CalVol(b,c,d,f));

   return(98.3538 * v / (h * s) );
}

float CalHexQal(  float4 a, float4 b, float4 c, float4 d,
                  float4 e, float4 f, float4 g, float4 h )
{
   float l, s, v;

   l = CalLen(d,c)
     + CalLen(a,b)
     + CalLen(e,f)
     + CalLen(h,g)
     + CalLen(d,h)
     + CalLen(c,g)
     + CalLen(b,f)
     + CalLen(a,e)
     + CalLen(d,a)
     + CalLen(h,e)
     + CalLen(g,f)
     + CalLen(c,b);

   s = CalQadSrf(d,a,e,h)
     + CalQadSrf(g,f,b,c)
     + CalQadSrf(d,c,b,a)
     + CalQadSrf(e,f,g,h)
     + CalQadSrf(d,h,g,c)
     + CalQadSrf(b,f,e,a);

   v =        CalVol(a,b,d,e);
   v = min(v, CalVol(b,c,d,g));
   v = min(v, CalVol(e,d,h,g));
   v = min(v, CalVol(e,f,b,g));
   v = min(v, CalVol(e,b,d,g));
   v = min(v, CalVol(e,f,a,h));
   v = min(v, CalVol(h,f,c,g));
   v = min(v, CalVol(a,c,d,h));
   v = min(v, CalVol(a,b,c,f));
   v = min(v, CalVol(a,f,c,h));

   return(72. * v / (l * s) );
}

void MulMatVec(double16 a, double4 x, double4 b)
{
   x.s0 += a.s0 * b.s0 + a.s1 * b.s1 + a.s2 * b.s2 + a.s3 * b.s3;
   x.s1 += a.s4 * b.s0 + a.s5 * b.s1 + a.s6 * b.s2 + a.s7 * b.s3;
   x.s2 += a.s8 * b.s0 + a.s9 * b.s1 + a.sa * b.s2 + a.sb * b.s3;
   x.s3 += a.sc * b.s0 + a.sd * b.s1 + a.se * b.s2 + a.sf * b.s3;
}
