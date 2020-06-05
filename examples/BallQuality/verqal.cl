{
   vq = tq[0];

   for(int i=1;i<VerTriDeg;i++)
      vq = min(vq, tq[i]);
}
