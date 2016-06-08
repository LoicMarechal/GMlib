QuadsBasicLoop: QuadsBasicLoop.c QuadsBasicLoop.cl
	../../utils/cl2h QuadsBasicLoop.cl QuadsBasicLoop.h QuadsBasicLoop
	$(CC) -O QuadsBasicLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
