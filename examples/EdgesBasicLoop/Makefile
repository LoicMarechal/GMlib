EdgesBasicLoop: EdgesBasicLoop.c EdgesBasicLoop.cl
	../../utils/cl2h EdgesBasicLoop.cl EdgesBasicLoop.h EdgesBasicLoop
	$(CC) -O EdgesBasicLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
