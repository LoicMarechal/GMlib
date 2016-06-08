TetrahedraBasicLoop: TetrahedraBasicLoop.c TetrahedraBasicLoop.cl
	../../utils/cl2h TetrahedraBasicLoop.cl TetrahedraBasicLoop.h TetrahedraBasicLoop
	$(CC) -O TetrahedraBasicLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
