TrianglesBasicLoop: TrianglesBasicLoop.c TrianglesBasicLoop.cl
	../../utils/cl2h TrianglesBasicLoop.cl TrianglesBasicLoop.h TrianglesBasicLoop
	$(CC) -O TrianglesBasicLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
