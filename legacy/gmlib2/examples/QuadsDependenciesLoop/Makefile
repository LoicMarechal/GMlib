QuadsDependenciesLoop: QuadsDependenciesLoop.c QuadsDependenciesLoop.cl
	../../utils/cl2h QuadsDependenciesLoop.cl QuadsDependenciesLoop.h QuadsDependenciesLoop
	$(CC) -O QuadsDependenciesLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
