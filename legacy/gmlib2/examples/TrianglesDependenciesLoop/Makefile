TrianglesDependenciesLoop: TrianglesDependenciesLoop.c TrianglesDependenciesLoop.cl
	../../utils/cl2h TrianglesDependenciesLoop.cl TrianglesDependenciesLoop.h TrianglesDependenciesLoop
	$(CC) -O TrianglesDependenciesLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
