EdgesDependenciesLoop: EdgesDependenciesLoop.c EdgesDependenciesLoop.cl
	../../utils/cl2h EdgesDependenciesLoop.cl EdgesDependenciesLoop.h EdgesDependenciesLoop
	$(CC) -O EdgesDependenciesLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
