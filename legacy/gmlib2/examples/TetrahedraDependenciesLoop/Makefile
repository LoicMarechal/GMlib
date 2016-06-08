TetrahedraDependenciesLoop: TetrahedraDependenciesLoop.c TetrahedraDependenciesLoop.cl
	../../utils/cl2h TetrahedraDependenciesLoop.cl TetrahedraDependenciesLoop.h TetrahedraDependenciesLoop
	$(CC) -O TetrahedraDependenciesLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
