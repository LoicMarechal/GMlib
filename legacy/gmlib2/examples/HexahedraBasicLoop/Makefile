HexahedraBasicLoop: HexahedraBasicLoop.c HexahedraBasicLoop.cl
	../../utils/cl2h HexahedraBasicLoop.cl HexahedraBasicLoop.h HexahedraBasicLoop
	$(CC) -O HexahedraBasicLoop.c ../libmesh5/libmesh5.c ../../sources/gmlib2.c -I ../libmesh5 -I ../../sources -o $@ -lm  -framework OPENCL
