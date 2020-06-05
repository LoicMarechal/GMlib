CC     = clang
CFLAGS = -Ofast -I . -I ${HOME}/include -DWITH_LIBMESHB
LIBS   = ${HOME}/lib/${ARCHI}/gmlib3.a  ${HOME}/lib/${ARCHI}/libmeshb7.a

# Define OpenCL compiling implicit rules
%.h : %.cl
	cl2h $< $@ `basename $@ .h`

MeshQuality: $(LIBS) MeshQuality.c parameters.h compute_quality.h
	$(CC) $(CFLAGS) MeshQuality.c $(LIBS) -o $@ -lm -framework OPENCL

clean:
	rm MeshQuality	compute_quality.h parameters.h
