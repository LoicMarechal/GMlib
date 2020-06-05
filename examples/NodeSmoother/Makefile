CC     = clang
CFLAGS = -Ofast -I . -I ${HOME}/include -DWITH_LIBMESHB
LIBS   = ${HOME}/lib/${ARCHI}/gmlib3.a  ${HOME}/lib/${ARCHI}/libmeshb7.a

# Define OpenCL compiling implicit rules
%.h : %.cl
	cl2h $< $@ `basename $@ .h`

NodeSmoothing: $(LIBS) NodeSmoothing.c parameters.h gather.h quality.h scatter.h
	$(CC) $(CFLAGS) NodeSmoothing.c  $(LIBS) -o $@ -lm -framework OPENCL

clean:
	rm  NodeSmoothing parameters.h gather.h quality.h scatter.h
