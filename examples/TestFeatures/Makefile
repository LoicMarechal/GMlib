CC     = clang
CFLAGS = -Ofast -I . -I ${HOME}/include -DWITH_LIBMESHB
LIBS   = ${HOME}/lib/${ARCHI}/gmlib3.a  ${HOME}/lib/${ARCHI}/libmeshb7.a

# Define OpenCL compiling implicit rules
%.h : %.cl
	cl2h $< $@ `basename $@ .h`

TestFeatures: $(LIBS) TestFeatures.c parameters.h double_precision.h downlink.h neighbours.h uplink.h
	$(CC) $(CFLAGS) TestFeatures.c $(LIBS) -o $@ -lm -framework OPENCL

clean:
	rm TestFeatures parameters.h double_precision.h downlink.h neighbours.h uplink.h 
