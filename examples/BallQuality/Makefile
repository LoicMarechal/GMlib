CC     = clang
CFLAGS = -Ofast -I . -I ${HOME}/include -DWITH_LIBMESHB
LIBS   = ${HOME}/lib/${ARCHI}/gmlib3.a  ${HOME}/lib/${ARCHI}/libmeshb7.a

# Define OpenCL compiling implicit rules
%.h : %.cl
	cl2h $< $@ `basename $@ .h`

BallQuality: $(LIBS) BallQuality.c parameters.h triqal.h verqal.h
	$(CC) $(CFLAGS) BallQuality.c  $(LIBS) -o $@ -lm -framework OPENCL

clean:
	rm BallQuality parameters.h triqal.h verqal.h
