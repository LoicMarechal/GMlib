CC     = clang
CFLAGS = -Ofast -I . -I ${HOME}/include -DWITH_LIBMESHB
LIBS   = ${HOME}/lib/${ARCHI}/gmlib3.a  ${HOME}/lib/${ARCHI}/libmeshb7.a

# Define OpenCL compiling implicit rules
%.h : %.cl
	cl2h $< $@ `basename $@ .h`

HeatCellCentered: $(LIBS) HeatCellCentered.c dt.h grd_ext.h ini_tet.h sol_ext.h flx_bal.h grd_tet.h param.h tim_int.h
	$(CC) $(CFLAGS) HeatCellCentered.c $(LIBS) -o $@ -lm -framework OPENCL

clean:
	rm HeatCellCentered dt.h grd_ext.h ini_tet.h sol_ext.h flx_bal.h grd_tet.h param.h tim_int.h
