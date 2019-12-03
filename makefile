
F = gfortran
CFLAGS = -lOpenCL

all:
	${F} ${CFLAGS} -o reactivetube reactivetube.f95
clean:
	rm *.o
