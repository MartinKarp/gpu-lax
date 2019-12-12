
F = gfortran
CL_LIB = /usr/lib64
FFLAGS = -lOpenCL -O3 -fcheck=all -fbacktrace -fopenmp

OUT		= reactivetube

OBJS = reactivetube.f95
OBJS2 = reactivetube_mp.f95
OBJS3 = reactivetube_opencl.f95


all:
	${F} ${FFLAGS} -o ${OUT} ${OBJS}
omp:
	${F} ${FFLAGS} -o ${OUT} ${OBJS2}
opencl: clfortran clroutines
	${F} -L${CL_LIB} ${FFLAGS} -o ${OUT} ${OBJS3} clroutines.o
clean:
	rm *.o
clfortran:
	$(F) -c clfortran.f90
clroutines:
	$(F) -c clroutines.f90

run: all
	./${OUT}
