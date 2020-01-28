
F = gfortran
CL_LIB = /usr/lib64
FFLAGS = -lOpenCL -O3 -fcheck=all -fbacktrace -fopenmp

OUT		= reactivetube

OBJS =
OBJS2 = reactivetube_mp.f95
OBJS3 = reactivetube_opencl.f95


all: utils
	${F} ${FFLAGS} -o ${OUT} reactivetube.f95 utils.o
omp: utils
	${F} ${FFLAGS} -o ${OUT} reactivetube_omp.f95 utils.o
opencl: clfortran clroutines utils
	${F} -L${CL_LIB} ${FFLAGS} -o ${OUT} reactivetube_opencl.f95 clroutines.o utils.o
trans: utils
	${F} ${FFLAGS} -o ${OUT} reactivetube_trans.f95 utils.o
clean:
	rm *.o
	rm *.mod
	rm ${OUT}
clfortran:
	$(F) -c clfortran.f90
clroutines:
	$(F) -c clroutines.f90
utils:
	$(F) -c utils.f90

run: all
	./${OUT}
