# gpu-lax
This repository was part of the project course in computer science at LTH, Lund University and revolved around porting a basic Lax-scheme solver for the 1D-reactive Euler equation from MATLAB to Fortran and in particular GPUs.

It made use of OpenCL for the GPU acceleration and I think the main take away from this project is how to use OpenCL in Fortran which is not supported natively. This was done by using the clfortran API hosted at https://github.com/cass-support/clfortran

If one would like to build this repo it is done by issuing make in the main folder. The program can then be run by executing ./reactivetube

If one would like to build a OpenMP version instead this is done by writing $make omp$ and the OpenCL version is built with $make opencl$. Depending on what number of cl platforms you have connected you may need to change the parameter iplatform in the data file to match what you need. The executable is always $./reactivetube$.

If one would like to change parameters or such, look in the data file. If someone else would like to use OpenCL in fortran look in clroutines.f90 in particular.
