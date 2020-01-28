program reactivetube
use clfortran
use clroutines
use ISO_C_BINDING
use utils
implicit none
include 'data'

real :: t, start, time, u(n), temp(n), maxspeed, flops
integer :: iter, count, rate
real, target :: dt, cl_dx
real, allocatable, target :: state(:,:), w(:,:), state_t(:,:), w_t(:,:)
!OpenCl variables
integer(c_size_t) :: byte_size, ret
integer(c_int32_t) :: err
character(len=1024) :: options, kernel_name1, kernel_name2
character(len=1, kind=c_char),allocatable :: kernel_str(:)

integer, target :: np
integer(c_size_t),target :: globalsize,localsize
integer(c_intptr_t), target :: cmd_queue, context, kernel1, kernel2, cl_state, cl_state_old, cl_w, cl_f
integer(c_intptr_t), allocatable, target :: platform_ids(:), device_ids(:)


allocate(state(1:n,1:4))
allocate(w(1:n,1:4))

allocate(state_t(1:4,1:n))
allocate(w_t(1:4,1:n))


dt = cfl * dx / char_c
cl_dx = dx
iter = 0
t = 0
w = 0
np = n

kernel_name1='lax'
kernel_name2='get_fw'
options = '-Werror'
byte_size = 8_8 * int(4 * n,8)
print *, byte_size, 4 * n *8

! state = [rho rho*u rho*E Y]
!print *, cp, bc_a_lbc_rho_l,bc_p_l,bc_u_l,bc_y_l,bc_rho_r,bc_p_r,bc_u_r,bc_y_r,bc_t_l,bc_a_l,bc_t_r,bc_a_r,c_char, n
call init(state)
call create_device_context(platform_ids, device_ids, context, cmd_queue)
call query_platform_info(platform_ids(1))
call read_file('lax_trans.cl', kernel_str)
call init_kernel(kernel1, context, platform_ids, device_ids,kernel_name1, kernel_str, options)
call init_kernel(kernel2, context, platform_ids, device_ids,kernel_name2, kernel_str, options)
!Allocate memory
cl_state = clCreateBuffer(context, CL_MEM_READ_WRITE,byte_size,C_NULL_PTR, err)
if (err.ne.0) stop 'clCreateBuffer'
cl_state_old = clCreateBuffer(context, CL_MEM_READ_WRITE,byte_size,C_NULL_PTR, err)
if (err.ne.0) stop 'clCreateBuffer'
cl_w = clCreateBuffer(context, CL_MEM_READ_WRITE,byte_size,C_NULL_PTR, err)
if (err.ne.0) stop 'clCreateBuffer'
cl_f = clCreateBuffer(context, CL_MEM_READ_WRITE,byte_size,C_NULL_PTR, err)
if (err.ne.0) stop 'clCreateBuffer'

w_t = transpose(w)
state_t = transpose(state)

!Copy data, 0_8 is a zero of kind=8
err=clEnqueueWriteBuffer(cmd_queue,cl_w,CL_TRUE,0_8,byte_size,C_LOC(w_t), 0,C_NULL_PTR,C_NULL_PTR)
if (err.ne.0) stop 'clEnqueueWriteBuffer'
err=clEnqueueWriteBuffer(cmd_queue,cl_state,CL_TRUE,0_8,byte_size,C_LOC(state_t), 0,C_NULL_PTR,C_NULL_PTR)
if (err.ne.0) stop 'clEnqueueWriteBuffer'

!Set kernel arguments
err=clSetKernelArg(kernel1,0,sizeof(dt),C_LOC(dt))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel1,1,sizeof(dx),C_LOC(cl_dx))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel1,2,sizeof(np),C_LOC(np))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel1,3,sizeof(cl_state),C_LOC(cl_state))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel1,4,sizeof(cl_state_old),C_LOC(cl_state_old))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel1,5,sizeof(cl_f),C_LOC(cl_f))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel1,6,sizeof(cl_w),C_LOC(cl_w))
if (err.ne.0) stop 'clSetKernelArg'

!Set kernel arguments
err=clSetKernelArg(kernel2,0,sizeof(np),C_LOC(np))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel2,1,sizeof(cl_state),C_LOC(cl_state))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel2,2,sizeof(cl_state_old),C_LOC(cl_state_old))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel2,3,sizeof(cl_f),C_LOC(cl_f))
if (err.ne.0) stop 'clSetKernelArg'
err=clSetKernelArg(kernel2,4,sizeof(cl_w),C_LOC(cl_w))
if (err.ne.0) stop 'clSetKernelArg'


!get the local size for the kernel
err=clGetKernelWorkGroupInfo(kernel1,device_ids(1), CL_KERNEL_WORK_GROUP_SIZE,sizeof(localsize), C_LOC(localsize),ret)
if (err.ne.0) stop 'clGetKernelWorkGroupInfo'
globalsize=int(n,8)
if (mod(globalsize,localsize).ne.0) then
    globalsize = globalsize+localsize-mod(globalsize,localsize)
end if


call system_clock(count, rate)
start = count / real(rate)
do while (t < t_end)
    iter = iter + 1
    t = t + dt
    err=clEnqueueNDRangeKernel(cmd_queue,kernel2,1,C_NULL_PTR,C_LOC(globalsize),C_LOC(localsize),0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueNDRangeKernel'
    !err=clFinish(cmd_queue)
    !if (err.ne.0) stop 'clFinish'
    err=clEnqueueNDRangeKernel(cmd_queue,kernel1,1,C_NULL_PTR,C_LOC(globalsize),C_LOC(localsize),0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueNDRangeKernel'
    !err=clFinish(cmd_queue)
    !if (err.ne.0) stop 'clFinish'
    !if  (modulo(iter , update) .EQ. 0) then
    !    maxspeed= maxval( abs(u * ( 1 / abs( u / sqrt( gamma * r * temp)) + 1))  );
    !    dt = cfl * dx / maxspeed;
    !    if (verbose) then
    !        print *,"Time = ", t , "dt = ", dt
    !    end if
    !end if
end do
call system_clock(count, rate)

err=clEnqueueNDRangeKernel(cmd_queue,kernel2,1,C_NULL_PTR,C_LOC(globalsize),C_LOC(localsize),0,C_NULL_PTR,C_NULL_PTR)
if (err.ne.0) stop 'clEnqueueNDRangeKernel'
err=clFinish(cmd_queue)
if (err.ne.0) stop 'clFinish'
err = clEnqueueReadBuffer(cmd_queue,cl_state,CL_TRUE,0_8,byte_size,C_LOC(state_t),0,C_NULL_PTR,C_NULL_PTR)
if (err.ne.0) stop 'clEnqueueReadBuffer'

state = transpose(state_t)
time = count / real(rate) - start
flops = cost(n,iter, update)
err = clReleaseCommandQueue(cmd_queue)
err = clReleaseContext(context)

print *,'Number of points: ', n
print *,'Iterations: ', iter
print *,'Time/s:', time
print *,'Flops/s: ', flops/time

call write_out(state, t, dt)

end program reactivetube
