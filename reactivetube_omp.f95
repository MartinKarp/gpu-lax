program reactivetube

use utils
include 'data'
real :: dt, t,  start, time, maxspeed, flops
real, allocatable ::state(:,:), f(:,:), w(:,:),u(:),m(:),e(:), temp(:)
integer :: iter, count, rate
call omp_set_num_threads(n_threads)

allocate(state(1:n,1:4))
allocate(w(1:n,1:4))
allocate(f(1:n,1:4))
allocate(u(1:n))
allocate(temp(1:n))
allocate(m(1:n))
allocate(e(1:n))

dt = cfl * dx / char_c
iter = 0
t = 0
w = 0.
flops = 0
call init(state)
! state = [rho rho*u rho*E Y]
!print *, cp, bc_a_lbc_rho_l,bc_p_l,bc_u_l,bc_y_l,bc_rho_r,bc_p_r,bc_u_r,bc_y_r,bc_t_l,bc_a_l,bc_t_r,bc_a_r,char_c, n
call system_clock(count, rate)
start = count / real(rate)

do while (t < t_end)
    iter = iter + 1
    t = t + dt
    call lax(state,f,w, dt, u, temp, e)
    !dt = something something
    if  (modulo(iter , update) .EQ. 0) then
        e = merge(0.,1.,u < min_u)
        u = u * e
        m = 1 / (u / sqrt( gamma * r * temp))
        maxspeed= maxval( abs(pack(u,m < HUGE(dx)) * (abs(pack(m,m < HUGE(dx))) + 1)));
        !print *, maxspeed
        !maxspeed = max(maxval(w(:,4)), maxspeed)
        dt = cfl * dx / maxspeed
        !if( dt < min_dt) dt = min_dt
        if (verbose) then
            print *,"Time = ", t , "dt = ", dt, "MaxSpeed = ", maxspeed
        end if
    end if
end do

call system_clock(count, rate)
time = count / real(rate) - start
flops = cost(n,iter, update)
print *,'Number of points: ', n
print *,'Iterations: ', iter
print *,'n_threads:', n_threads
print *,'Time/s:', time
print *,'Flops/s: ', flops/time
call write_out(state, t, dt)

end program reactivetube

subroutine lax(state, f,w, dt, u, temp, e)
    use utils
    include 'data'
    real, dimension(n,4) :: state, f, w
    real, dimension(n) :: u, e, temp
    real :: dt
    !$omp parallel workshare
    u = state(:,2)/state(:,1)
    e = state(:,3)/state(:,1)
    temp = (e - state(:,4)*dh_f0 - 0.5*u**2)/cv;

    f(:,1) = state(:,2)
    f(:,2) = state(:,2) * u + temp * state(:,1) * r
    f(:,3) = state(:,2) * (e + r * temp)
    f(:,4) = state(:,2) * state(:,4)
    w(2:n-1,4) = state(2:n-1,1) / tau_c * &
    (1.0 - 0.5* (state(1:n-2,4) + state(3:n,4)))*exp(-2 * t_a/(temp(1:n-2) + temp(3:n)))

    !state_s = state
    !$omp end parallel workshare
    !$omp parallel do
    do j = 1, 4
        state(2:n-1,j) = 0.5*( state(3:n,j) + state(1:n-2,j) ) - 0.5*dt/dx * ( f(3:n,j) - f(1:n-2,j) ) + 0.5 * dt * w(2:n-1,j)
    end do
    !$omp end parallel do
    e = merge(0.,1.,state(:,4) > 1)
    state(:,4) = state(:,4) * e + (1-e)
    e = merge(0.,1.,state(:,4) < 0)
    state(:,4) = state(:,4) * e
    !state(2:n-1,:) = 0.5*( state(3:n,:) + state(1:n-2,:) ) - 0.5*dt/dx * ( f(3:n,:) - f(1:n-2,:) ) + 0.5 * dt * w(2:n-1,:)
    call set_bc(state)
end subroutine lax
