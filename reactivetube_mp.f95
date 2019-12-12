program reactivetube

include 'data'
real :: dt, t,  start, time, maxspeed, flops
real, allocatable ::state(:,:), f(:,:), w(:,:),u(:),m(:),e(:), temp(:)
integer :: iter, count, rate
real :: cost
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
        e = merge(0.,1.,u < 1e-10)
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


subroutine get_fw(state,f,w)
    include 'data'
    real f(n,4), state(n,4), u(n), e(n), temp(n), h(n), p(n), w(n,4)
    !$omp parallel workshare
    u = state(:,2)/state(:,1)
    e = state(:,3)/state(:,1)
    temp = (e - state(:,4)*dh_f0 - 0.5*u**2)/cv;
    h = e + r*temp;
    p = temp * state(:,1) * r

    f(:n,1) = state(:,2)
    f(:n,2) = state(:,2) * u + p
    f(:n,3) = state(:,2) * h
    f(:n,4) = state(:,2) * state(:,4)
    !w(2:n-1,4) = 0.5 * (state(1:n-2,1) + state(3:n,1)) / tau_c * &
    !(1.0 - 0.5 * (state(1:n-2,4) + state(3:n,4)))*exp(-2 * t_a/(temp(1:n-2) + temp(3:n)))
    w(2:n-1,4) = state(2:n-1,1) / tau_c * &
    (1.0 - 0.5* (state(1:n-2,4) + state(3:n,4)))*exp(-3 * t_a/(temp(1:n-2) + temp(3:n)))
    !$omp end parallel workshare
end subroutine get_fw

subroutine set_bc(state)
    include 'data'
    real state(n,4)
    state(1,:) = state(2,:)
    state(n,:) = state(n-1,:)
end subroutine set_bc

subroutine init(state)
    include 'data'
    real state(n,4), e_l, h_l, e_r, h_r
    integer mid
    mid = int(n/2)

    h_l = cp*bc_t_l+0.5 * bc_u_l**2 + dh_f0 * bc_y_l;
    e_l  = h_l - r * bc_t_l
    state(:mid,1) = bc_rho_l
    state(:mid,2) = bc_rho_l * bc_u_l
    state(:mid,3) = bc_rho_l * e_l
    state(:mid,4) = bc_rho_l * bc_y_l

    h_r = cp*bc_t_r+0.5 * bc_u_r**2 + dh_f0 * bc_y_r;
    e_r  = h_r - r * bc_t_r
    state(mid:,1) = bc_rho_r
    state(mid:,2) = bc_rho_r * bc_u_r
    state(mid:,3) = bc_rho_r * e_r
    state(mid:,4) = bc_rho_r * bc_y_r
    !print *,state
end subroutine init

subroutine write_out(state,t, dt)
    include 'data'
    real state(n,4), t, dt
    open(13, file=out, action="write", status="replace", form="unformatted")
    print *, 'Writing state to output'
    !print *, dx, dt, n, state(n/3,:)
    write(13) n
    write(13) domainlength, t, dt
    write(13) state
    close(13)
end subroutine write_out

function cost(n, iter, update)
    integer :: div, mult, add, exp, updateflop
    integer, intent(in) :: n, iter, update
    real :: cost
    div = 4 * n
    mult = 23 * n
    add = 22 * n
    exp = 1 * n
    updateflop = 8 * n / update
    cost = real(iter) * (div + mult + add + exp + updateflop)
end function cost
