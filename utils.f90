module utils
    use ISO_C_BINDING
    implicit none

contains
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

end module utils
