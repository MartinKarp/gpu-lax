program reactivetube

include 'data'
real t, state(4,n), f(4,n+1), w(4,n)
integer step

t = 0

!print *, cp, bc_a_lbc_rho_l,bc_p_l,bc_u_l,bc_y_l,bc_rho_r,bc_p_r,bc_u_r,bc_y_r,bc_t_l,bc_a_l,bc_t_r,bc_a_r,c_char, n
do while (t < t_end)
    step = step + 1
    t = t + dt
    call roe(state,f);
end do

end

subroutine lax(state, f)
    real state(4,n), f(4,n+1)
    get_laxf(f,state)
    state(:,2:n-1) = 0.5*( state(:,3:n) + state(:,1:n-2) ) - 0.5*dt/dx * ( f(:,3:n) - f(:,1:n-2) ) + 0.5 * dt * w(:,2:n-1)
    end
    set_bc(state)
return
end
subroutine get_laxf(f,state):
    include 'data'
    real f(4,n+1), state(4,n)
    f(1,1:n) = state(2,:)
    f(2,1:n) = state(2,:)**2/state(1,:)
    state(3,:)/state(1,:)  - 0.5*u*u )/Cv;
    H_new   = E_new + R*T_new + Y_new*dH_f0;
end

subroutine roe(state,f)
    include 'data'
    real state(4,n), f(4,n+1)
    update_roef(f, state)
    update_w(w, state)
    state(:,2:n-1) =  state(:,2:n-1) - dt/dx * (f(:,2:n-1) - f(:,1:n-2))  + dt*w(:,2:n-1);
    set_bc(state)
return
end


subroutine update_roef(f, state)
    include 'data'
    for j = 1 : data.n_points-1
        % Left state is j  , Right State is j+1
        sq_rhoL = sqrt( data.rho(1,n-1) )
        sq_rhoR = sqrt( data.rho(2,n) )
        RoeAve_rho = sq_rhoL*sq_rhoR ;
        RoeAve_H   = ((data.H(1:n-1) - data.Y(1:n-1) * dH_f0)*sq_rhoL + (data.H(j+1) - data.Y(j+1) * dH_f0)*sq_rhoR) / (sq_rhoL + sq_rhoR) ;
        RoeAve_u   = (data.u(1:n-1)*sq_rhoL + data.u(j+1)*sq_rhoR) / (sq_rhoL + sq_rhoR) ;
        RoeAve_Y   = (data.Y(1:n-1)*sq_rhoL + data.Y(j+1)*sq_rhoR) / (sq_rhoL + sq_rhoR) ;
        RoeAve_c   = sqrt( (gamma-1)*(RoeAve_H - 0.5*RoeAve_u^2 ) );

        diff_P = data.P(j+1)-data.P(j);
        diff_u = data.u(j+1)-data.u(j);
        diff_rho= data.rho(j+1)-data.rho(j);
        %diff_Y

        % beta = [T^{-1}] diff(U)
        beta(1) =  0.5/ (RoeAve_c^2) * ( diff_P - RoeAve_c * RoeAve_rho * diff_u) ;
        beta(2) =  diff_rho-diff_P/ (RoeAve_c^2);
        beta(3) =  0.5/ (RoeAve_c^2) * ( diff_P + RoeAve_c * RoeAve_rho * diff_u);
        % absolute value of 3 eigenvalues of A_hat
        lambda(1) = ( RoeAve_u - RoeAve_c ) ;
        lambda(2) = ( RoeAve_u            ) ;
        lambda(3) = ( RoeAve_u + RoeAve_c ) ;

        lambda = abs(lambda);
        % Roe entropy fix (it will be disabled if eps is set too small)
        for k = 1 : 3
            if lambda(k) < eps
                lambda(k) = ( lambda(k)*lambda(k) + eps^2 ) / eps/2;
            end
        end

        % eigenvalue matrix T(3x3) is here!
        T(1,1) = 1 ;                T(1,2) = 1;         T(1,3) =1;
        T(2,1) = RoeAve_u-RoeAve_c ; T(2,2) = RoeAve_u; T(2,3) =RoeAve_u+RoeAve_c;
        T(3,1) = RoeAve_H-RoeAve_u*RoeAve_c ; T(3,2) = 0.5*RoeAve_u^2; T(3,3) =RoeAve_H+RoeAve_u*RoeAve_c ;
        %T(4,:) = T(2,:);
        F_L = get_F(j,data);
        F_R = get_F(j+1,data);
        RoeFlux = 0.5* (  F_L(1:3) + F_R(1:3) -   ( T*(lambda.*beta)' )' );
        rhouH_c_avg = 0.5 * dH_f0 * (data.rho(j) * data.Y(j) * data.u(j)+ data.rho(j+1) * data.Y(j+1) * data.u(j+1));
        RoeFlux(3) = RoeFlux(3) - rhouH_c_avg;
        if RoeAve_u > 0
            RoeFlux(3) = RoeFlux(3) + dH_f0 * data.rho(j) * data.Y(j) * data.u(j);
            RoeFlux(4) = data.u(j) * data.rho(j) * data.Y(j);
        else
            RoeFlux(3) = RoeFlux(3) + dH_f0 * data.rho(j+1) * data.Y(j+1) * data.u(j+1);
            RoeFlux(4) = data.u(j+1) * data.rho(j+1) * data.Y(j+1);
        end

        for i = 1: 4
            data.F_jhalf(j,i)=  RoeFlux(i);
        end
    end
end
