
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code demonstrates a simple implementaion of 6 numerical schemes
%  to solve the unstead 1D Euler equations.
%
%  This code can be directly applied to study a 1D shock-tube problem just
%  by imposing a jump intial conditions in the middle of the domain.
%
%  The for solving 1D shock-tube problem
%  togther
% with analytical solution
%
%  LTH course: MVKN70 (Advanced CFD ...)
%       Tutor: Rixin Yu
%              rixin.yu@energy.lth.se
%              2018-08-30
%
%   Further developed by Martin Karp to compare and solve for 1D reactive flow with
%   one variable of state, the massfraction of fuel Y.
%   tpi15mka@student.lu.se
%   2019-12-10
%
%   There might be some bugs, but it should be runnable pretty easily.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This code is for numerically solving the unsteady 1D Euler equation:
%          d_t(U) + d_x(F) = 0;
%   with
%         U = [rho,       rho*u,   rho*E],
%         F = [rho*u, rho*u^2+P,   rho*u*H],
%   with relations:
%  (1) H  =  E + P/rho = Cp*T
%  (2)       E = Cv*T + u*u/2
%  (3)           P = rho*R*T
%
%  the above variables mean:
%   rho : density
%   P   : pressure
%   u   : velocity
%   H   : total enthalpy (include kinetic energy)
%   E   : total energy   (include kinetic energy)
%   Cp  : specific heat capacity at constant pressure
%   Cv  : specific heat capacity at constant volume
%   R   : universal gas constant
%
%   Note!!!! in this implementaion, we choose the following core-set "V"
%   formed by 3 unknowns in primative form
%          V =[rho,   u,   H],
%   In principle, any other quantie(such as P,T,.., etc.) can be obtained by express it
%    as a function of the three symbols inside V.
%
%   After setting intial conditions, the solver enters a main loop doing explicit
%   time-advancement. The time-advacement means to update the values of three terms
%   inside the core-set V over all spatial grids from t^n to t^n+1, as
%   described in the following steps:
%
%   Step 1:The core-set V are known at t^n (including t=0);
%   Step 2:
%          Evalute all "relevant terms" given in any chosen numerical schemes using
%          the values of V given at t^n , such "relevant terms" includes (i) the
%          three conservative variable in U, (ii) the three flux terms in F and (iii) etc.
%   Step 3:
%          Update U to  t^n+1 as:
%
%           U(j, t^n+1)  =  SomeSchemeDependentFunction( V ;   j-2, j-1, j, j+1, j+2 , t^n)
%   Step 4:
%          Translate the three variable U(t^n+1) to V(t^n+1),  then enter
%          next loop
%


%   During the intermeidate computaion, it is often to switch between V , U , F .
%   The nominal values used for air at 300 K are Cp = 1.00 kJ/kg.K, Cv = 0.718 kJ/kg.K,, and gamma = 1.4.
% The SI unit based on [kg, m, s , Kelvin] is used, note the unit of [joule=kg*(m/s)^2] [pascal=kg/m/s^2]
global Cp Cv gamma R tau_c T_A dH_f0;
Cp    = 1000 ; % Do not Change
gamma = 1.4;   % Do not Change
Cv= Cp/gamma;
R = Cp - Cv;
tau_c  = 0.0000009/(3*340);
T_A = 2000 * 6;
dH_f0 = -700 * Cp;
%%%%%%%%%%%%%%%%%%%%%%%%




BC.type = 'ShockTube';   %BC.type = 'NozzleFlow';

if strcmp( BC.type, 'ShockTube')
    n_points =1000 * 0.2; % Total Number of grid cells

    %%%%%%%%%%%%%%%%%%
    % here you can set initial condition for shocktube problem
    % _L for left side, _R for right side)
     BC.rho_L = 1.225*10/2;            % kg/m^3
     BC.P_L   = 100000*10;        % pascal
     BC.u_L   = 0;
     BC.Y_L   = 0;

     BC.rho_R = 1.225/2;         % kg/m^3
     BC.P_R   = 100000*1;        % pascal
     BC.u_R   = 0;                % m/s
     BC.Y_R   = 0;
    %%%%%%%%%%%%%%%%%%

%    BC.rho_L = 1.225/4*10;    % kg/m^3
%    BC.P_L   = 100000*10 ;    % pascal
%    BC.u_L   = 0;
%    BC.Y_L   = 0;
%
%    BC.rho_R = 1.225/4 ;      % kg/m^3
%    BC.P_R   = 100000*1;      % pascal
%    BC.u_R   = 0;              % m/s
%    BC.Y_R   = 0;

    BC.T_L = BC.P_L/(BC.rho_L*R);
    BC.a_L = sqrt( gamma * R * BC.T_L);

    BC.T_R = BC.P_R/(BC.rho_R*R);
    BC.a_R = sqrt( gamma * R * BC.T_R);

    c_CharacteriticSoundSpeed = max(BC.a_L, BC.a_R ) ;
end


CFL  = 0.5;

% The spatial-grid
DomainLength=  2; % total domain length [m]: better not to change
x = linspace(-DomainLength/2,DomainLength/2,n_points);  x=x';
dx = DomainLength/(n_points);  % cell size
%
data.x = x;
% Change the total time of the simulation here if you want longer time of simulations
if strcmp( BC.type, 'ShockTube')
  TotalSimulationTime = 1/1000 * 0.9; %DomainLength/c_CharacteriticSoundSpeed/3 *20 ;
end

dt = CFL *dx/c_CharacteriticSoundSpeed;  % time-step size
Num_T = round( TotalSimulationTime/dt ); % total number of time stepping


close all;
AllResults = {};
data.n_points = n_points;

data = SetInitialConditions(data, BC);
% plot prepartion
figure(1);
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6*6 4*4]);
set(gcf,'Units','inches','Position',[1 1 6*6 4*4]);
set(0, 'DefaultLineLineWidth', 0.5);

%  main loop of time-advancement
data.CurrentTime = 0;
data.W = zeros(data.n_points,4);
data2 = data;
TimeStep = 0;
tic
%pause_plot(x,data,data2,'Roe and van Leer');

while data.CurrentTime < TotalSimulationTime
    TimeStep = TimeStep + 1;
    data.CurrentTime = data.CurrentTime  + dt;
    data = lax2(data, dt, dx)
    %data  = Lax(data,dt,dx,n_points,BC);
    %data  = StegerWarming(data,dt,dx,n_points,BC);
    %data  = VanLeer(data,dt,dx,n_points,BC);
    %data = AUSM(data,dt,dx,n_points,BC);
    %data = Roe(data,dt,dx,n_points,BC);


    %  Plot every 10 time step togher with analytical shock-tube solutions
    %if mod(TimeStep , 10 ) == 0
    %    %pause_plot(x,data,data,'Roe and van Leer');
    %    %time = data.CurrentTime * 1000
    %    toc
    %end

for j = 2:n_points-1
    data = U_to_V(data.state(j,:),j,data);
end
    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    pause_plot(data,data,'Lax n = 200');
    if  mod(TimeStep , 20 ) == 0
        %here is to adpatively adjust time step only for nozzle flow !
        data.M = 1 ./ (data.u ./ sqrt( gamma * R * data.T))
        maxSpeed = max( abs(data.u .*( 1./abs(data.M) + 1))  );
        maxSpeed
        dt = CFL *dx/maxSpeed
        Progress = data.CurrentTime/TotalSimulationTime
        TimeStep
        toc
    end
end % end loop of time-advancement
en = toc;
costy = cost(n_points, TimeStep, 20)
costy/en
for j = 2:n_points-1
    data = U_to_V(data.state(j,:),j,data);
end

data=Set_BoundaryCondition(data,BC,dt,dx);
data=Update_PTEsM_AfterReset_V(data);
data.CurrentTime = data.CurrentTime + dt;

%AllResults{end+1} = { str_NameOfNumericalScheme, data};
pause_plot(data,data,'Roe and van Leer');
%if  size(AllResults,2 ) > 1
%    close all;
%    plot_summary(x,AllResults);
%end
%save('lax200.mat','data')
return;

function data= U_to_V22(U,data)
 global Cp Cv gamma R dH_f0;
  data.rho = U(1,:) ;
  data.u   = U(2,:)./U(1,:) ;
  Y        = U(4,:)./U(1,:);

  Y(Y > 1) = 1;
  Y(Y < 0) = 0;

  data.Y = Y;
  data.H   = gamma.*U(3,:)./U(1,:) -(gamma-1).*0.5.* (data.u).^2 - (gamma - 1) .* Y .* dH_f0;
end

function [data] = Update_PTEsM_AfterReset_V2(data)
    global Cp Cv gamma R dH_f0;
    rho = data.rho;  % V = [rho, u, H]
    u   = data.u;
    Y   = data.Y;
    H_c = Y * dH_f0;
    H   = data.H - H_c;          % total enthalpy
    T   = (H-0.5.*u.*u )./Cp ;
    E   = Cv .* T + 0.5 .* u .* u + H_c;       % total energy
    P   = rho.*R.*T;

    data.P = P;  % get P, T, E to ease later computation of U and E
    data.T = T;
    data.E = E;
    data.s = Cv*log( P./(rho).^gamma );
    data.M = u./sqrt( gamma .* R .* T);
end

% change here if you want other boundary conditions
function [data]= Set_BoundaryCondition(data, BC,dt,dx)
    if     strcmp( BC.type, 'ShockTube')
        data = BoundaryCondition_ZeroGradient(data,BC);
        %data = BoundaryCondition_Wall_zeroPressureGradient(data,BC,dt,dx);
        %data = BoundaryCondition_WallReflection(data,BC,dt,dx);
        %data = BoundaryCondition_OpenEnd(data,BC,dt,dx);
    end
end

function cost = cost(n, iter, update)
    div = 4 * n;
    mult = 23 * n;
    add = 22 * n;
    expo = 1 * n;
    updateflop = 8 * n / update;
    cost = iter * (div + mult + add + expo + updateflop);
end

function [data] = lax2(data, dt, dx)
    global Cp Cv gamma R dH_f0 tau_c T_A
    data.name = 'Lax';

    data.u = data.state(:,2)./data.state(:,1);
    data.E = data.state(:,3)./data.state(:,1);
    data.T = (data.E - data.state(:,4).*dH_f0 - 0.5.*data.u.^2)/Cv;
    data.f(:,1) = data.state(:,2);
    data.f(:,2) = data.state(:,2) .* data.u + data.T .* data.state(:,1) .* R;
    data.f(:,3) = data.state(:,2) .* (data.E + R*data.T);
    data.f(:,4) = data.state(:,2) .* data.state(:,4);

    data.W(2:data.n_points-1,4) = data.state(2:data.n_points-1,1) ./ tau_c .* (1.0 - 0.5.* (data.state(1:data.n_points-2,4) + data.state(3:data.n_points,4))).*exp(-2 .* T_A./(data.T(1:data.n_points-2) + data.T(3:data.n_points)));
    data.state(2:data.n_points-1,:) = 0.5.*( data.state(3:data.n_points,:) + data.state(1:data.n_points-2,:) ) - 0.5.*dt./dx .* ( data.f(3:data.n_points,:) - data.f(1:data.n_points-2,:) ) + 0.5 .* dt .* data.W(2:data.n_points-1,:);

    data.state(data.state(:,4) > 1,4) = 1;
    data.state(data.state(:,4) < 0,4) = 0;

    data.state(1,:) = data.state(2,:);
    data.state(data.n_points,:) = data.state(data.n_points-1,:);
end


function [data] = Lax(data,dt,dx,n_points,BC)
    data.name = 'Lax';
    data_save = data;
    for j= 2: n_points-1 % j: loop over spatial grid. j=1 and n_points delegates to the boundary conditions!
        % U = [rho rho*u rho*E]
        U = 0.5*( get_U(j+1,data_save) + get_U(j-1,data_save) ) - 0.5*dt/dx * ( get_F(j+1,data_save) - get_F(j-1,data_save) ) + 0.5 * dt * get_W(j,data_save);
        data = U_to_V(U,j,data);
    end

    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    data.CurrentTime = data.CurrentTime + dt;
end

function [data] = StegerWarming(data,dt,dx,n_points,BC)
    data.name = 'Steger-Warming';
    data = Update_FluxSplitting_AfterReset_V(data,2);
    for j= 2: n_points-1
        U =  get_U(j,data) - dt/dx * ( get_F_plus(j,data) - get_F_plus(j-1,data) +  get_F_minus(j+1,data) - get_F_minus(j,data)  ) + dt * get_W(j,data);
        data = U_to_V(U,j,data);
    end
    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    data=Update_FluxSplitting_AfterReset_V(data,2);
    data.CurrentTime = data.CurrentTime + dt;
end

function [data] = VanLeer(data,dt,dx,n_points,BC)
    data.name = 'van Leer';
    data = Update_FluxSplitting_AfterReset_V(data,1);
    for j= 2: n_points-1
        U =  get_U(j,data) - dt/dx * ( get_F_plus(j,data) - get_F_plus(j-1,data) +  get_F_minus(j+1,data) - get_F_minus(j,data)  )  + dt * get_W(j,data);
        data = U_to_V(U,j,data);
    end
    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    data = Update_FluxSplitting_AfterReset_V(data,1);
    data.CurrentTime = data.CurrentTime + dt;
end

function [data] = AUSM(data,dt,dx,n_points,BC)
    data.name = 'AUSM';
    data=Update_AUSM_Flux(data);
    for j= 2: n_points-1            % j loop for spatial grid
        U =  get_U(j,data) - dt/dx * ( get_F_jhalf(j,data) - get_F_jhalf(j-1,data) ) +dt*get_W(j,data);
        data = U_to_V(U,j,data);
    end

    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    data=Update_AUSM_Flux(data);
    data.CurrentTime = data.CurrentTime + dt;
end
function [data] = Roe(data,dt,dx,n_points,BC)
    data.name = 'Roe';
    data=Update_Roe_Flux(data);
    for j= 2: n_points-1            % j loop for spatial grid
        U =  get_U(j,data) - dt/dx * ( get_F_jhalf(j,data) - get_F_jhalf(j-1,data) )  + dt*get_W(j,data);
        data = U_to_V(U,j,data);
    end
    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    data=Update_Roe_Flux(data);
    data.CurrentTime = data.CurrentTime + dt;
end
function [data] = Roe2(data,dt,dx,n_points,BC)
    data.name = 'Roe';
    data=Update_Roe_Flux(data);
    U = zeros(4,n_points);
    F_jhalf = zeros(4,n_points );
    W = zeros(4,n_points);

    for j= 1: n_points-1            % j loop for spatial grid
        U(:,j) = get_U(j,data);
        F_jhalf(:,j) = get_F_jhalf(j,data);
        W(:,j) = get_W(j,data);
    end
    U(:,n_points) = get_U(n_points,data);

    %for j= 2: n_points-1            % j loop for spatial grid
    %    U =  get_U(j,data) - dt/dx * ( get_F_jhalf(j,data) - get_F_jhalf(j-1,data) )  + dt*get_W(j,data)
    %    data = U_to_V(U,j,data);;
    %end
    U(:,2:end-1) =  U(:,2:end-1) - dt/dx .* (F_jhalf(:,2:end-1) - F_jhalf(:,1:end-2))  + dt*W(:,2:end-1);
    data = U_to_V2(U,data);
    data=Set_BoundaryCondition(data,BC,dt,dx);
    data=Update_PTEsM_AfterReset_V(data);
    data=Update_Roe_Flux(data);
    data.CurrentTime = data.CurrentTime + dt;
end

function [data] = Update_Roe_Flux(data)
    global Cp Cv gamma R dH_f0;
    eps=100 ; %to enable entropy fix, set a large value for eps, say 100
    for j = 1 : data.n_points-1
        % Left data.state is j  , Right data.state is j+1
        sq_rhoL = sqrt( data.rho(j) );  sq_rhoR = sqrt( data.rho(j+1) );
        RoeAve_rho = sq_rhoL*sq_rhoR ;
        RoeAve_H   = ((data.H(j) - data.Y(j) * dH_f0)*sq_rhoL + (data.H(j+1) - data.Y(j+1) * dH_f0)*sq_rhoR) / (sq_rhoL + sq_rhoR) ;
        RoeAve_u   = (data.u(j)*sq_rhoL + data.u(j+1)*sq_rhoR) / (sq_rhoL + sq_rhoR) ;
        RoeAve_Y   = (data.Y(j)*sq_rhoL + data.Y(j+1)*sq_rhoR) / (sq_rhoL + sq_rhoR) ;
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

function [data] = Update_AUSM_Flux(data)
    global Cp Cv gamma R dH_f0;
    % Check the lecture notes
    % Calculate terms (M+, M- ; p+, p-, rho*c) at cell center j
    for j = 1 : data.n_points
        u = data.u(j);
        P = data.P(j);
        c = sqrt( gamma * R * data.T(j)) ;
        rhoc(j) = data.rho(j) * c;      % to be used in the followed loop
        M = u / c ;                     % Ma number
        if( abs (M) < 1) % subsonic
            p_plus(j)  = 0.5* P * (1 + M) ;
        else  %        supersoinc rightgoing vs  supersoinc leftgoing
            if  (M>1) p_plus(j)  = P ; else   p_plus(j)  = 0;   end
        end
        p_minus(j) = P- p_plus(j) ;

        if( abs (M) < 1) % subsonic
            M_plus(j)  =  0.25*  ( M+1)^2 ;
            M_minus(j) = -0.25*  ( M-1)^2 ;
        else  %     supersoinc rightgoing vs  supersoinc leftgoing
            if (M>1)    M_plus(j)  = M ;   else  M_plus(j)  = 0;       end
            M_minus(j) = M- M_plus(j);
        end
    end

    % here to compute flux at cell-interface of j+1/2
    for j = 1 : data.n_points-1
        % Left data.state is j  , Right data.state is j+1
        M_jhalf = M_plus(j) +  M_minus(j+1) ;

        %Francesco
        if( M_jhalf > 0 )
            FluxConv_jhalf = M_jhalf * [ rhoc(j) , rhoc(j)*data.u(j), rhoc(j)*data.H(j),  rhoc(j)*data.Y(j)];
        else %Francesco
            FluxConv_jhalf = M_jhalf * [ rhoc(j+1) , rhoc(j+1)*data.u(j+1),   rhoc(j+1)*data.H(j+1),  rhoc(j+1)*data.Y(j+1)];
        end

        % Combined flux due to convection- and pressure- contributions
        data.F_jhalf(j,1) = FluxConv_jhalf(1);
        data.F_jhalf(j,2) = FluxConv_jhalf(2) + ( p_plus(j)+p_minus(j+1) ) ;
        data.F_jhalf(j,3) = FluxConv_jhalf(3);
        %Francesco
        data.F_jhalf(j,4) = FluxConv_jhalf(4);
   end
end

function [data] = Update_FluxSplitting_AfterReset_V(data, k)
    for j = 1 : data.n_points
        if k == 1
            [F_plus,F_minus]= get_F_Splitting_VanLeer(j,data);
        else
            [F_plus,F_minus]= get_F_Splitting_StegerWarming(j,data);
        end
        for i = 1: 4 % i loop the vector
            data.F_plus(j,i) = F_plus(i);
            data.F_minus(j,i)= F_minus(i);
        end
    end
end
% obtain the two split F-vector for each j-cell
function [F_plus,F_minus] = get_F_Splitting_StegerWarming(j,data)
    global Cp Cv gamma R dH_f0;
    c   = sqrt( gamma* R* data.T(j) ) ; % local sound speed
    u   =  data.u(j);
    rho = data.rho(j);
    rhou= rho*u;
    Y   = data.Y(j);
    F(1)= rhou ;
    F(2)= rhou*u+data.P(j);
    F(3)= rhou*data.H(j);
    F(4)= rhou*data.Y(j);



    if( abs(u) < c ) % subsonic
        % check lecture notes for the Steger-Warming splitting
        coeff = 0.5*rho/gamma*( u-c );
        F_minus(1) =  coeff;
        F_minus(2) =  coeff* ( u-c) ;
        F_minus(3) =  coeff* 0.5*( (u-c)^2 + c*c*( 3-gamma)/(gamma-1)  );  % Energy eq

    else % supersonic
        if( u > c) % all 3 right-going waves
            F_minus(1) = 0;    F_minus(2) = 0;    F_minus(3) = 0;
        else % all three left going waves
            F_minus(1) = F(1); F_minus(2) = F(2); F_minus(3) = F(3) ;
        end
    end

    if u < 0
        F_minus(4) = F(4);
    else
        F_minus(4) = 0;
    end
    F_minus(3) = F_minus(3) + F_minus(4)*dH_f0;
    F_plus = F - F_minus;
end

function [F_plus,F_minus] = get_F_Splitting_VanLeer(j,data)
    global Cp Cv gamma R dH_f0;
    c = sqrt( gamma* R* data.T(j) ) ; % local sound speed
    u =  data.u(j);
    rho = data.rho(j);
    rhou = rho*u;
    Y =  data.Y(j);

    F(1)= rhou;
    F(2)= rhou*u +  data.P(j);
    F(3)= rhou*data.H(j) ;
    F(4)= rhou * Y;

    M = u/c;
    coeff = 0.25*rho*c*(M+1)^2;
    F_plus(1) = coeff;
    F_plus(2) = coeff*2*c/gamma *( 1+ 0.5* (gamma-1)*M );
    %this is incorrect
    if u >= 0
        F_plus(4) = F(4);
    else
        F_plus(4) = 0;
    end
    F_plus(3) = coeff*2*c*c/(gamma*gamma-1) * ( 1+ 0.5* (gamma-1)*M )^2 + dH_f0 * F_plus(4);
    %F_plus(4) = coeff*2*c*c/(gamma*gamma-1) * ( 1+ 0.5* (gamma-1)*M )^2;
    F_minus = F - F_plus;
end


function [data] = Update_PTEsM_AfterReset_V(data)
    global Cp Cv gamma R dH_f0;
    for j = 1 : data.n_points
        rho = data.rho(j);  % V = [rho, u, H]
        u   = data.u(j);
        Y   = data.Y(j);
        H_c = Y * dH_f0;
        H   = data.H(j) - H_c;          % total enthalpy
        T   = (H-0.5*u*u )/Cp ;
        E   = Cv * T + 0.5 * u * u + H_c;       % total energy
        P   = rho*R*T;

        data.P(j) = P;  % get P, T, E to ease later computation of U and E
        data.T(j) = T;
        data.E(j) = E;
        data.s(j) = Cv*log( P/(rho)^gamma );
        data.M(j) = u/sqrt( gamma * R * T);
   end
end

function [data]= SetInitialConditions(data, BC)
    global Cp Cv gamma R;
    if    strcmp( BC.type, 'ShockTube'  )
        %
        data.state = zeros(data.n_points,4)
        T_L = BC.P_L/(R* BC.rho_L);  H_L = Cp*T_L+0.5 * BC.u_L^2 * BC.Y_L;
        T_R = BC.P_R/(R* BC.rho_R);  H_R = Cp*T_R+0.5 * BC.u_R^2 * BC.Y_R;
        n_points= data.n_points;
        for j=1 : n_points
            if( j<= n_points * 0.5 )
                data.rho(j)=BC.rho_L; data.u(j)=BC.u_L; data.H(j)=H_L; data.Y(j) = BC.Y_L;
            else
                data.rho(j)=BC.rho_R; data.u(j)=BC.u_R; data.H(j)=H_R; data.Y(j) = BC.Y_R;
            end
        end

        e_l  = H_L - R * BC.T_L

        e_r  = H_R - R * BC.T_R
        mid = round(data.n_points/2);
        data.state(1:mid,1) = BC.rho_L;
        data.state(1:mid,2) = BC.rho_L * BC.u_L;
        data.state(1:mid,3) = BC.rho_L * e_l;
        data.state(1:mid,4) = BC.rho_L * BC.Y_L;
        data.state(mid:end,1) = BC.rho_R;
        data.state(mid:end,2) = BC.rho_R * BC.u_R;
        data.state(mid:end,3) = BC.rho_R * e_r;
        data.state(mid:end,4) = BC.rho_R * BC.Y_R;
        data = Update_PTEsM_AfterReset_V(data);
        %
    end
end
% obtain the F-vector
function [F] = get_F(j,data)
    rhou = data.rho(j)*data.u(j);
    F(1)= rhou ;
    F(2)= rhou*data.u(j) +  data.P(j);
    F(3)= rhou*data.H(j) ;
    F(4)= data.rho(j)*data.u(j)*data.Y(j);
end

% obtain the U-vector
function [U] = get_U(j,data)
    U(1)= data.rho(j)        ;
    U(2)= data.rho(j)*data.u(j) ;
    U(3)= data.rho(j)*data.E(j) ;
    U(4)= data.rho(j)*data.Y(j) ;
end
% obtain the W-vector
function [W] = get_W(j,data)
    global tau_c T_A;
    W(1)= 0;
    W(2)= 0;
    W(3)= 0;
    %W(4)=  data.rho(j) / tau_c * (1 - data.Y(j))*exp(-T_A/data.T(j));
    W(4)=  0.5 * (data.rho(j+1) + data.rho(j-1)) / tau_c * (1 - 0.5 * (data.Y(j-1)  + data.Y(j+1)))*exp(-T_A/(0.5 * (data.T(j+1) + data.T(j-1))));
end

function [F_minus] = get_F_minus(j,data)
    for i = 1: 4
        F_minus(i)=data.F_minus(j,i) ;
    end
end

function [F_plus] = get_F_plus(j,data)
    for i = 1: 4
        F_plus(i)=data.F_plus(j,i);
    end
end

function [F_jhalf] = get_F_jhalf(j,data)
    for i = 1: 4
        F_jhalf(i)=data.F_jhalf(j,i) ;
    end
end

% Translate the inputting U to the 3 primitive variable [rho,u,H] inside
% core-set V, then write to the correponding spatial memories carried in
% the structure "data" at the location of j cell
function data= U_to_V(U,j,data)
 global Cp Cv gamma R dH_f0;
  data.rho(j) = U(1) ;
  data.u(j)   = U(2)/U(1) ;
  Y           = U(4)/U(1);

  if Y > 1
      Y = 1;
  elseif Y < 0
      Y = 0;
  end

  data.Y(j) = Y;
  data.H(j)   = gamma*U(3)/U(1) -(gamma-1)*0.5* ( U(2)/U(1) )^2 - (gamma - 1) * Y * dH_f0;
end
function data= U_to_V2(U,data)
 global Cp Cv gamma R dH_f0;
  data.rho = U(1,:) ;
  data.u   = U(2,:)./U(1,:) ;
  Y           = U(4,:)./U(1,:);

  Y(Y > 1) = 1;
  Y(Y < 0) = 0;

  data.Y = Y;
  data.H   = gamma.*U(3,:)./U(1,:) -(gamma-1).*0.5.* ( U(2,:)/U(1,:) ).^2 - (gamma - 1) .* Y .* dH_f0;
end
% obtain the two split F-vector for each j-cell


function [data]= BoundaryCondition_ZeroGradient(data, BC)
    N = data.n_points;
    data.rho(1) = data.rho(2); data.rho(N)= data.rho(N-1);
    data.u(1) = data.u(2);     data.u(N)=data.u(N-1);
    data.H(1) = data.H(2);     data.H(N)= data.H(N-1);
    data.Y(1) = data.Y(2);     data.Y(N)= data.Y(N-1);
end

function [data]= BoundaryCondition_Wall_zeroPressureGradient(data, BC , dt,dx)
    global Cp Cv gamma R;
    N = data.n_points;
    % set zero veloicty at both ends of wall
    data.u(1) = 0;     data.u(N) = 0;

     %%%%%%%%%%%%%%%%%%%%%%
     %   left-wall boundary
     u   = 0;
     rho = data.rho(1);
     H   = data.H(1);

     % intermeidate value
     T  = (H-0.5*u*u )/Cp ;
     P  = rho*R*T;                   P_old_save = P;
     c  = sqrt( gamma * R * T);

     % The zero-velocity-wall dictaties the following two conditions
     %  dx(p)=0  due to the momemetum eq.  dt(u) + u*dx(u) + dx(P) /rho = 0;  <== u=0.
     P_new = data.P(2);  % 1st order spatial approximation
     % continuity:  dt(rho) + rho*dx(u) = 0        <== u = 0
     % energy    :  dt( P ) + rho*c^2* dx(u) = 0   <== u =0
     % 2: Therefore  dt(rho) = dt(P) /c^2;
     rho_new = rho +  ( P_new - P_old_save) /c^2;  % first order tempral approxmation
     H_new = P_new/(rho_new*R)*Cp + 0.5 *u^2;
     %

     % update left boundary
     data.rho(1) = rho_new;
     data.H(1)   = H_new;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
     % the right wall boundary
     u   = 0;
     rho = data.rho(N);
     H   = data.H(N);

     % intermeidate value
     T  = (H-0.5*u*u )/Cp ;
     P  = rho*R*T;                   P_old_save = P;
     c  = sqrt( gamma * R * T);

     % The zero-velocity-wall dictaties the following two conditions
     % 1:  dx(p)=0  due to the momemetum eq.  dt(u) + u*dx(u) + dx(P) /rho = 0;  <== u=0.
     P_new = data.P(N-1);  % 1st order spatial approximation
     % continuity:  dt(rho) + rho*dx(u) = 0        <== u = 0
     % energy    :  dt( P ) + rho*c^2* dx(u) = 0   <== u =0
     % 2: Therefore  dt(rho) = dt(P) /c^2;
     rho_new = rho +  ( P_new - P_old_save) / c^2;  % first order tempral approxmation
     H_new = P_new/(rho_new*R)*Cp + 0.5 *u^2;
     %

     % update the new value for the right boundary
     data.rho(N) = rho_new;
     data.H(N)   = H_new;
end

function [data]= BoundaryCondition_WallReflection(data, BC , dt,dx)
    global Cp Cv gamma R dH_f0 tau_c T_A;

    N = data.n_points;

    % set zero veloicty at both ends of wall
    data.u(1) = 0;     data.u(N) = 0;

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %   left-wall boundary
     u   = 0;
     rho = data.rho(1);
     H   = data.H(1); %Total enthalpy
     Y  = data.Y(1);

     % intermeidate value
     T  = (H-0.5*u*u - Y*dH_f0 )/Cp ;
     E  = H - R*T - Y*dH_f0; % Not total energy
     P  = rho*R*T;
     s  = Cv*log( P/(rho)^gamma );
     c  = sqrt( gamma * R * T);
     M  = u/c;
     W  = rho/ tau_c * ( 1 - Y )* exp(- T_A / T );
     % The three values representing (negative) time-variation of wave amplitude for each of characertics wave:
     % L1, L2, L3
     % L1 = lambda1*(dp/dx-rho*c*du/dx);
     % L2 = lambda2*(c^2*drho/dx-dp/dx)
     % L3 = lambda3*(dpdx+rho*c*du/dx);
     %

     % lambda1-wave goes from the right-interior-domain-point to the left-wall
     % therefore compute L W(4)*dH_f0)1 using one-side difference using the interior point
     L1 = ( u-c ) * ( ( data.P(2) - data.P(1) )/dx - rho*c* ( data.u(2) - data.u(1) )/dx   )  ;

     % The zero-velocity-wall dictaties the following two conditions
     L3=L1;
     L2= 0;

     % Now update equation of continiuty and energy ( momentum equation is not needed)
     d1 = (L2+ (L3+L1)/2 )/c^2;
     d2 = (L3+L1)/2;
     d3 = (L3-L1)/(2*rho*c);

     rho_new = rho   - dt* d1;
     rhoE_new= rho*(E ) - dt* ( 0.5*u^2*d1+  d2/(gamma-1) + rho*u*d3 +  W*dH_f0  ); %+(dH_f0*Y/(2*c)) * ( L3/(u+c) - L1/(u-c))
     rhoY_new= rho*Y - dt* ( (Y/(2*c)) * ( L3/(u+c) - L1/(u-c)) - W);

     Y_new   = rhoY_new/rho_new;

     E_new   =  (rhoE_new)/rho_new ;
     T_new   = (E_new  -0.5*u*u )/Cv;
     H_new   = E_new + R*T_new + Y_new*dH_f0;

     % update left boundary
     data.rho(1) = rho_new;
     data.H(1)   = H_new;
     data.Y(1)   = Y_new;


     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
     % the right wall boundary
     u  = 0;
     rho= data.rho(N);
     H  = data.H(N);
     Y  = data.Y(N);

     % intermeidate value
     T  = (H-0.5*u*u-Y*dH_f0)/Cp ;
     E  = H - R*T -Y*dH_f0;
     P  = rho*R*T;
     s  = Cv*log( P/(rho)^gamma );
     c  = sqrt( gamma * R * T);
     M  = u/c;
     W  = rho/ tau_c * ( 1 - Y )* exp(- T_A / T );
     % The three values representing (negative) time-variation of wave amplitude for each of characertics wave:
     % L1, L2, L3
     % L1 = lambda1*(dp/dx-rho*c*du/dx);
     % L2 = lambda2*(c^2*drho/dx-dp/dx)
     % L3 = lambda3*(dpdx+rho*c*du/dx);
     %

     % lambda3-wave goes from the left-interior-domain-point to the right-wall
     % therefore compute L3 using one-side difference using the interior point
     L3= ( u+c ) * ( ( data.P(N) - data.P(N-1) )/dx + rho*c* ( data.u(N) - data.u(N-1) )/dx   )  ;

     % The zero-velocity-wall dictaties the following two conditions
     L1=L3;
     L2= 0;

     % Now update equation of continiuty and energy ( momentum equation is not needed)
     d1 = (L2+ (L3+L1)/2 )/c^2;
     d2 = (L3+L1)/2;
     d3 = (L3-L1)/(2*rho*c);

     rho_new  = rho   - dt* d1;
     rhoE_new = rho*(E ) - dt* ( 0.5*u^2*d1+  d2/(gamma-1) + rho*u*d3 + W*dH_f0  ); % +  (dH_f0*Y/(2*c)) * ( L3/(u+c) - L1/(u-c))
     rhoY_new = rho*Y - dt* ( (Y/(2*c)) * ( L3/(u+c) - L1/(u-c)) - W);

     % the newly updated condionts

     Y_new    = rhoY_new/rho_new;

     E_new   =  (rhoE_new)/rho_new ;
     T_new   = (E_new  -0.5*u*u )/Cv;
     H_new   =  E_new+ R*T_new + Y_new*dH_f0;

     % update the new value for the right boundary
     data.rho(N) = rho_new;
     data.H(N)   = H_new;
     data.Y(N)   = Y_new;
end

function [data]= BoundaryCondition_OpenEnd(data, BC , dt,dx)
    global Cp Cv gamma R dH_f0 tau_c T_A;

    N = data.n_points;

    % set zero veloicty at left end of pipe
    data.u(1) = 0;     %data.u(N) = 0;
    %   left-wall boundary
    u   = 0;
    rho = data.rho(1);
    H   = data.H(1); %Total enthalpy
    Y  = data.Y(1);

    % intermeidate value
    T  = (H-0.5*u*u - Y*dH_f0 )/Cp ;
    E  = H - R*T - Y*dH_f0; % Not total energy
    P  = rho*R*T;
    s  = Cv*log( P/(rho)^gamma );
    c  = sqrt( gamma * R * T);
    M  = u/c;
    W  = rho/ tau_c * ( 1 - Y )* exp(- T_A / T );
    % The three values representing (negative) time-variation of wave amplitude for each of characertics wave:
    % L1, L2, L3
    % L1 = lambda1*(dp/dx-rho*c*du/dx);
    % L2 = lambda2*(c^2*drho/dx-dp/dx)
    % L3 = lambda3*(dpdx+rho*c*du/dx);
    %

    % lambda1-wave goes from the right-interior-domain-point to the left-wall
    % therefore compute L W(4)*dH_f0)1 using one-side difference using the interior point
    L1 = ( u-c ) * ( ( data.P(2) - data.P(1))/dx - rho*c* ( data.u(2) - data.u(1) )/dx   )  ;

    % The zero-velocity-wall dictaties the following two conditions
    L3=L1;
    L2= 0;

    % Now update equation of continiuty and energy ( momentum equation is not needed)
    d1 = (L2+ (L3+L1)/2 )/c^2;
    d2 = (L3+L1)/2;
    d3 = (L3-L1)/(2*rho*c);

    rho_new = rho   - dt* d1;
    rhoE_new= rho*(E ) - dt* ( 0.5*u^2*d1+  d2/(gamma-1) + rho*u*d3 +  W*dH_f0  ); %+(dH_f0*Y/(2*c)) * ( L3/(u+c) - L1/(u-c))
    rhoY_new= rho*Y - dt* ( (Y/(2*c)) * ( L3/(u+c) - L1/(u-c)) - W);

    Y_new   = rhoY_new/rho_new;

    E_new   =  (rhoE_new)/rho_new ;
    T_new   = (E_new  -0.5*u*u )/Cv;
    H_new   = E_new + R*T_new + Y_new*dH_f0;

    % update left boundary
    data.rho(1) = rho_new;
    data.H(1)   = H_new;
    data.Y(1)   = Y_new;

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
     % the right wall boundary
     rho= data.rho(N);
     H  = data.H(N);
     Y  = data.Y(N);
     u = data.u(N);

     % intermeidate value
     T  = (H-0.5*u*u-Y*dH_f0)/Cp ;
     E  = H - R*T -Y*dH_f0;
     P  = rho*R*T;
     s  = Cv*log( P/(rho)^gamma );
     c  = sqrt( gamma * R * T);
     M  = u/c;
     W  = rho/ tau_c * ( 1 - Y )* exp(- T_A / T );
     %supersonic outlet
     if (u - c) > 0;

        L1 = (u-c) * ((data.P(N) - data.P(N-1)) / dx - c * rho * (data.u(N) - data.u(N-1)) / dx);
        L2 = u * (c^2*(( data.rho(N) - data.rho(N-1))/dx) - (data.P(N) - data.P(N-1) )/dx);
        L3= ( u+c ) * ( ( data.P(N) - data.P(N-1) )/dx + rho*c* ( data.u(N) - data.u(N-1) )/dx);

        drhouY = Y/(2 * c) * (L3 / (u + c) - L1 /(u - c)) + Y * u / c^2 * (L2/u + 1/2 * (L3/(u + c) + L1 / (u - c))) + rho * u * (data.Y(N) - data.Y(N-1)) / dx;

        d1 = (L2+ (L3+L1)/2 )/c^2;
        d2 = (L3+L1)/2;
        d3 = (L3-L1)/(2*rho*c);
        % Now update equation of continiuty and energy ( momentum equation is not needed)
        rho_new  = rho   - dt* d1;
        u_new = 1 / rho_new * (rho * u - dt * (u * d1 + rho * d3));
        rhoE_new = rho*E - dt* ( 0.5*u^2*d1+  d2/(gamma-1) + rho*u*d3 );
        rhoY_new = rho * Y - dt * (drhouY - W);

        Y_new    = rhoY_new/rho_new;
        E_new = rhoE_new/rho_new ;
        T_new   = ( E_new   -0.5*u_new*u_new)/Cv;
        H_new   = E_new + R*T_new + Y_new * dH_f0;

        data.rho(N) = rho_new;
        data.H(N)   = H_new;
        data.u(N)   = u_new;
        data.Y(N)   = Y_new;

     %subsonic outlet
      elseif u > 0;
         % p_t = 0
         P_new = 100000;

         L2 = u * (c^2*(( data.rho(N) - data.rho(N-1))/dx) - (data.P(N) - data.P(N-1) )/dx);
         L3= ( u+c ) * ( ( data.P(N) - data.P(N-1) )/dx + rho*c* ( data.u(N) - data.u(N-1) )/dx);
         L1 = -L3;

         drhouY = Y/(2 * c) * (L3 / (u + c) - L1 /(u - c)) + Y * u / c^2 * (L2/u + 1/2 * (L3/(u + c) + L1 / (u - c))) + rho * u * (data.Y(N) - data.Y(N-1)) / dx;

         d1 = (L2+ (L3+L1)/2 )/c^2;
         d2 = (L3+L1)/2;
         d3 = (L3-L1)/(2*rho*c);
         % Now update equation of continiuty and energy ( momentum equation is not needed)
         rho_new  = rho   - dt* d1;
         rhou_new = (rho * u - dt * (u * d1 + rho * d3));
         rhoY_new = rho * Y - dt * (drhouY - W);

         Y_new    = rhoY_new/rho_new;
         T_new = P_new/(R*rho_new);
         u_new = rhou_new/rho_new;
         H_new   = T_new*Cp + 0.5 * u_new * u_new + Y_new * dH_f0;

         data.rho(N) = rho_new;
         data.H(N)   = H_new;
         data.u(N)   = u_new;
         data.Y(N)   = Y_new;
     %subsonic inlet
     else u < 0;
        %T_t = p_t = 0
        T_new   = 300;
        P_new   = 100000;
        Y_new   = 0;

        L3= ( u+c ) * ( ( data.P(N) - data.P(N-1) )/dx + rho*c* ( data.u(N) - data.u(N-1) )/dx);
        L1 = - L3;
        L2 = 0;

        d1 = (L2+ (L3+L1)/2 )/c^2;
        d2 = (L3+L1)/2;
        d3 = (L3-L1)/(2*rho*c);
        % Now update equation of continiuty and energy ( momentum equation is not needed)
        rhou_new =(rho * u - dt * (u * d1 + rho * d3));

        rho_new = P_new/(R*T_new);
        u_new = rhou_new/rho_new;
        H_new   = T_new*Cp + 0.5 * u_new * u_new;

        data.rho(N) = rho_new;
        data.H(N)   = H_new;
        data.u(N)   = u_new;
        data.Y(N)   = Y_new;
     end
end
