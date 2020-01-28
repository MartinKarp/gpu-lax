global Cp Cv gamma R tau_c T_A dH_f0;
Cp    = 1000 ; % Do not Change
gamma = 1.4;   % Do not Change
Cv= Cp/gamma;
R = Cp - Cv;
tau_c  = 0.0000009/(3*340);
T_A = 2000 * 6;
dH_f0 = -700 * Cp;
%%%%%%%%%%%%%%%%%%%%%%%%

% MATLAB script to compare two differnet simulations

[n, domainl, t, dt, state] = open_out('output1e6.dat');
%domainl
x = linspace(-domainl/2,domainl/2,n);
data_fort.x = x';
data_fort.n_points = n;
data_fort.CurrentTime = t;
data_fort = U_to_V(state,data_fort);
data_fort = Update_PTEsM_AfterReset_V(data_fort);
data_fort
%ReactiveTube
data_mat = load('lax200.mat').data
pause_plot(data_mat, data_fort, 'Red, n = 1e6 Blue, n = 200')



%open file written by the fortran code.
function [n, domainl, t, dt, state] = open_out(file)
    fid=fopen(file, 'rb');        % Open the file.
    hr1=fread(fid, 1, 'int32');             % Read the first record start tag. Returns hr1 = 72
    n= fread(fid, 1, 'int32')';  % Read the first record.  Returns name = I am a genius                                        % groundhog!  Run!  Run!  Run!
    hr1=fread(fid, 1, 'int32');             % Read the first record end tag. Returns hr1 = 72
    hr2=fread(fid, 1, 'int32');             % Read the second record start tag. Returns hr2 = 12
    domainl = fread(fid, 1, 'float32');
    t=fread(fid, 1, 'float32');       % Read data. Returns xsource = 1.1000
    dt=fread(fid, 1, 'float32');       % Read data. Returns zsource = 2.2000
    hr2=fread(fid, 1, 'int32');             % Read the second record end tag. Returns hr2 = 12
    hr3=fread(fid, 1, 'int32');             % Read the third record start tag. Returns hr3 = 8
    state=fread(fid, n*4, 'float32');
    hr3=fread(fid, 1, 'int32');             % Read the third record end tag. Returns hr3 = 8
    fclose(fid);                           % Close file.
    state = reshape(state,[n 4])';
end

% get the V vector
function data= U_to_V(U,data)
 global Cp Cv gamma R dH_f0;
  data.rho = U(1,:) ;
  data.u   = U(2,:)./U(1,:) ;
  Y        = U(4,:)./U(1,:);

  Y(Y > 1) = 1;
  Y(Y < 0) = 0;

  data.Y = Y;
  data.H   = gamma.*U(3,:)./U(1,:) -(gamma-1).*0.5.* (data.u).^2 - (gamma - 1) .* Y .* dH_f0;
end

%Get the different physical quantitites.
function [data] = Update_PTEsM_AfterReset_V(data)
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
