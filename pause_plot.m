
function pause_plot(data1, data2 , str_NameOfNumericalScheme)
% Plots six images of some different physical qunatities.
    % clear previous title
    delete(findall(gcf,'Tag','SuperTitle'));

%figure (1);
    strTitle = [str_NameOfNumericalScheme ': t = ' num2str(data2.CurrentTime*1000) 'ms'];

% ha = tight_subplot(2,3,[.01 .03],[.1 .01],[.01 .01]);


    subplot(2,3,1);
    plot(data2.x, data2.P/100000 ,  '-r' ); hold on;
    plot(data1.x, data1.P/100000   ,  '--b'  ,'LineWidth' , 2);  hold on;
    ylabel('P (Bar)'); title('Pressure '); grid on;
    hold off;
    subplot(2,3,2);
    plot(data2.x, data2.rho,  '-r' ) ;    hold on;
    plot(data1.x, data1.rho  ,  '--b'  ,'LineWidth' , 2);    hold on;
    ylabel('rho (kg/m^3)'); title('Density'); grid on;
    hold off;

    %subplot(2,3,3);
    %plot(x, data2.s ,  '-r' ); hold on;
    %plot(x, data1.s   ,  '--b' ,'LineWidth' , 2 ); hold on;
    %xlabel('x(m)'); ylabel('s (J/K/kg)'); title('Entropy'); grid on;
    %hold off;

    subplot(2,3,3);
    plot(data2.x, data2.Y ,  '-r' ); hold on;
    plot(data1.x, data1.Y   ,  '--b' ,'LineWidth' , 2 ); hold on;
    xlabel('x(m)'); ylabel('Y (kg^-1)'); title('Massfraction'); grid on;
    ylim([0 1]);
    hold off;

    ha =  axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0  1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    %text(0.35, 0.98, strTitle , 'Fontsize', 20);


    subplot(2,3,4);
    plot(data2.x, data2.T ,  '-r' ); hold on;
    plot(data1.x, data1.T   ,  '--b' ,'LineWidth' , 2 ); hold on;
    xlabel('x(m)'); ylabel('T (K)'); title('Temperature '); grid on;
    hold off;

    subplot(2,3,5);
    plot(data2.x, data2.u ,  '-r' ); hold on;
    plot(data1.x, data1.u,  '--b'  ,'LineWidth' , 2); hold on;
    ylabel('u (m/s)'); title('Velocity '); grid on;
    hold off;


    hold off;
    subplot(2,3,6);
    plot(data2.x, data2.M ,  '-r' ); hold on;
    plot(data1.x, data1.M   ,  '--b' ,'LineWidth' , 2 ); hold on;
    xlabel('x(m)'); ylabel('M'); title('Local Mach number'); grid on;
    hold off;
    annotation('textbox', [0 0.9 1 0.1],  'String', strTitle, 'EdgeColor', 'none',  'FontSize',12,  'HorizontalAlignment', 'center', 'Tag' , 'SuperTitle');



    pause (0.001);
end
