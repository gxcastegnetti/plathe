%% plth_plasticityProtocols
% ~~~
% Simulates presynaptic rate-induced plasticity protocol and STDP protocol
% under varying theta frequency and phase.
% ~~~
% GX Castegnetti --- 2017

clear
close all

%% Specify the input protocol
Input       = 'freq';     % which figure to replicate - freq / stdp

%% Provide parameters for the neuron, synapse and plasticity models
p.v_reset   = -75;      % reset potential
p.v_thr     = -55;      % firing threshold
p.v_E       = -65;      % reversal potential
p.v_rest    = -65;
p.tau_m     = 0.02;     % time constant of the neural model
p.R         = 10;       % resistance in the neural model
p.dt        = 0.001;    % width of the time step
p.lambda    = 0;        % set to 1 only to reproduce Shouval's results

% theta
p.A_theta   = 0;        % theta amplitude

% EPSP
p.tau_1     = 50;
p.tau_2     = 5;

% I_NMDA
p.P_0       = 0.5;
p.G_NMDA    = -1/500;
p.I_f       = 0.5;
p.I_s       = 0.5;
p.tau_f     = 50;
p.tau_s     = 200;
p.V_r       = 130;
p.Mg        = 1;

% Ca
p.tau_Ca    = 50;

% BPAP
p.I_bs_f    = 0.75;
p.I_bs_s    = 1 - p.I_bs_f;
p.tau_bs_f  = 3;
p.tau_bs_s  = 25;

% IPSP
p.tau_1i    = 50;
p.tau_2i    = 5;

% tau, eta
p.P_1       = 0.1;
p.P_2       = p.P_1/10^4;
p.P_3       = 3;
p.P_4       = 1;

% OMEGA
p.alpha1    = 0.35;
p.alpha2    = 0.55;
p.beta1     = 80;
p.beta2     = 80;
p.W_0       = 0.25; % initial synaptic strength

phi         = [-pi/2:pi/2:pi 2*pi];         % different theta phases
phi = 0;
phases      = length(phi);                  % number of different phases to test

switch Input
    case 'freq'
        % 900 pre-syn spikes at various frequencies with no post-syn firing
        p.A_EPSP    = 1.45;                         % EPSP amplitude: 1.45 gives 10mV, 0.145 gives 1mV
        p.NumStim   = 20;                           % number of presynaptic spikes
        p.f         = 0.5:0.5:10.5;                 % pre-synaptic stimulation frequencies
%         p.f = 5;
        W_log       = nan(length(p.f),length(phi)); % assign memory for the output
        runs        = length(p.f);                  % number of simulations
    case 'stdp'
        % STDP is 100 pairings with deltaT = -100 : 200 ms at 1Hz
        p.A_EPSP    = 0.145;                        % EPSP amplitude: 1.45 gives 10mV, 0.145 gives 1mV
        p.NumStim   = 100;                          % number of spike pairings
        p.delta_T   = -100 : 5 : 200;               % relative spike timings
        p.f         = 5;                            % theta frequency (1 Hz -> 5 periods)
        W_log       = nan(length(p.delta_T),length(phi)); % assign memory for the output
        runs        = length(p.delta_T);            % number of simulations
end

%% loop over phase shifts
count = 0;
for j = 1:phases
    
    %% set theta phase wrt presyn. firing (phi = 0 -> peak)
    p.phase     = phi(j);
    if p.phase == 2*pi, p.A_theta = 0; end
    %% Run the dynamics for frequency
    for k       = 1 : runs
        tic
        % update the user
        count   = count + 1;
        disp(['Simulation ' int2str(count) ' of ' int2str(runs*phases) '...']); drawnow
        
        switch Input
            case 'freq'
                T           = 1/p.f(k);                     % time interval between spikes
                SpikePre    = T/4:T:T*p.NumStim;            % input spike times
                SpikePost   = [];                           % post-synaptic spike times
                p.t_end     = SpikePre(end) + T;            % length of simulation
                clear T
            case 'stdp'
                T           = 1/p.f;
                SpikePre    = T + [T/4:5*T:5*T*p.NumStim];          % input spike times (5*T to make stimulations well separated at one Hz)
                SpikePost   = SpikePre + p.delta_T(k)/1000;         % post-synaptic spike times
                p.t_end     = SpikePost(end) + 2*T;                 % length of simulation
        end
        
        % Initialise dynamic variables
        p.time      = p.dt:p.dt:p.t_end;                    % time vector
        v           = p.v_rest;                             % synaptic contribution to membrane potential (no BPAP)
        Ca          = 0;                                    % calcium concentration
        W           = p.W_0;                                % synaptic strength
        NMDAlog     = nan(length(p.time),1);
        Vlog        = nan(length(p.time),1);
        Calog       = nan(length(p.time),1);
        EPSP_exp1   = 0;                                    % parameters controlling AMPA synaptic currents
        EPSP_exp2   = 0;
        NMDA_exp1   = 0;                                    % parameters controlling NMDA synaptic currents
        NMDA_exp2   = 0;
        BPAP_exp1   = 0;                                    % parameters controlling BPAP dynamics
        BPAP_exp2   = 0;
        
        % temporal evolution
        for t = 1 : length(p.time)
            
            %% Presynaptic spikes
            if ismember(round(p.time(t)*1000),round(SpikePre*1000))
                EPSP_exp1   = EPSP_exp1 + 1;                                        % update AMPA time decay
                EPSP_exp2   = EPSP_exp2 + 1;
                NMDA_exp1   = NMDA_exp1 + p.P_0*(1-NMDA_exp1);                      % update NMDA time decay
                NMDA_exp2   = NMDA_exp2 + p.P_0*(1-NMDA_exp2);
            end
            
            %% Postsynaptic spikes
            if ismember(round(p.time(t)*1000),round(SpikePost*1000))
                BPAP_exp1   = BPAP_exp1 + 1;                                        % update BPAP time decay
                BPAP_exp2   = BPAP_exp2 + 1;
            end
            
            %% Compute synaptic currents, update membrane potential
            EPSP            = p.A_EPSP/0.7 * (EPSP_exp1 - EPSP_exp2);               % AMPA current
            v_inf           = p.v_E + p.R*EPSP;                                     % membrane potential update
            v               = v_inf + (v - v_inf) * exp(-p.dt/p.tau_m); clear v_inf % synaptic membrane potential
            BPAP            = (p.I_bs_f*BPAP_exp1 + p.I_bs_s*BPAP_exp2) * 100;      % BPAP voltage (delay by 2ms?)
            switch Input                                                            % theta input
                case 'freq'
                    theta           = p.A_theta*sin(2*pi*p.f(k)*(p.time(t)) + p.phase);     
                case 'stdp'
                    theta           = p.A_theta*sin(2*pi*p.f*(p.time(t)) + p.phase);
            end
            V               = BPAP + theta + v;                                     % total membrane potential
            B               = 1 / (1+exp(-0.062*V)*(p.Mg/3.57));                    % NMDA voltage dependency
            H               = B * (V-p.V_r);
            I_NMDA          = p.G_NMDA * H * (p.I_f*NMDA_exp1 + p.I_s*NMDA_exp2);   % NMDA current
            clear EPSP theta v_inf BPAP B H
            
            %% Calcium concentration
            dCa             = I_NMDA - (1/p.tau_Ca) * Ca;                           % calculate time derivative of [Ca2+]
            Ca              = Ca + dCa; clear dCa I_NMDA                            % update [Ca2+]
            
            %% Synaptic plasticity
            tau             = p.P_1./(p.P_2 + Ca.^p.P_3) + p.P_4;                   % learning rate
            eta             = 1/tau; clear tau
            sig1            = exp((Ca-p.alpha1).*p.beta1) ./ (1 + exp((Ca-p.alpha1).*p.beta1));
            sig2            = exp((Ca-p.alpha2).*p.beta2) ./ (1 + exp((Ca-p.alpha2).*p.beta2));
            OMEGA           = 0.25 + sig2 - 0.25*sig1; clear sig1 sig2              % plasticity parameter
            dW              = eta * (OMEGA - p.lambda*W); clear eta OMEGA           % calculate derivative of synaptic strength and update it
            W               = W + dW * p.dt; clear dW                               % update synaptic weight
            
            %% Update decay
            EPSP_exp1       = EPSP_exp1 * (1 - 1/p.tau_1);
            EPSP_exp2       = EPSP_exp2 * (1 - 1/p.tau_2);
            NMDA_exp1       = NMDA_exp1 * (1 - 1/p.tau_f);
            NMDA_exp2       = NMDA_exp2 * (1 - 1/p.tau_s);
            BPAP_exp1       = BPAP_exp1 * (1 - 1/p.tau_bs_f);
            BPAP_exp2       = BPAP_exp2 * (1 - 1/p.tau_bs_s);
            
            %% Log dynamic variables where necessary
            NMDAlog(t,1)	= p.I_f*NMDA_exp1 + p.I_s*NMDA_exp2;
            Vlog(t,1)       = V;
            Calog(t,1)      = Ca;
            
        end
        clear t
        
        % Store the final relative synaptic weight
        W_log(k,j)          = W / p.W_0;
        
        % Clear all of the dynamic variables before the next iteration
        clear v V Ca W EPSP_exp1 EPSP_exp2 NMDA_exp1 NMDA_exp2 BPAP_exp1
        clear BPAP_exp2
        
%         figure('color',[1 1 1])
%         subplot(3,1,1)
%         plot(p.time,Vlog,'k','LineWidth',3),hold on
%         stem(SpikePre,100*ones(length(SpikePre),1),'BaseValue',-100)
%         if ~isempty(SpikePost), stem(SpikePost,100*ones(length(SpikePost),1),'BaseValue',-100), end
%         hold off
%         xlabel('Time (s)','FontSize',16)
%         ylim([min(Vlog)-5 max(Vlog)+5]),xlim([0 1])
%         ylabel('V (mV)','FontSize',16)
%         switch Input
%             case 'freq'
%                 title(['\phi = ',num2str(phi(j)),' f = ',num2str(p.f(k))])
%             case 'stdp'
%                 title(['\phi = ',num2str(phi(j)),' \Deltat = ',num2str(p.delta_T(k))])
%         end
%         set(gca,'FontSize',14)
%         keyboard      
%         subplot(3,1,2)
%         plot(p.time,NMDAlog,'k','LineWidth',3),hold on
%         stem(SpikePre,100*ones(length(SpikePre),1),'BaseValue',-100)
%         if strcmp(Input,'stdp'), stem(SpikePost,100*ones(length(SpikePost),1),'BaseValue',-100), end
%         hold off
%         ylim([min(NMDAlog)-0.1*min(NMDAlog) max(NMDAlog)+0.1*max(NMDAlog)])
%         xlabel('Time (s)','FontSize',16)
%         ylabel('Fraction NMDA bound','FontSize',16)
%         set(gca,'FontSize',14)
%         
%         subplot(3,1,3)
%         plot(p.time,Calog,'k','LineWidth',3),hold on
%         stem(SpikePre,100*ones(length(SpikePre),1),'BaseValue',-100)
%         if strcmp(Input,'stdp'), stem(SpikePost,100*ones(length(SpikePost),1),'BaseValue',-100), end
%         hold off
%         ylim([min(Calog)-0.1*min(Calog) max(Calog)+0.1*max(Calog)])
%         xlabel('Time (s)','FontSize',16)
%         ylabel('Ca^2^+ (uM)','FontSize',16)
%         set(gca,'FontSize',14)
        toc
    end
    clear k
    
end
%% Plot some key output
figure('color',[1 1 1])
co = [0.10  0.10  0.10
      0.45  0.45  0.45
      0.60  0.60  0.60
      0.85  0.85  0.85
    0.2  0.8  0.8
      0    1  0.9];
set(groot,'defaultAxesColorOrder',co)
switch Input                                                            % theta input
    case 'freq'
        plot(p.f(1:20),W_log(1:20,1:end),'LineWidth',2.5),hold on
        plot(p.f(1:20),W_log(1:20,end),'LineWidth',2.5,'color','k','linestyle','--')
        plot(p.f(1:20),ones(length(p.f(1:20)),1),'color',[0.5 0.5 0.5],'linestyle','--')
        legend('phi = -\pi/2','\phi = 0','phi = \pi/2','phi = \pi','No theta','location','southeast')
        xlabel('Theta frequency (Hz)')
        ylabel('Relative Synaptic Weight')
        set(gca,'fontsize',14)
    case 'stdp'
        plot(p.delta_T,W_log(:,1:end-1),'LineWidth',2.5),hold on
        plot(p.delta_T,W_log(:,end),'LineWidth',2.5,'color','k','linestyle','--')
        plot(p.delta_T,ones(length(p.delta_T),1),'color',[0.5 0.5 0.5],'linestyle','--')
        legend('phi = -\pi/2','\phi = 0','phi = \pi/2','phi = \pi','No theta','location','southeast')
        xlabel('\Deltat (ms)')
        ylabel('Relative Synaptic Weight')
        set(gca,'fontsize',14)
end

% save('STDP_05')