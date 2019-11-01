%% plth_loop
% ~~~
% GX Castegnetti --- 2018

clear
close all

restoredefaultpath, clear RESTOREDEFAULTPATH_EXECUTED

%% Network and arena parameters
p.PCs       = 160;      % number of total place cells (half in safe, half in dangerous compartment)
p.FCs       = 40;       % number of fear cells
p.W_0       = 0.25;     % initial synaptic strenght
actSpont    = 0.85;     % spontaneous firing rate (Par� and Collins, 2000)
freezeThr   = 5;      % firing rate of BLA neurons during freezing (Par� and Collins, 2000). We take it as threshold for freezing.
nSweeps     = 20;

%% Provide parameters for the neuron, synapse and plasticity models
p.v_reset   = -75;      % reset potential
p.v_thr     = -55;      % firing threshold
p.v_E       = -65;      % reversal potential
p.v_rest    = -65;      % resting potential
p.tau_m     = 0.02;     % time constant of the neural model
p.R         = 10;       % resistance in the neural model
p.dt        = 0.001;    % width of the time step
p.lambda    = 0.1;

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


%% Additional settings
p.pulses   = 100;  % duration of the training phase (s)
p.TimeRecall = 5;  % how many theta cycles during the test phase

%% set EPSP amplitude
p.A_EPSP = 10; % 13 mV - within observed range (Strober at al., 2015; Rosenkranz 2012; Cho et al. 2012)

%% run model
in_freq = [4.5 5]; % training frequency
c = 1;
for sweep = 1:nSweeps
    W_0 = p.W_0 + zeros(p.PCs,p.FCs);               % initial synaptic strength vector
    W_AftCond = zeros(p.PCs,p.FCs,length(in_freq)); % initialise matrix for W after training
    for f = 1:length(in_freq)
        tic
        disp(['Simulation ' int2str(c) ' of ' int2str(length(in_freq)*nSweeps) '...']); % update user
        c = c+1;
        
        %% Training 
        % ---------------------------------------

        % Presynaptic activity 
        % ---------------------------------------

        % Theta frequency
        in.freq = in_freq(f);

        % Acetylcholine is high during training (= novel environment)
        in.ACh = 1;

        % Hippocampal input as a Poisson process with theta-modulated rate
        in.time     = 1 : p.pulses / in_freq(f) / p.dt;       % Time axis in milliseconds
        rate        = 1 + sin(in_freq(f) * 2*pi .* in.time .* p.dt);	% Rate function
        rate        = rate ./ sum(rate) * p.pulses;           % Normalise rate function
        actHpc      = poissrnd(repmat(rate,p.PCs,1));         % Poisson spike input
        clear rate

        % Assign 1st and 2nd half of PCs to safe and dang compartment, respectively
        actHpc_trSafe = actHpc;
        actHpc_trSafe(1+round(p.PCs/2):end,:) = 0;
        actHpc_trDang = actHpc;
        actHpc_trDang(1:round(p.PCs/2),:) = 0;
        clear hpc_spike i


        % Postsynaptic activity
        % ---------------------------------------

        % Spontaneous activity (0.85 Hz in both safe and dangerous compartments)
        rng('shuffle')
        actAmy_trSafe = poissrnd(actSpont*p.dt*ones(p.FCs,length(in.time)));
        actAmy_trDang = poissrnd((1+actSpont)*p.dt*ones(p.FCs,length(in.time)));


        % Time evolution
        % ---------------------------------------

        % Simulate training - safe compartment
        in.in_hpc = actHpc_trSafe;
        [W_AftSafe{f},~,acprops(f,1).pre_safe] = plth_evolve(p,in,actAmy_trSafe,W_0);

        % Simulate training - dangerous compartment
        in.in_hpc = actHpc_trDang;
        [W_AftAll{f},~,acprops(f,1).pre_dang] = plth_evolve(p,in,actAmy_trDang,W_AftSafe{f});


        %% Recall
        % ---------------------------------------

        % Timing stuff
        in.t_end = p.TimeRecall;
        in.time  = p.dt:p.dt:in.t_end;


        % Presynaptic activity
        % ---------------------------------------

        % Acetylcholine is low during recall (= familiar environment)
        in.ACh = 0;

        % Theta frequency (constant during recall)
        freqRecall = 5;
        T = 1/in.freq;

        % Hippocampal input as a Poisson process with theta-modulated rate
        rate        = 1 + sin(freqRecall * 2*pi .* in.time);   % Rate function
        rate        = rate ./ sum(rate) * freqRecall * p.TimeRecall;      % Normalise rate function
        actHpc      = poissrnd(repmat(rate,p.PCs,1));    % Poisson spike input
        clear rate

        % hippocampal input at theta frequency
        hpcSpkTime_recall = repmat((T+(1-in.ACh)*T/2):T:in.t_end,p.PCs,1); % if ACh == 1 -> HPC fires at theta peak (maximal depolarisation)
        hpcFiring_recall = zeros(p.PCs,length(in.time));
        for i = 1:p.PCs
            hpcFiring_recall(i,round(hpcSpkTime_recall/p.dt)) = 1;
        end

        hpcFiring_recall = actHpc;

    %     % add spontaneous activity
    %     rng('shuffle')
    %     rndPois_hpc = rand(p.PCs,length(in.time));
    %     randAct_hpc = rndPois_hpc < actSpont*p.dt; clear rndPois_hpc
    %     
    %     hpcFiring_recall = hpcFiring_recall + randAct_hpc;
         hpcFiring_recall = heaviside(hpcFiring_recall - 0.1);


        % Postsynaptic activity
        % ---------------------------------------

        % only spontaneous activity during recall
        rng('shuffle')
        rndPois = rand(p.FCs,length(in.time));
        amyFire_recall = rndPois < actSpont*p.dt;


        % Time evolution
        % ---------------------------------------

        % safe
        in.in_hpc = hpcFiring_recall;
        in.in_hpc(1+round(p.PCs/2):end,:) = 0; % <--- retain only place cells tuned on safe compartment
        [~,frTestSafe{f},acprops(f,1).post_safe] = plth_evolve(p,in,amyFire_recall,W_AftAll{f}); %#ok<*SAGROW>

        % dang
        in.in_hpc = hpcFiring_recall;
        in.in_hpc(1:round(p.PCs/2),:) = 0; % <--- retain only place cells tuned on dang compartment
        [~,frTestDang{f},acprops(f,1).post_dang] = plth_evolve(p,in,amyFire_recall,W_AftAll{f});
        
        
        %% summary
        
        % average firing rate
        frRecall_mean_safe(sweep,f) = mean(frTestSafe{f}); %#ok<*SAGROW>
        frRecall_mean_dang(sweep,f) = mean(frTestDang{f});
        
        % percentage of neurons that overcome threshold
        frRecall_abThr_safe(sweep,f) = sum(frTestSafe{f} > freezeThr)/p.FCs;
        frRecall_abThr_dang(sweep,f) = sum(frTestDang{f} > freezeThr)/p.FCs;
        
        toc
    end, clear f i T
end

%% Percentage of simulations in which avg firing rate overcomes threshold.
frRecall_avgAbThr_safe = freezeThr < frRecall_mean_safe;
frRecall_avgAbThr_dang = freezeThr < frRecall_mean_dang;
freezePercOfSimul_safe = mean(frRecall_avgAbThr_safe,1);
freezePercOfSimul_dang = mean(frRecall_avgAbThr_dang,1);
percFreezeSafeStd = std(frRecall_avgAbThr_safe,1);
percFreezeDangStd = std(frRecall_avgAbThr_dang,1);

%% Percentage of neurons overcoming threshold
frRecall_abThr_mean_safe = mean(frRecall_abThr_safe,1);
frRecall_abThr_mean_dang = mean(frRecall_abThr_dang,1);
frRecall_abThr_std_safe  = std(frRecall_abThr_safe,1);
frRecall_abThr_std_dang  = std(frRecall_abThr_dang,1);

%% plot freezing percentage - mean
figure('color',[1 1 1]),hold on
bar(in_freq,[freezePercOfSimul_safe' freezePercOfSimul_dang'])
xlabel('Training theta frequency (Hz)'),ylabel('% freezing')
legend('Safe comp.','Threatening comp.','location','northwest'), legend boxoff
set(gca,'fontsize',18,'xtick',in_freq), ylim([-0.05,1])

%% plot average number of neurons above threshold
figure('color',[1 1 1]),hold on
bar(in_freq,[frRecall_abThr_mean_safe' frRecall_abThr_mean_dang'])
xlabel('Training theta frequency (Hz)'),ylabel('% cells > threshold')
legend('Safe comp.','Threatening comp.','location','northwest'), legend boxoff
set(gca,'fontsize',18,'xtick',in_freq), ylim([0,1.2*max(frRecall_abThr_mean_dang)])

%% plot theta coherence

for i = 1:length(in_freq)
    idxCoh = i;
    figure
    
    subplot(2,2,1)
    plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).pre_safe.power,'k','LineWidth',3)
    ylim([0 0.2]),title(['f = ', num2str(in_freq(i)),'- PreSafe'])
    axis square
    
    subplot(2,2,2)
    plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).post_safe.power,'k','LineWidth',3)
    ylim([0 0.2]),title(['f = ', num2str(in_freq(i)),'- PostSafe'])
    axis square
    
    subplot(2,2,3)
    plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).pre_dang.power,'k','LineWidth',3)
    ylim([0 0.2]),title(['f = ', num2str(in_freq(i)),'- PreDang'])
    axis square
    
    subplot(2,2,4)
    plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).post_dang.power,'k','LineWidth',3)
    ylim([0 0.2]),title(['f = ', num2str(in_freq(i)),'- PostDang'])
    axis square
    
end

% save([pwd,filesep,'out',filesep,'simulation_150818_errorBars'])

%     thetaBand = [4 8];
%     figure
%     thetaLims = [find(acprops(idxCoh).pre_safe.F>=thetaBand(1),1,'first') : find(acprops(idxCoh).pre_safe.F<=thetaBand(2),1,'last')];
%     data      = [mean(acprops(idxCoh).pre_safe.Cxy(thetaLims)) mean(acprops(idxCoh).post_safe.Cxy(thetaLims)) nan mean(acprops(idxCoh).pre_dang.Cxy(thetaLims)) mean(acprops(idxCoh).post_dang.Cxy(thetaLims))];
%     bar(data,'EdgeColor','k','LineWidth',2,'FaceColor',[0.8 0.8 0.8])
%     axis square
