%% thanx_contextCond
% ~~~
% Simulates synaptic plasticity during contextual conditioning and recall,
% and how it is affected by theta activity.
% ~~~
% GX Castegnetti --- 2018

clear
close all

restoredefaultpath, clear RESTOREDEFAULTPATH_EXECUTED
% addpath([pwd,'/tools']);

%% Network and arena parameters
p.PCs       = 100;     % number of total place cells (half in safe, half in dangerous compartment)
p.FCs       = 25;     % number of fear cells
p.W_0       = 0.25;    % initial synaptic strenght
SpontFiring = 0.85;    % spontaneous firing rate (Par� and Collins, 2000)
freezeThr   = 1.5;     % firing rate of BLA neurons during freezing (Parè and Collins, 2000). We take it as threshold for freezing.
numTrainSpikes = 10;   % fixed number of spikes delivered during training

%% Provide parameters for the neuron, synapse and plasticity models
p.v_reset   = -75;      % reset potential
p.v_thr     = -55;      % firing threshold
p.v_E       = -65;      % reversal potential
p.v_rest    = -65;      % resting potential
p.tau_m     = 0.02;     % time constant of the neural model
p.R         = 10;       % resistance in the neural model
p.dt        = 0.001;    % width of the time step
p.lambda    = 0;

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

%% additional settings
p.TrialNum   = 20; % how many stimuli during the training phase
p.TrialDur   = 1;  % trial duration during training
p.TimeRecall = 5;  % how many theta cycles during the test phase

%% set EPSP amplitude
p.A_EPSP = 5; % 10 mV - within observed range (Strober at al., 2015; Rosenkranz 2012; Cho et al. 2012)

%% run model
in_freq = [1.0 5.5];                     % training frequency
%in_freq = 8:3:13;
W_0 = p.W_0 + zeros(p.PCs,p.FCs);     % initial synaptic strength vector
for f = 1:length(in_freq)
    
    disp(['Simulation ' int2str(f) ' of ' int2str(length(in_freq)) '...']); % update user
    
    %% training
    
    % timing stuff
    in.t_end = p.TrialNum*p.TrialDur;   % training duration
    in.time  = p.dt:p.dt:in.t_end;      % training time vector
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % presynaptic activity %
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    % theta frequency
    in.freq = in_freq(f);
    T = 1/in.freq;
    
    % Acetylcholine is high during training (= novel environment)
    in.ACh = 1;
    
    % hippocampal input at theta frequency
    hpc_times = repmat((T+(1-in.ACh)*T/2):T:in.t_end,p.PCs,1); % if ACh == 1 -> HPC fires at theta peak (maximal depolarisation)
    hpc_spike = zeros(p.PCs,length(in.time));
    
    % prepare spontaneous activity to add
    rng('shuffle')
    rndPois_hpc = rand(p.PCs,length(in.time));
    randAct_hpc = rndPois_hpc < SpontFiring*p.dt; clear rndPois_hpc
    
    for i = 1:p.PCs
        hpc_spike(i,round(hpc_times/p.dt)) = 1;
    end
    hpc_spike = hpc_spike + randAct_hpc;
    hpc_spike = heaviside(hpc_spike - 0.1);
    hpcFiring_train_safe = hpc_spike;
    hpcFiring_train_safe(1+round(p.PCs/2):end,:) = 0;
    hpcFiring_train_dang = hpc_spike;
    hpcFiring_train_dang(1:round(p.PCs/2),:) = 0;
    clear hpc_spike i
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % postsynaptic activity %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    % spontaneous activity (0.85 Hz in both safe and dangerous compartments)
    rng('shuffle')
    rndPois_safe = rand(p.FCs,length(in.time));
    rndPois_dang = rand(p.FCs,length(in.time));
    amyFire_safe = rndPois_safe < SpontFiring*p.dt; clear rndPois_safe
    amyFire_dang = rndPois_dang < SpontFiring*p.dt; clear rndPois_dang
    
    % add threat-related spikes to dangerous compartment
    for i = 1:p.FCs
        timeIdxVecShuff = randperm(1000);
        addConstTime = 0:1000:(1000*(numTrainSpikes-1));
        spikeIdx = timeIdxVecShuff(1:numTrainSpikes) + addConstTime;
        spikeBoo = false(1,length(in.time));
        spikeBoo(spikeIdx) = true;
        amyFire_dang(i,:) = amyFire_dang(i,:) + spikeBoo;
    end, clear spikeIdxdx spikeBoo timeIdxVecShuff i
    
    %%%%%%%%%%%%%%%%%%
    % time evolution %
    %%%%%%%%%%%%%%%%%%
    
    % simulate training - safe compartment
    in.in_hpc = hpcFiring_train_safe;
    [W_AftSafe{f},~,acprops(f,1).pre_safe] = plth_evolve(p,in,amyFire_safe,W_0);
    
    % simulate training - dangerous compartment
    in.in_hpc = hpcFiring_train_dang;
    [W_AftAll{f},~,acprops(f,1).pre_dang] = plth_evolve(p,in,amyFire_dang,W_AftSafe{f});
    
    %% recall
    
    % timing stuff
    in.t_end = p.TimeRecall;
    in.time  = p.dt:p.dt:in.t_end;
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % presynaptic activity %
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    % Acetylcholine is low during recall (= familiar environment)
    in.ACh = 0;
    
    % theta frequency (constant during recall)
    in.freq = 5;
    T = 1/in.freq;
    
    % hippocampal input at theta frequency
    hpcSpkTime_recall = repmat((T+(1-in.ACh)*T/2):T:in.t_end,p.PCs,1); % if ACh == 1 -> HPC fires at theta peak (maximal depolarisation)
    hpcFiring_recall = zeros(p.PCs,length(in.time));
    for i = 1:p.PCs
        hpcFiring_recall(i,round(hpcSpkTime_recall/p.dt)) = 1;
    end
    
    % add spontaneous activity
    rng('shuffle')
    rndPois_hpc = rand(p.PCs,length(in.time));
    randAct_hpc = rndPois_hpc < SpontFiring*p.dt; clear rndPois_hpc
    
    hpcFiring_recall = hpcFiring_recall + randAct_hpc;
    hpcFiring_recall = heaviside(hpcFiring_recall - 0.1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % postsynaptic activity %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    % only spontaneous activity during recall
    rng('shuffle')
    rndPois = rand(p.FCs,length(in.time));
    amyFire_recall = rndPois < SpontFiring*p.dt;
    
    %%%%%%%%%%%%%%%%%%
    % time evolution %
    %%%%%%%%%%%%%%%%%%
    
    % safe
    in.in_hpc = hpcFiring_recall;
    in.in_hpc(1+round(p.PCs/2):end,:) = 0; % <--- retain only place cells tuned on safe compartment
    [~,frTestSafe{f},acprops(f,1).post_safe] = plth_evolve(p,in,amyFire_recall,W_AftAll{f}); %#ok<*SAGROW>
    
    % dang
    in.in_hpc = hpcFiring_recall;
    in.in_hpc(1:round(p.PCs/2),:) = 0; % <--- retain only place cells tuned on dang compartment
    [~,frTestDang{f},acprops(f,1).post_dang] = plth_evolve(p,in,amyFire_recall,W_AftAll{f});
    
    %% estimate freezing time
    numCellsAboveThresh_safe = freezeThr < frTestSafe{f};
    numCellsAboveThresh_dang = freezeThr < frTestDang{f};
    
    % average firing rate
    frRecall_mean_safe(f) = mean(frTestSafe{f}); %#ok<*SAGROW>
    frRecall_mean_dang(f) = mean(frTestDang{f});
    
    % mean and sem over sweeps
    percCellsAboveThresh_safe(f) = mean(numCellsAboveThresh_safe,1);
    percCellsAboveThresh_dang(f) = mean(numCellsAboveThresh_dang,1);
    clear numCellsAboveThresh_safe numCellsAboveThresh_dang amyFire_recall
    
end, clear ach f i T freezeThr

%% plot synaptic weight distribution
% pd_x = 0:0.02:2;
% figure('color',[1 1 1])
% hold on
% for f = 1:length(in_freq)
%     values_safe = W_AftAll{f}(1:round(p.PCs/2),:);
%     pd_safe = fitdist(values_safe(:),'Kernel');
%     values_dang = W_AftAll{f}(1+round(p.PCs/2):end,:);
%     pd_dang = fitdist(values_dang(:),'Kernel');
%     pd_y_safe = pdf(pd_safe,pd_x);
%     pd_y_dang = pdf(pd_dang,pd_x);
%     plot3(pd_x,in_freq(f)*ones(length(pd_x)),pd_y_safe,'color','b')
%     plot3(pd_x,0.1+in_freq(f)*ones(length(pd_x)),pd_y_dang,'color','r')
%     set(gca,'fontsize',14)
%     xlabel('W'),ylabel('Frequency')
% end
% view(3)

%% plot firing rate distribution
pd_x = 0:0.1:15;
figure('color',[1 1 1])
hold on
for f = 1:length(in_freq)
    values_safe = frTestSafe{f};
    pd_safe = fitdist(values_safe(:),'Kernel');
    values_dang = frTestDang{f};
    pd_dang = fitdist(values_dang(:),'Kernel');
    pd_y_safe = pdf(pd_safe,pd_x);
    pd_y_dang = pdf(pd_dang,pd_x);
    plot3(pd_x,in_freq(f)*ones(length(pd_x)),pd_y_safe,'color','b')
    plot3(pd_x,0.1+in_freq(f)*ones(length(pd_x)),pd_y_dang,'color','r')
    set(gca,'fontsize',14)
    xlabel('Firing rate'),ylabel('Frequency')
end, clear values_safe values_dang
view(3)

%% plot synaptic weight histograms
figure('color',[1 1 1])
hold on
for f = 1:length(in_freq)
    W_safe = mean(W_AftAll{f}(1:round(p.PCs/2),:),1);
    W_dang = mean(W_AftAll{f}(1+round(p.PCs/2):end,:),1);
    mean_W_safe(f) = mean(W_safe(:));
    mean_W_dang(f) = mean(W_dang(:));
    
    % 95% CI for safe
    sem_safe = std(W_safe(:))/sqrt(length(W_safe(:)));  % Standard Error
    ts = tinv([0.025  0.975],length(W_safe(:))-1);      % T-Score
    ci_W_safe(f,:) = mean(W_safe(:)) + ts*sem_safe;          % Confidence Intervals
    
    % 95% CI for dang
    sem_dang = std(W_dang(:))/sqrt(length(W_dang(:)));  % Standard Error
    ts = tinv([0.025  0.975],length(W_dang(:))-1);      % T-Score
    ci_W_dang(f,:) = mean(W_dang(:)) + ts*sem_dang;          % Confidence Intervals
    
end
Ws = [mean_W_safe',mean_W_dang'];
CIs_upper = [ci_W_safe(:,2), ci_W_dang(:,2)];
CIs_lower = [ci_W_safe(:,1), ci_W_dang(:,1)];
CIs_centr = (CIs_upper + CIs_lower)/2;
CIs_width = CIs_upper - CIs_centr; % clear CIs_upper CIs_lower ci_W_safe ci_W_dang W_safe W_dang
hb = bar(Ws);
myColors = [0,0,1;1,0,0];
for ib = 1:numel(hb)
    xData = hb(ib).XData+hb(ib).XOffset;
    errorbar(xData',CIs_centr(:,ib),CIs_width(:,ib),'k.')
    hb(ib).FaceColor = myColors(ib,:);
end
set(gca,'fontsize',14)
xlabel('Theta frequency'),ylabel('W')
legend('Safe comp.','Threatening comp.','location','northwest'), legend boxoff
set(gca,'fontsize',18)

%% plot firing rate scatter
figure('color',[1 1 1])
hold on
for f = 1:length(in_freq)
    scatter(in_freq(f)-0.1+0.025*randn(length(frTestSafe{f}),1),frTestSafe{f},50,'b','filled')
    scatter(in_freq(f)+0.1+0.025*randn(length(frTestSafe{f}),1),frTestDang{f},50,'r','filled')
end
xlabel('Theta frequency'),ylabel('Firing rate'),ylim([-1, 1.1*max(frTestDang{end}(:))])
% legend('Safe comp.','Threatening comp.','location','northwest'), legend boxoff
set(gca,'fontsize',18)

%% plot percentage of cells above threshold
% figure('color',[1 1 1]),hold on
% bar(in_freq,[percCellsAboveThresh_safe' percCellsAboveThresh_dang'])
% xlabel('Training theta frequency (Hz)'),ylabel('% cells > firing thresh.')
% legend('Safe comp.','Threatening comp.','location','northwest'), legend boxoff
% set(gca,'fontsize',18)

%% plot theta coherence

% for i = 1:length(in_freq)
%     idxCoh = i;
%     figure
%
%     subplot(2,2,1)
%     plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).pre_safe.power,'k','LineWidth',3)
%     ylim([0 0.05]),title(['f = ', num2str(in_freq(i)),'- PreSafe'])
%     axis square
%
%     subplot(2,2,2)
%     plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).post_safe.power,'k','LineWidth',3)
%     ylim([0 0.05]),title(['f = ', num2str(in_freq(i)),'- PostSafe'])
%     axis square
%
%     subplot(2,2,3)
%     plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).pre_dang.power,'k','LineWidth',3)
%     ylim([0 0.05]),title(['f = ', num2str(in_freq(i)),'- PreDang'])
%     axis square
%
%     subplot(2,2,4)
%     plot(acprops(idxCoh).pre_safe.freq,acprops(idxCoh).post_dang.power,'k','LineWidth',3)
%     ylim([0 0.05]),title(['f = ', num2str(in_freq(i)),'- PostDang'])
%     axis square
%
% end

figure('color',[1 1 1])

%% low freq
yMax = 1;

subplot(2,4,1)
plot(acprops(2).pre_safe.freq,acprops(1).pre_safe.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(1)),'- PreSafe'])
axis square

subplot(2,4,2)
plot(acprops(2).pre_safe.freq,acprops(1).post_safe.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(1)),'- PostSafe'])
axis square

subplot(2,4,5)
plot(acprops(2).pre_safe.freq,acprops(1).pre_dang.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(1)),'- PreDang'])
axis square

subplot(2,4,6)
plot(acprops(2).pre_safe.freq,acprops(1).post_dang.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(1)),'- PostDang'])
axis square

%% high freq

subplot(2,4,3)
plot(acprops(2).pre_safe.freq,acprops(2).pre_safe.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(2)),'- PreSafe'])
axis square

subplot(2,4,4)
plot(acprops(2).pre_safe.freq,acprops(2).post_safe.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(2)),'- PostSafe'])
axis square

subplot(2,4,7)
plot(acprops(2).pre_safe.freq,acprops(2).pre_dang.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(2)),'- PreDang'])
axis square

subplot(2,4,8)
plot(acprops(2).pre_safe.freq,acprops(2).post_dang.power,'k','LineWidth',3)
ylim([0 yMax]),title(['f = ', num2str(in_freq(2)),'- PostDang'])
axis square

%% Nice figure for paper
endFreq = 52;
yMax = 10;
fontSize = 16;
colorSafe = [0.6 0.6 0.8];
colorDang = [0.2 0.2 0.7];
freqSpan = acprops(1).pre_safe.freq(1:endFreq);
figure('color',[1 1 1])

% safe low freq
subplot(4,4,[1 2 5 6]), hold on
plot(freqSpan,acprops(1).pre_safe.power(1:endFreq),'color',[0.6 0.6 0.8],'LineWidth',5,'LineStyle','-')
plot(freqSpan,acprops(1).post_safe.power(1:endFreq),'color',[0.2 0.2 0.7],'LineWidth',7,'LineStyle',':')
set(gca,'fontsize',fontSize), legend('Pre-conditioning','Post-conditioning','location','northwest'), legend boxoff
ylim([-0.01 yMax])

% dang low freq
subplot(4,4,[3 4 7 8]), hold on
plot(freqSpan,acprops(1).pre_dang.power(1:endFreq),'color',[0.6 0.6 0.8],'LineWidth',5,'LineStyle','-')
plot(freqSpan,acprops(1).post_dang.power(1:endFreq),'color',[0.2 0.2 0.7],'LineWidth',7,'LineStyle',':')
set(gca,'fontsize',fontSize)
ylim([-0.01 yMax])

% safe high freq
subplot(4,4,[9 10 13 14]), hold on
plot(freqSpan,acprops(2).pre_safe.power(1:endFreq),'color',[0.6 0.6 0.8],'LineWidth',5,'LineStyle','-')
plot(freqSpan,acprops(2).post_safe.power(1:endFreq),'color',[0.2 0.2 0.7],'LineWidth',7,'LineStyle',':')
set(gca,'fontsize',fontSize)
ylim([-0.01 yMax])
xlabel('Frequency (Hz)')
ylabel('Power (a.u.)')

% dang high freq
subplot(4,4,[11 12 15 16]), hold on
plot(freqSpan,acprops(2).pre_dang.power(1:endFreq),'color',[0.6 0.6 0.8],'LineWidth',5,'LineStyle','-')
plot(freqSpan,acprops(2).post_dang.power(1:endFreq),'color',[0.2 0.2 0.7],'LineWidth',7,'LineStyle',':')
set(gca,'fontsize',fontSize)
ylim([-0.01 yMax])

%% plot power in theta band (4-10Hz)
freqThetaBand = 23:52;

% take avgs
thetaPower.low.safe.pre = mean(acprops(1).pre_safe.power(freqThetaBand));
thetaPower.low.safe.post = mean(acprops(1).post_safe.power(freqThetaBand));
thetaPower.low.dang.pre = mean(acprops(1).pre_dang.power(freqThetaBand));
thetaPower.low.dang.post = mean(acprops(1).post_dang.power(freqThetaBand));
thetaPower.hig.safe.pre = mean(acprops(2).pre_safe.power(freqThetaBand));
thetaPower.hig.safe.post = mean(acprops(2).post_safe.power(freqThetaBand));
thetaPower.hig.dang.pre = mean(acprops(2).pre_dang.power(freqThetaBand));
thetaPower.hig.dang.post = mean(acprops(2).post_dang.power(freqThetaBand));

aaa(1) = mean(acprops(1).pre_safe.power(freqThetaBand));
aaa(2) = mean(acprops(1).post_safe.power(freqThetaBand));
aaa(3) = mean(acprops(1).pre_dang.power(freqThetaBand));
aaa(4) = mean(acprops(1).post_dang.power(freqThetaBand));
aaa(5) = mean(acprops(2).pre_safe.power(freqThetaBand));
aaa(6) = mean(acprops(2).post_safe.power(freqThetaBand));
aaa(7) = mean(acprops(2).pre_dang.power(freqThetaBand));
aaa(8) = mean(acprops(2).post_dang.power(freqThetaBand));

figure('color',[1 1 1])
bar(aaa)