function [Connect_out,FiringRate,ACProps] = plth_evolve(p,in,spk_amy,Connect_in)
% Simulates fear learning in HPC-BLA during contextual fear conditioning
% ~~~
% GX Castegnetti --- 2018

%% Initialise dynamic variables
v  = p.v_rest*ones(p.PCs,p.FCs);      % membrane potential at the spines of fear cells (no BPAP)
Ca = zeros(p.PCs,p.FCs);              % calcium concentration
EPSP_pre_exp1_HPC = zeros(p.PCs,p.FCs); % parameters controlling AMPA synaptic currents
EPSP_pre_exp2_HPC = zeros(p.PCs,p.FCs);
NMDA_exp1 = zeros(p.PCs,p.FCs);       % parameters controlling NMDA synaptic currents
NMDA_exp2 = zeros(p.PCs,p.FCs);
BPAP_exp1 = zeros(p.FCs,1);           % parameters controlling BPAP dynamics
BPAP_exp2 = zeros(p.FCs,1);

%% assign memory to save some variables
EPSP_log  = NaN(p.PCs,p.FCs,length(in.time));
NMDA_log  = NaN(p.PCs,p.FCs,length(in.time));
BPAP_log  = NaN(p.FCs,length(in.time));
Ca_log    = NaN(p.PCs,p.FCs,length(in.time));
V_log     = NaN(p.FCs,length(in.time));

%% temporal evolution
Connect_up = Connect_in;
for t = 1:length(in.time) - 1
    
    %% hippocampal input
    now_hpc = logical(in.in_hpc(:,t));
    if sum(now_hpc) > 0
        now_hpc = repmat(now_hpc,1,p.FCs);
        EPSP_pre_exp1_HPC(now_hpc) = EPSP_pre_exp1_HPC(now_hpc) + 1;                 % update AMPA time decay (only when ACh is less than 1)
        EPSP_pre_exp2_HPC(now_hpc) = EPSP_pre_exp2_HPC(now_hpc) + 1;
        NMDA_exp1(now_hpc) = NMDA_exp1(now_hpc) + p.P_0*(1-NMDA_exp1(now_hpc));      % update NMDA time decay
        NMDA_exp2(now_hpc) = NMDA_exp2(now_hpc) + p.P_0*(1-NMDA_exp2(now_hpc));
    end
    
    %% amygdala spikes
    amySpikesNow = find(spk_amy(:,t));
    if ~isempty(amySpikesNow)
        BPAP_exp1(amySpikesNow) = BPAP_exp1(amySpikesNow) + 1;               % update BPAP time decay
        BPAP_exp2(amySpikesNow) = BPAP_exp2(amySpikesNow) + 1;
        v(:,amySpikesNow) = p.v_reset;
    end
        
    %% Compute membrane potential at the spines; DETERMINES PLASTICITY
    epsp            = 2*Connect_up.*p.A_EPSP*(1-in.ACh)/0.7.*(EPSP_pre_exp1_HPC-EPSP_pre_exp2_HPC); % AMPA current from HPC (modulated by ACh)
    bpap_foo        = (p.I_bs_f*BPAP_exp1 + p.I_bs_s*BPAP_exp2) * 100;       % BPAP voltage
    bpap            = repmat(bpap_foo',p.PCs,1);                             % BPAP voltage
    V               = bpap + epsp + p.v_E;                                   % total membrane potential at the spines - determines plasticity
    Vsoma           = (sum(V - bpap,1)/p.PCs)';                              % total membrane potential at the soma - determines firing
    nAct            = Vsoma > p.v_thr;
    v(nAct)         = p.v_reset;                                             % firing condition
    spk_amy(nAct,t+1) = 1;                                                   % record spike time
    B               = 1 ./ (1+exp(-0.062*V)*(p.Mg/3.57));                    % NMDA voltage dependency
    H               = B .* (V-p.V_r);
    I_nmda          = p.G_NMDA * H .* (p.I_f*NMDA_exp1 + p.I_s*NMDA_exp2);   % NMDA current
    
    %% Calcium concentration
    dCa             = I_nmda - (1/p.tau_Ca) * Ca;                            % calculate time derivative of [Ca2+]
    Ca              = Ca + dCa;                                              % update [Ca2+]
    
    %% Synaptic plasticity
    tau             = p.P_1./(p.P_2 + Ca.^p.P_3) + p.P_4; % learning rate
    eta             = 1./tau;
    sig1            = exp((Ca-p.alpha1).*p.beta1) ./ (1 + exp((Ca-p.alpha1).*p.beta1));
    sig2            = exp((Ca-p.alpha2).*p.beta2) ./ (1 + exp((Ca-p.alpha2).*p.beta2));
    OMEGA           = 0.25 + sig2 - 0.25*sig1;            % plasticity parameter
    dW              = eta.*(OMEGA - p.lambda*Connect_up); % calculate derivative of synaptic strength and update it
    Connect_up      = Connect_up + in.ACh * dW * p.dt;    % update synaptic weight
    
    %% Update decay
    EPSP_pre_exp1_HPC    = EPSP_pre_exp1_HPC * (1 - 1/p.tau_1);
    EPSP_pre_exp2_HPC    = EPSP_pre_exp2_HPC * (1 - 1/p.tau_2);
    NMDA_exp1            = NMDA_exp1 * (1 - 1/p.tau_f);
    NMDA_exp2            = NMDA_exp2 * (1 - 1/p.tau_s);
    BPAP_exp1            = BPAP_exp1 * (1 - 1/p.tau_bs_f);
    BPAP_exp2            = BPAP_exp2 * (1 - 1/p.tau_bs_s);
    
    %% Log dynamic variables where necessary
    NMDA_log(:,:,t) = p.I_f*NMDA_exp1 + p.I_s*NMDA_exp2;
    BPAP_log(:,t)   = bpap_foo;
    V_log(:,t)      = Vsoma';
    Ca_log(:,:,t)   = Ca;
    EPSP_log(:,:,t) = epsp;
    
end
figure,plot(squeeze(V_log(1,:)))

%% find average firing rate for every postsynaptic cell
numSpikes = sum(spk_amy,2);

%% output connectivity matrix, firing rates, and LFP
Connect_out = Connect_up;
FiringRate = numSpikes/in.time(end);
timeMat = repmat(in.time,p.FCs,1);
spikeTimes = timeMat(spk_amy);
spikeTimes = sort(spikeTimes(:));
if ~isempty(spikeTimes)
    ACProps = autoCorrProps(spikeTimes);
else
    ACProps = [];
end
hpcRates = sum(in.in_hpc(round(p.PCs/2)+1:end,:));
hpcRates = fastsmooth(hpcRates,50,3,0);
amyRates = histc(spikeTimes,in.time);
amyRates = fastsmooth(amyRates,50,3,0);
[ACProps.Cxy,ACProps.F]	= mscohere(hpcRates,amyRates,1000,500,512,1000);
clear spikeTimes hpcRates amyRates

end

