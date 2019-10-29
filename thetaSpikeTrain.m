%% Generate theta modulated Poisson spike train

f_theta     = 4;                                    % Theta frequency
pulses      = 1000;                                 % (Average) number of stimulation pulses
time        = 1 : pulses / f_theta * 1000;          % Time axis in milliseconds
rate        = 1 + sin(f_theta*2*pi.*time./1000);	% Rate function
rate        = rate ./ sum(rate) * pulses;           % Normalise rate function
spikes      = poissrnd(rate);                       % Poisson spike input
clear rate