function [out] = autoCorrProps(spikeTimes)

% Provide some settings
acBinSize       = 0.01;                                                     % Auto correlation bin size (s) - to match bat paper
acWindow        = 1;                                                        % Portion of autocorr window to use for power spec
maxFreqRange    = 50;                                                       % Top limit to truncate powerspec - calc power against this - dropped
smthKern        = 2;                                                        % Size of Gaussian smoothing window
s2nWdth         = 2;                                                        % Width of band around theta peak to calc signal to noise for 1hz is normal
fftLength       = 512;

% Construct a spike train histogram to by used for the autocorrelogram
nHistBins       = max(spikeTimes) / acBinSize;                              % Work out number of bins for the histogram of spike times
binCentres      = (0:acBinSize:(acBinSize * (nHistBins-1)))+ acBinSize/2;   % Identify the centre point of each bin
spkTrHist       = hist(spikeTimes, binCentres(:));                          % Calculate the spike time histogram
% clear           binCentres

[RawAutoCorr,Lags] = xcorr(spkTrHist,'unbiased');                           % Generate the raw temporal auto-correlation
% clear           spkTrHist

% Whittle it down to just one side, and only the window length of interest
MidPoint        = ceil(length(RawAutoCorr)/2);                              % Find the mid-point (i.e. zero lag) of the auto-correlation
BinsToKeep      = acWindow/acBinSize;                                       % Identify the auto-correlation bins to be kept
RawAutoCorr     = RawAutoCorr(MidPoint:MidPoint + BinsToKeep);              % Extract that part of the raw auto-correlation
Lags            = Lags(MidPoint:MidPoint + BinsToKeep);                     % Extract that part of the lags
Lags            = Lags .* acBinSize .* 1000;                                % Change the lag values to ms 

% Mean normalise, do the Fourier transform and extract the power spectra
MeanNormAC      = [0 RawAutoCorr(2:end) - mean(RawAutoCorr(2:end))];
fftRes          = fft(MeanNormAC, fftLength);
Power           = (fftRes.*conj(fftRes))/length(fftRes);
Freqs           = ([0:(length(fftRes)-1)]*((1/acBinSize)/length(fftRes)))';
Freqs           = Freqs(Freqs<=maxFreqRange);
Power           = Power(1:length(Freqs));       % ...and matching power

% Smooth to make more reliable
Power           = fastsmooth(Power,smthKern,3);

% Generate output
out.lags        = Lags; 
out.meanNormAC  = MeanNormAC;
out.freq        = Freqs;
out.power       = Power;