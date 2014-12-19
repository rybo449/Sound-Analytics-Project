import numpy as np
from scipy.io import wavfile
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.spatial import distance

pylab.rcParams['figure.figsize'] = 16, 8
pylab.rcParams['font.size'] = 16

fs, samples = wavfile.read('audio/brooklyn_street.wav')

dur = float(len(samples)) / fs
sample_len = len(samples)

samples = samples / 32768.0
T = np.linspace(0, dur, num=sample_len)

ab_samples = np.abs(samples)

plt.plot(T, ab_samples)
plt.ylim([0, 1.5])
plt.xlim([0, np.max(T)])
plt.title('Brooklyn street Absolute Amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Amplitude')
plt.show()

window_size = 5700
hop_size = window_size/2

print 'Window length in ms: ', '%.2f' % (float(window_size)/float(fs)*1000)

w = np.hanning(window_size)
plt.plot(w * ab_samples[fs*10:(fs*10)+window_size])
plt.xlim([0, window_size])
plt.title('Hanning window')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

window_avg = [np.mean(w*ab_samples[i:i+window_size]) for i in range(0, len(ab_samples), hop_size)if i+window_size <= len(ab_samples)]
print 'Length of audio samples array:    ', len(ab_samples)
print 'Length of windowed average array: ', len(window_avg)
T = np.linspace(0, dur, num=len(window_avg))

plt.plot(T, window_avg)
plt.xlim([0, np.max(T)])
plt.title('Brooklyn street')
plt.xlabel('Time (s)')
plt.ylabel('Absolute amplitude')
plt.show()

dft_output = np.fft.rfft(w * samples[0:window_size])
magnitude_spectrum = [np.sqrt(i.real**2 + i.imag**2)/len(dft_output) for i in dft_output]
freqs = np.linspace(0,fs/2, num=len(dft_output))

thresh = np.mean(magnitude_spectrum) + (5 * np.std(magnitude_spectrum))

plt.plot(freqs, magnitude_spectrum, 'g')
#plt.xlim([0, 500])
plt.axhspan(0, thresh, facecolor='0.5', alpha=0.5)
plt.title('Magnitude spectrum of Brooklyn Street')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

com_mag_spec_thr = (magnitude_spectrum > thresh) * magnitude_spectrum
diff_spec = np.diff(com_mag_spec_thr)
diff_spec = np.roll(diff_spec,1)
plt.xlim([0, 500])
plt.plot(diff_spec)
plt.show()
match = np.convolve(np.sign(diff_spec), [-1,1])
match = np.roll(match, 1)
print np.argmax(diff_spec)
idx = np.nonzero(match>0)[0]-2
print freqs[idx]





mag_spec_array_complex = [np.fft.rfft(w * samples[i:i + window_size]) for i in range(0, len(samples), hop_size)if i + window_size <= len(samples)]

mag_spec_array = []
for x in range(0, len(mag_spec_array_complex)):
    mag_spec_array.append([np.sqrt(i.real**2 + i.imag**2)/len(mag_spec_array_complex[x]) for i in mag_spec_array_complex[x]])
spec_cent_array = np.zeros(len(mag_spec_array))

for x in range(0, len(mag_spec_array)):
    spec_cent_array[x] = np.sum(mag_spec_array[x]*freqs) / np.sum(mag_spec_array[x])


fig, ax1 = plt.subplots()

ax1.plot(T, window_avg)

for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Absolute amplitude')

ax2 = ax1.twinx()

ax2.plot(T,spec_cent_array, 'r', alpha=0.7)
ax2.set_ylabel('Spectral centroid (Hz)')

for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.xlim([0, np.max(T)])
plt.show()


spec_array = np.array(mag_spec_array).T
plt.imshow(spec_array, aspect='auto', origin='lower')
plt.set_cmap('gnuplot')
plt.show()
from scipy.signal import chirp

dur = 3

chir_samp_len = fs * dur


#dur = float(len(samples)) / fs
#sample_len = len(samples)

#samples = samples / 32768.0
#T = np.linspace(0, dur, num=sample_len)
#plt.plot(T, freqs)

plt.specgram(samples, NFFT = window_size, noverlap = hop_size, Fs = fs, mode = 'magnitude', scale = 'dB');
plt.ylim([0, 17000])
#plt.xlim([0, np.max(T)])
plt.set_cmap('gist_rainbow_r')
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(b=True, which='both', color='0.6',linestyle='--')
plt.show()


T = np.linspace(0, dur, num=len(window_avg))

spec_flux_array = np.zeros(len(mag_spec_array) - 1)

for x in range(0, len(mag_spec_array) - 1):
    spec_flux_array[x] = distance.euclidean(mag_spec_array[x], mag_spec_array[x+1])

plt.plot(T[0:len(spec_flux_array)],spec_flux_array, 'r')
plt.xlim([0, np.max(T)])
plt.title('Brooklyn street - spectral flux over time')
plt.xlabel('Time (ms)')
plt.ylabel('Spectral flux')
plt.show()