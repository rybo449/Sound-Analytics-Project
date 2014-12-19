
# coding: utf-8

# In[11]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
from scipy.io import wavfile
import scipy.stats as sps
import scipy.signal as signal
import matplotlib.pyplot as plt
import mpld3
import matplotlib.pylab as pylab


# In[2]:

#mpld3.enable_notebook()
pylab.rcParams['figure.figsize'] = 12, 8
pylab.rcParams['font.size'] = 16


# In[3]:

fs = 8000
dur = 2

A = [1.0, 0.75, 0.5, 5]
freq = [110, 220, 440]
sample_len = fs*dur

f, axarr = plt.subplots(2,2, sharex = True)#, sharey = True)
# In[4]:
full_sines = np.zeros(sample_len)
for x in xrange(4):
	sine_samples = np.zeros(sample_len)


	# In[5]:

	T = np.linspace(0, dur, num=sample_len)


	
	if x == 3:
		for i in range(sample_len):
			sine_samples[i] = A[x] * np.random.uniform(-1, 1)
		axarr[1,1].plot(T[0:500]*1000, sine_samples[0:500], marker='o')
		axarr[1,1].set_ylim([-6.1, 6.1])
		axarr[1,1].set_xlim([0, T[500]*1000])
		axarr[1,1].grid(True)
		axarr[1,1].set_title('White Noise')
		axarr[1,1].set_xlabel('Time (ms)')
		axarr[1,1].set_ylabel('Amplitude')	
		comb_samples = np.int16(sine_samples * 32768)
		wavfile.write('audio/noise.wav', fs, comb_samples)	
	elif x == 0:
		for i in range(sample_len):
			sine_samples[i] = A[x] * np.sin(2 * np.pi * freq[x] * T[i])
		last_samples = sine_samples

	# In[10]:

		axarr[0,0].plot(T[0:500]*1000, sine_samples[0:500], marker='o')
		axarr[0,0].set_ylim([-1.1, 1.1])
		axarr[0,0].set_xlim([0, T[500]*1000])
		axarr[0,0].grid(True)
		axarr[0,0].set_title('%i(Hz) sine wave'%(freq[x]))
		axarr[0,0].set_xlabel('Time (ms)')
		axarr[0,0].set_ylabel('Amplitude')
		comb_samples = np.int16(sine_samples * 32768)
		wavfile.write('audio/110.wav', fs, comb_samples)
	elif x == 1:
		for i in range(sample_len):
			sine_samples[i] = A[x] * np.sin(2 * np.pi * freq[x] * T[i])
		last_samples = sine_samples

	# In[10]:

		axarr[0,1].plot(T[0:500]*1000, sine_samples[0:500], marker='o')
		axarr[0,1].set_ylim([-1.1, 1.1])
		axarr[0,1].set_xlim([0, T[500]*1000])
		axarr[0,1].grid(True)
		axarr[0,1].set_title('%i(Hz) sine wave'%(freq[x]))
		axarr[0,1].set_xlabel('Time (ms)')
		axarr[0,1].set_ylabel('Amplitude')
		comb_samples = np.int16(sine_samples * 32768)
		wavfile.write('audio/220.wav', fs, comb_samples)
	elif x == 2:
		for i in range(sample_len):
			sine_samples[i] = A[x] * np.sin(2 * np.pi * freq[x] * T[i])
		last_samples = sine_samples

	# In[10]:

		axarr[1,0].plot(T[0:500]*1000, sine_samples[0:500], marker='o')
		axarr[1,0].set_ylim([-1.1, 1.1])
		axarr[1,0].set_xlim([0, T[500]*1000])
		axarr[1,0].grid(True)
		axarr[1,0].set_title('%i(Hz) sine wave'%(freq[x]))
		axarr[1,0].set_xlabel('Time (ms)')
		axarr[1,0].set_ylabel('Amplitude')
		comb_samples = np.int16(sine_samples * 32768)
		wavfile.write('audio/440.wav', fs, comb_samples)
	full_sines = full_sines + sine_samples
f.tight_layout()
#plt.xlabel('Time (ms)')
#plt.ylabel('Amplitude')
plt.show()


comb_samples = full_sines / np.max(full_sines)
plt.plot(T[0:100]*1000, comb_samples[0:100], marker='o')
plt.ylim([-1.1, 1.1])
plt.title('Combined and normalized white noise and sine waves')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)
comb_samples = np.int16(comb_samples * 32768)
wavfile.write('audio/comb.wav', fs, comb_samples)
plt.show()



dft_output = np.fft.rfft(full_sines)
print dft_output[1]


# In[14]:

magnitude_spectrum = [np.sqrt(i.real**2 + i.imag**2)/len(dft_output) for i in dft_output]
print magnitude_spectrum[1]


# In[15]:

freqs = np.linspace(0,fs/2, num=len(dft_output))


# In[21]:
plt.plot(freqs, magnitude_spectrum, 'g')
plt.grid(True)


plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.suptitle('Magnitude spectrum of 1000Hz sine wave')
	
plt.show()
thresh = np.mean(magnitude_spectrum) + (5 * np.std(magnitude_spectrum))

plt.plot(freqs, magnitude_spectrum, 'g')
plt.xlim([0, 500])
plt.axhspan(0.001, thresh, facecolor='0.5', alpha=0.5)
plt.title('Magnitude spectrum of combined sinusoids and white noise signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()


f, axarr = plt.subplots(2,2, sharex = True)
for x in xrange(4):


	# In[9]:

	#Fourier Transformation
	if x != 3:
		for i in range(sample_len):
			sine_samples[i] = A[x] * np.sin(2 * np.pi * freq[x] * T[i])
	else:
		for i in range(sample_len):
			sine_samples[i] = A[x] * np.random.uniform(-1, 1)
	# In[12]:

	dft_output = np.fft.rfft(sine_samples)
	print dft_output[1]


	# In[14]:

	magnitude_spectrum = [np.sqrt(i.real**2 + i.imag**2)/len(dft_output) for i in dft_output]
	print magnitude_spectrum[1]


	# In[15]:

	freqs = np.linspace(0,fs/2, num=len(dft_output))


	# In[21]:
	if x == 0:
		axarr[0,0].plot(freqs, magnitude_spectrum, 'g')
		axarr[0,0].grid(True)
	if x == 1:
		axarr[0,1].plot(freqs, magnitude_spectrum, 'g')
		axarr[0,1].grid(True)
	if x == 2:
		axarr[1,0].plot(freqs, magnitude_spectrum, 'g')
		axarr[1,0].grid(True)
	if x == 3:
		axarr[1,1].plot(freqs, magnitude_spectrum, 'g')
		axarr[1,1].grid(True)

	#plt.xlim([0, 4000])
f.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
f.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.suptitle('Magnitude spectrum of 1000Hz sine wave')
	
plt.show()


	


