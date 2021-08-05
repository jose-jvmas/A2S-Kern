import joblib
import numpy as np
import matplotlib.pyplot as plt
from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram, Spectrogram

#Joblib settings:
memory = joblib.memory.Memory('./joblib_cache', mmap_mode='r', verbose=1)

@memory.cache
def get_x_from_file(audiofilename):
	audio_options = dict(
		num_channels=1,
		sample_rate=44100,
		filterbank=LogarithmicFilterbank,
		frame_size=4096,
		fft_size=4096,
		hop_size=441 * 4,  # 25 fps
		num_bands=48,
		fmin=30,
		fmax=8000.0,
		fref=440.0,
		norm_filters=True,
		unique_filters=True,
		circular_shift=False,
		norm=True
	)

	dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])
	x = LogarithmicFilteredSpectrogram(audiofilename, **audio_options)

	return x


if __name__ == '__main__':
	x = get_x_from_file('Saarland/Data/Audios/Bach_BWV849-01_001_20090916-SMD.wav')
	x = np.flip(np.transpose(x),0)
	x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
	print("hello")