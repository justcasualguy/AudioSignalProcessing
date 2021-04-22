import numpy as np

def pre_emphasis(signal,filter_coefficient):
   return np.append(signal[0], signal[1:] - filter_coefficient * signal[:-1]) # returns pre-emphasized signal

#
#frame length - in seconds
#frame hop - in seconds
#
def framing(signal, sampling_rate, frame_length, frame_hop):
    frame_size= frame_length * sampling_rate,   # Convert from seconds to samples
    frame_hop= frame_hop * sampling_rate
    print(f"hop: {frame_hop}")
    print(f"size: {frame_size}")
    signal_length = len(signal)
    frame_length = int(round(frame_size))
    frame_hop= int(round(frame_hop))
    num_frames = int(signal_length - frame_size / frame_hop)+1 #calculate number of frames

    pad_signal_length = num_frames * frame_hop + frame_size
    pad_zeros = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, pad_zeros)  # to make all frames have equal number of samples

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_hop, frame_hop), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

#
# frame length in samples
#
def hamming_window(frames,frame_length):
    frames *= np.hamming(frame_length)
    return frames

def stft(frames,stft_points):
    magnitude_frames = np.absolute(np.fft.rfft(frames, stft_points))  # magnitude of the signal frames
    return magnitude_frames

def power_spectrum(frames,stft_points):
    pow_spectrum_frames = ((1.0 / stft_points) * ((frames) ** 2))  # power spectrum of signal frames
    return pow_spectrum_frames
