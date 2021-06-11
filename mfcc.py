import numpy as np
import scipy.fftpack as fftpack
import librosa
import scipy.io.wavfile
import time

def pre_emphasis(signal, filter_coefficient):
    return np.append(signal[0], signal[1:] - filter_coefficient * signal[:-1])  # returns pre-emphasized signal


#
# frame length - in seconds
# frame hop - in seconds
#
def framing(signal, sampling_rate, frame_length, frame_hop):
    frame_size = frame_length * sampling_rate  # Convert from seconds to samples
    frame_hop = frame_hop * sampling_rate
    signal_length = len(signal)
    frame_length = int(round(frame_size))
    frame_hop = int(round(frame_hop))
    num_frames = int((signal_length - frame_size) / frame_hop) + 1  # calculate number of frames

    pad_signal_length = int(num_frames * frame_hop + frame_size)
    pad_zeros = np.zeros(pad_signal_length - signal_length)
    pad_signal = np.append(signal, pad_zeros)  # to make all frames have equal number of samples

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_hop, frame_hop), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames


#
# frame length in samples
#
def hamming_window(frames, frame_length):
    frames *= np.hamming(frame_length)
    return frames


def dft(frames):
    frames_magnitude = np.absolute(np.fft.rfft(frames))  # magnitude of the signal frames
    return frames_magnitude


def power_spectrum(frames, stft_points):
    pow_spectrum_frames = ((1.0 / stft_points) * ((frames) ** 2))  # power spectrum of signal frames
    return pow_spectrum_frames


# Returs frames filetered using mel filter banks
# filters_num - number of filter banks
# sample_rate - signal sample rate
# frame_size - size of a frame used in dft
# def mel_filter_banks(filters_count, sample_rate, frames):
#     mel_min = 0
#     mel_max = freqToMelFreq(sample_rate / 2)
#     mel_bands = np.linspace(mel_min, mel_max, filters_count + 2)
#     freq_bands = np.array(list(map(melFreqToFreq, mel_bands)))
#     frame_size = len(frames[0])
#
#     # frequency of each magnitude
#     mag_freqs = np.linspace(0, sample_rate / 2, frame_size)
#     filtered_frames = np.zeros((len(frames), frame_size))
#     # apply filterbank to each magnitude in each frame
#     for f, frame in enumerate(frames):
#         for m, mag_freq in enumerate(mag_freqs):
#             for fb in range(1, len(freq_bands) - 1):
#                 left = freq_bands[fb - 1]
#                 center = freq_bands[fb]
#                 right = freq_bands[fb + 1]
#                 if mag_freq >= left and mag_freq <= center:
#                     weight = 2 * (mag_freq - left) / (center - left)
#                     filtered_frames[f, m] += frame[m] ** 2 * weight
#                 elif mag_freq <= right and mag_freq > center:
#                     weight = 2 * (right - mag_freq) / (right - center)
#                     filtered_frames[f, m] += frame[m] ** 2 * weight
#
#     filtered_frames = np.where(filtered_frames == 0, np.finfo(float).eps, filtered_frames)
#     filtered_frames = 20 * np.log10(filtered_frames)
#
#     return filtered_frames
#

def mel_filter_banks(num_filters, sample_rate, frames):
    mel_min = 0
    mel_max = freqToMelFreq(sample_rate / 2)
    mel_bands = np.linspace(mel_min, mel_max, num_filters + 2)
    freq_bands = np.array(list(map(melFreqToFreq, mel_bands)))
    frame_size = len(frames[0])

    # frequency of each magnitude
    mag_freqs = np.linspace(0, sample_rate / 2, frame_size)
    filtered_frames = np.zeros((len(frames), frame_size))
    # apply filterbank to each magnitude in each frame
    fbank = np.zeros((num_filters, int(np.floor(sample_rate / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])


    for f, frame in enumerate(frames):
        for m, mag_freq in enumerate(mag_freqs):
            for fb in range(1, len(freq_bands) - 1):
                left = freq_bands[fb - 1]
                center = freq_bands[fb]
                right = freq_bands[fb + 1]
                if mag_freq >= left and mag_freq <= center:
                    weight = 2 * (mag_freq - left) / (center - left)
                    filtered_frames[f, m] += frame[m] ** 2 * weight
                elif mag_freq <= right and mag_freq > center:
                    weight = 2 * (right - mag_freq) / (right - center)
                    filtered_frames[f, m] += frame[m] ** 2 * weight

    filtered_frames = np.where(filtered_frames == 0, np.finfo(float).eps, filtered_frames)
    filtered_frames = 20 * np.log10(filtered_frames)

    return filtered_frames



def freqToMelFreq(freq):
    return 2595 * np.log10(1 + freq / 700)


def melFreqToFreq(mel_freq):
    return 700 * (np.power(10, mel_freq / 2595) - 1)


def dct(frames):
    return fftpack.dct(frames)

def deltaCoefficients(coefficients):
    deltas = [coefficients[0]]
    for i in range(1,len(coefficients)-1):
        deltas.append(coefficients[i]-coefficients[i-1])
    return deltas

def deltaCoefficientsForEachFrame(mfccs):
    deltas=[]
    for frame in mfccs:
        deltas.append(deltaCoefficients(frame))
    return deltas


"""
:param frame_length_sec: frame length in secs
:param frame_hop_sec: frame hope in sec 
:param num_mfccs: number of mfccs
:param signal: input signal
:param sample_rate: sample rate of signal 
"""

def getMfccs(signal,sample_rate,frame_length_sec,frame_hop_sec,num_filterbanks,num_mfccs):
    start = time.time();
    signal = pre_emphasis(signal, 0.95)
    frames = framing(signal, sample_rate, frame_length_sec, frame_hop_sec)
    frames_windowed = hamming_window(frames, int(frame_length_sec * sample_rate))
    frames_magnitude = dft(frames_windowed)
    frames_filtered = mel_filter_banks(num_filterbanks, sample_rate, frames_magnitude)
    coefficients = dct(frames_filtered)[:, 1:num_mfccs + 1]
    stop = time.time()
    print("Time: "+str(stop-start))
    return coefficients