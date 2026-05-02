import librosa
import numpy as np
from scipy.signal import butter, filtfilt

def resample_audio(audio_path, target_sr=16000):
    """Loads an audio file and forces it to the target sample rate."""
    y, sr = librosa.load(audio_path, sr=target_sr)
    return y, sr

def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """Applies a mathematical low-pass filter to remove high-pitch noise."""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def convert_to_model_input(y, target_length=16000):
    """Pads or trims the audio array so every file is exactly the same length."""
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y

def compute_logmel_spectrogram(y, sr, n_mels=128, hop_length=512):
    """Converts the 1D audio wave into a 2D Log-Mel Spectrogram image."""
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    logmel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return logmel_spectrogram