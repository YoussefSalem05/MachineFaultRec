"""
Industrial Acoustic Preprocessing Pipeline
Handles DSP filtering, trimming, normalization, and quality gating.
"""
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

# ============================================================
# --- 1. ENGINEERING TOOLS ---
# ============================================================

def remove_dc_offset(y):
    return y - np.mean(y)

def apply_lowpass_filter(y, cutoff=4000, sr=16000, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, y)

def apply_trim(y, top_db=26):
    y_trimmed, trim_indices = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed, trim_indices

def apply_peak_normalization(y):
    peak = np.max(np.abs(y))
    return y / (peak + 1e-9)

def extract_sliding_windows(y, sr, chunk_sec=0.5, overlap_sec=0.25):
    chunk_samples = int(chunk_sec * sr)
    step_samples  = int((chunk_sec - overlap_sec) * sr)
    chunks, starts = [], []
    for start in range(0, len(y) - chunk_samples + 1, step_samples):
        chunks.append(y[start:start + chunk_samples])
        starts.append(start)
    return chunks, starts

# ============================================================
# --- 2. CV QUALITY GATE ---
# ============================================================

def compute_rms(chunk):
    return np.sqrt(np.mean(chunk ** 2))

def is_uneven_chunk(chunk, n_windows=8, cv_threshold=1.2):
    window_size = len(chunk) // n_windows
    rms_values  = np.array([
        compute_rms(chunk[i * window_size:(i + 1) * window_size])
        for i in range(n_windows)
    ])
    
    mean_rms = rms_values.mean()
    if mean_rms == 0:
        return True  
    
    cv = rms_values.std() / mean_rms
    return cv > cv_threshold

def filter_chunks_by_quality(chunks, starts, rms_threshold_ratio=0.30, cv_threshold=1.2):
    if not chunks:
        return [], [], [], [], [], []
        
    rms_values = np.array([compute_rms(c) for c in chunks])
    peak_rms   = rms_values.max()
    threshold  = peak_rms * rms_threshold_ratio
    
    kept_chunks, kept_starts, kept_rms = [], [], []
    rejected_chunks, rejected_starts, rejected_rms = [], [], []
    
    for chunk, start, rms in zip(chunks, starts, rms_values):
        too_quiet  = rms < threshold
        too_uneven = is_uneven_chunk(chunk, cv_threshold=cv_threshold)
        
        if not too_quiet and not too_uneven:
            kept_chunks.append(chunk)
            kept_starts.append(start)
            kept_rms.append(rms)
        else:
            rejected_chunks.append(chunk)
            rejected_starts.append(start)
            rejected_rms.append(rms)
            
    return (kept_chunks, kept_starts, kept_rms,
            rejected_chunks, rejected_starts, rejected_rms)

# ============================================================
# --- 3. MASTER PROCESS FUNCTION ---
# ============================================================

def process_and_chunk(file_path, sr=16000, chunk_sec=0.5, overlap_sec=0.25, 
                      rms_threshold_ratio=0.30, cv_threshold=1.2):
    """
    Executes the full preprocessing pipeline on a single file.
    Returns only the dictionary of chunks and metadata needed for extraction.
    """
    y_raw, _ = librosa.load(file_path, sr=sr)
    
    y_dc = remove_dc_offset(y_raw)
    y_filtered = apply_lowpass_filter(y_dc, cutoff=4000, sr=sr)
    y_trimmed, _ = apply_trim(y_filtered, top_db=26)
    
    # Handle files that were mostly silence and got trimmed to nothing
    if len(y_trimmed) < int(chunk_sec * sr):
        return {"kept_chunks": []} # Return empty if file is too short after trim
        
    y_normalized = apply_peak_normalization(y_trimmed)
    
    chunks, starts = extract_sliding_windows(
        y_normalized, sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec
    )
    
    (kept_chunks, _, _, _, _, _) = filter_chunks_by_quality(
        chunks, starts, 
        rms_threshold_ratio=rms_threshold_ratio, 
        cv_threshold=cv_threshold
    )
    
    return {"kept_chunks": kept_chunks}