import os
import numpy as np
# 1. IMPORT YOUR TOOLBOX!
from preprocess import resample_audio, butter_lowpass_filter, convert_to_model_input, compute_logmel_spectrogram

# 2. SETUP PATHS
DATASET_ROOT = "H:/.shortcut-targets-by-id/1pFPnn7lbpxWVfOmrDEyvHxXp0_6GFB90/Students"
SAVE_DIR = "H:/Processed_Machine_Data"
os.makedirs(SAVE_DIR, exist_ok=True)

def process_and_save_batches():
    """Crawls the dataset, processes audio, and saves features in batches."""
    machines = ["Machine 1", "Machine 2", "Machine 3"]
    states = {"Normal": 0, "Abnormal": 1} 
    target_sr = 16000
    
    print("🚀 Starting Batch Extraction Pipeline...")
    
    for machine in machines:
        machine_spectrograms = []
        machine_labels = []
        
        print(f"\n⚙️ Processing {machine}...")
        
        for state_name, state_label in states.items():
            folder_path = os.path.join(DATASET_ROOT, machine, "machine_data", state_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Could not find {folder_path}")
                continue
                
            audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            print(f"  -> Found {len(audio_files)} {state_name} files.")
            
            # --- THE BATCH PROCESSING LOOP ---
            for i, filename in enumerate(audio_files):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # Step A: Use Toolbox to get clean audio
                    y, sr = resample_audio(file_path, target_sr=target_sr)
                    y_filtered = butter_lowpass_filter(y, cutoff_freq=4000, sample_rate=sr)
                    y_fixed = convert_to_model_input(y_filtered, target_length=target_sr) # 1 second
                    
                    # Step B: Use Toolbox to get Spectrogram
                    logmel = compute_logmel_spectrogram(y_fixed, sr=sr)
                    
                    # Step C: Add to our batch
                    machine_spectrograms.append(logmel)
                    machine_labels.append(state_label)
                    
                except Exception as e:
                    print(f"  ❌ Error on {filename}: {e}")
                
                if (i + 1) % 50 == 0:
                    print(f"     Processed {i + 1}/{len(audio_files)}...")

        # --- SAVE THE BATCH ---
        # Convert lists to NumPy arrays
        X = np.array(machine_spectrograms)
        y = np.array(machine_labels)
        
        # Save this machine's data to the hard drive!
        save_file = os.path.join(SAVE_DIR, f"{machine.replace(' ', '_')}_features.npz")
        np.savez_compressed(save_file, features=X, labels=y)
        
        print(f"✅ {machine} complete! Saved to {save_file}")
        print(f"📊 Final batch shape - Features: {X.shape}, Labels: {y.shape}")

if __name__ == "__main__":
    process_and_save_batches()