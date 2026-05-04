"""
Batch Data Extractor
Crawls the dataset, applies the DSP pipeline, and saves discrete chunk .wav files.
Enforces Parent-File naming conventions to prevent train/val leakage.
"""
import os
import soundfile as sf
from tqdm import tqdm
from preprocess import process_and_chunk

# ============================================================
# --- SETUP & PATHS ---
# ============================================================

# This dynamically builds the path to your actual system Downloads folder
user_home = os.path.expanduser("~") 
SAVE_DIR = os.path.join(user_home, "Downloads", "Processed_Machine_Chunks")
DATASET_ROOT = "H:/.shortcut-targets-by-id/1pFPnn7lbpxWVfOmrDEyvHxXp0_6GFB90/Students"
SAVE_DIR = "Downloads/Processed_Machine_Chunks" # Changed name to reflect wav chunks
TARGET_SR = 16000

MACHINES = ["Machine 1", "Machine 2", "Machine 3"]
STATES = ["Normal", "Abnormal"]

def run_extraction_pipeline():
    print("🚀 Starting Industrial Audio Extraction Pipeline...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    total_files_processed = 0
    total_chunks_yielded = 0
    
    for machine in MACHINES:
        print(f"\n⚙️  Processing {machine}...")
        
        for state in STATES:
            folder_path = os.path.join(DATASET_ROOT, machine, "machine_data", state)
            
            # Create matching output directory structure
            output_dir = os.path.join(SAVE_DIR, machine, state)
            os.makedirs(output_dir, exist_ok=True)
            
            if not os.path.exists(folder_path):
                print(f"  ⚠️ Warning: Path not found: {folder_path}")
                continue
                
            audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            print(f"  -> Found {len(audio_files)} {state} files. Extracting chunks...")
            
            # Use tqdm for a professional progress bar
            for filename in tqdm(audio_files, desc=f"{state} Files", unit="file"):
                file_path = os.path.join(folder_path, filename)
                
                # Get the base name without extension (e.g., 'rec_001' from 'rec_001.wav')
                base_name = os.path.splitext(filename)[0]
                
                try:
                    # 1. Run the heavy DSP pipeline
                    result = process_and_chunk(
                        file_path, 
                        sr=TARGET_SR,
                        chunk_sec=0.5,
                        overlap_sec=0.25,
                        rms_threshold_ratio=0.30,
                        cv_threshold=1.2
                    )
                    
                    kept_chunks = result["kept_chunks"]
                    
                    # 2. Save the surviving chunks with PARENT-LINKED naming
                    for idx, chunk in enumerate(kept_chunks):
                        # Naming convention: rec001_chunk_00.wav
                        chunk_filename = f"{base_name}_chunk_{idx:02d}.wav"
                        chunk_save_path = os.path.join(output_dir, chunk_filename)
                        
                        # sf.write is the industry standard for writing audio files
                        sf.write(chunk_save_path, chunk, TARGET_SR)
                        
                        total_chunks_yielded += 1
                        
                except Exception as e:
                    print(f"\n  ❌ Pipeline Error on {filename}: {str(e)}")
                    
                total_files_processed += 1

    print("\n" + "="*50)
    print("✅ BATCH EXTRACTION COMPLETE")
    print(f"📊 Total source files processed: {total_files_processed}")
    print(f"📊 Total valid chunks yielded:   {total_chunks_yielded}")
    print(f"📁 Chunks saved to: {SAVE_DIR}")
    print("="*50)

if __name__ == "__main__":
    run_extraction_pipeline()