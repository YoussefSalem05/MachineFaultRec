import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ============================================================
# --- 1. CONFIGURATION & HYPERPARAMETERS ---
# ============================================================

# Audio / Spectrogram Parameters
TARGET_SR = 16000
N_MELS = 128         
N_FFT = 1024         
HOP_LENGTH = 512     

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30          
LEARNING_RATE = 0.001

# Maro: Set to None to use ALL 1.2 million files. 
# If Kaggle times out after 12 hours, change this to 100000 to cap the giant "Normal" classes.
MAX_SAMPLES_PER_CLASS = None 

# Maro: Early Stopping Parameters
PATIENCE = 4         
MIN_DELTA = 0.5      # Requires at least a 0.5% jump in F1-Score to count as learning

# Maro: Focal Loss parameter. 
# Gamma = 2.0 is the industry standard. Higher (e.g., 3.0) forces the model to focus 
# even harder on the rare classes if your M2 Abnormal F1 score is still too low.
FOCAL_GAMMA = 2.0    

# Class mapping (0 to 5)
CLASS_MAP = {
    "Machine 1_Normal": 0,
    "Machine 1_Abnormal": 1,
    "Machine 2_Normal": 2,
    "Machine 2_Abnormal": 3,
    "Machine 3_Normal": 4,
    "Machine 3_Abnormal": 5
}

# ============================================================
# --- 2. CUSTOM FOCAL LOSS & DATASET CLASS ---
# ============================================================

class FocalLoss(nn.Module):
    """
    Dynamically scales the loss based on confidence.
    Punishes the model heavily for getting rare Abnormal classes wrong,
    but ignores the millions of easy Normal classes once it learns them.
    """
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class MachineAudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        
        # Maro: Tune SpecAugment if F1 score struggles. 
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if self.transform:
            mel_spec = self.transform(waveform)
            mel_spec_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spec)
        else:
            mel_spec_db = waveform
            
        if self.is_train:
            mel_spec_db = self.freq_mask(mel_spec_db)
            mel_spec_db = self.time_mask(mel_spec_db)
            
        return mel_spec_db, label

# ============================================================
# --- 3. DATA GATHERING ---
# ============================================================

def gather_data(max_per_class):
    file_paths = []
    labels = []
    
    path_mapping = {
        "Machine 1_Normal": "/kaggle/input/datasets/mohamedehab2901/machine-1-fault-audio-data/Machine_1_Normal/Normal",
        "Machine 1_Abnormal": "/kaggle/input/datasets/mohamedehab2901/machine-1-fault-audio-data/Machine_1_Abnormal/Abnormal",
        "Machine 2_Normal": "/kaggle/input/datasets/mohamedehab2901/my-processed-audio/Normal",
        "Machine 2_Abnormal": "/kaggle/input/datasets/mohamedehab2901/my-processed-audio/Abnormal",
        "Machine 3_Normal": "/kaggle/input/datasets/mohamedehab2901/machine-3-fault-audio-data/Processed_Machine_Chunks/Machine 3/Normal",
        "Machine 3_Abnormal": "/kaggle/input/datasets/mohamedehab2901/machine-3-fault-audio-data/Processed_Machine_Chunks/Machine 3/Abnormal"
    }
    
    for class_name, folder_path in path_mapping.items():
        class_idx = CLASS_MAP[class_name]
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Directory not found: {folder_path}")
            continue
            
        class_count = 0
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.wav'):
                    file_paths.append(entry.path)
                    labels.append(class_idx)
                    class_count += 1
                    
                    if max_per_class is not None and class_count >= max_per_class:
                        break 
                        
        print(f"Loaded {class_count} files for {class_name} (Class {class_idx})")
            
    return file_paths, labels

# ============================================================
# --- 4. MODEL ARCHITECTURE ---
# ============================================================

def build_model(num_classes=6):
    model = models.resnet18(pretrained=True)
    
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        model.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# ============================================================
# --- 5. MAIN TRAINING LOOP ---
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    paths, labels = gather_data(MAX_SAMPLES_PER_CLASS)
    if len(paths) == 0:
        print("❌ No data found.")
        return
        
    # PyTorch automatically shuffles all 1.2 million perfectly when using DataLoader(shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    train_dataset = MachineAudioDataset(X_train, y_train, transform=mel_transform, is_train=True)
    val_dataset = MachineAudioDataset(X_val, y_val, transform=mel_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = build_model(num_classes=6).to(device)
    
    # Maro: Swapped to Focal Loss to handle the 1.2 million imbalanced files
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    best_val_f1 = 0.0
    trigger_times = 0
    
    print(f"\n🚀 Starting Training (Max Epochs: {EPOCHS}, Patience: {PATIENCE})...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation Loop (Calculating Macro F1 Score)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions to calculate F1 score later
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        # Maro: Calculate Macro F1 Score (The ultimate metric for imbalanced data)
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        print(f"\n📈 Epoch {epoch+1} Results:")
        print(f"Train Focal Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Focal Loss:   {val_loss/len(val_loader):.4f} | Val Macro F1: {val_f1:.2f}%")

        # Maro: Early Stopping is now strictly monitoring the F1 Score!
        if val_f1 > (best_val_f1 + MIN_DELTA):
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_machine_model.pth")
            trigger_times = 0
            print(f"🌟 Meaningful improvement! Best model saved (Macro F1: {best_val_f1:.2f}%)")
        else:
            trigger_times += 1
            print(f"⚠️ No significant improvement. Early stop counter: {trigger_times}/{PATIENCE}")
            
            if trigger_times >= PATIENCE:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
                print("Exiting training loop to save compute time.")
                break 
                
    print(f"\n✅ Training Complete. Best Validation Macro F1 Score: {best_val_f1:.2f}%")

if __name__ == "__main__":
    main()