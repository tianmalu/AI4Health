import csv
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import random
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_labels_as_dict():
    label_dict = {}
    with open("../ComParE2017_Cold_4students/lab/ComParE2017_Cold.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        for row in tqdm(rows, desc="Loading labels"):
            label_dict[row["file_name"]] = row["Cold (upper respiratory tract infection)"]
        return label_dict

def search_in_ground_truth(file_id: str, label_dict: dict) -> str:
    wav_name = file_id + ".wav"
    return label_dict.get(wav_name, None)

def load_physical_features_as_df(csv_path: str, label_dict: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=",", encoding="utf-8")
    df_filtered = df[df['filename'].isin(label_dict.keys())]
    
    print(f"📊 Physical features loaded:")
    print(f"  Total rows in CSV: {len(df)}")
    print(f"  Filtered rows: {len(df_filtered)}")
    print(f"  Features: {list(df_filtered.columns)}")
    
    return df_filtered


def load_single_acoustic_embedding(npy_file):
    basename = os.path.splitext(os.path.basename(npy_file))[0]  
    embedding = np.load(npy_file)
    return basename, embedding

def load_acoustic_embeddings(embedding_dir: str):
    npy_files = glob.glob(os.path.join(embedding_dir, "*.npy"))

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_single_acoustic_embedding, npy_files),
                            total=len(npy_files),
                            desc="Loading acoustic embeddings"))

    embedding_dict = dict(results)
    print(f"✅ Loaded {len(embedding_dict)} embeddings.")
    print(f"🧪 Sample shape: {next(iter(embedding_dict.values())).shape}")
    return embedding_dict

def create_multimodal_features_with_concatenate(embeddings_dict, physical_features_df):    
    acoustic_data = []
    for file_id, embedding in embeddings_dict.items():
        filename = f"{file_id}.wav"
        
        acoustic_data.append({
            'filename': filename,
            'file_id': file_id,
            'embedding_idx': 0,
            'acoustic_features': embedding
        })
    
    acoustic_df = pd.DataFrame(acoustic_data)
    
    numeric_columns = physical_features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns_to_remove = ['split'] 
    for col in columns_to_remove:
        if col in numeric_columns:
            numeric_columns.remove(col)
            print(f"⚠️  Removed column: {col}")
    
    physical_subset = physical_features_df[['filename'] + numeric_columns].copy()
    
    merged_df = acoustic_df.merge(physical_subset, on='filename', how='left')
    
    merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)
    
    combined_features = {}
    
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Creating combined features"):
        file_id = row['file_id']
        embedding_idx = row['embedding_idx']
        
        acoustic = row['acoustic_features']
        physical = row[numeric_columns].values.astype(np.float32)
        combined = np.concatenate([acoustic, physical])
        
        if file_id not in combined_features:
            combined_features[file_id] = []
        combined_features[file_id].append(combined)
    
    for file_id in combined_features:
        if len(combined_features[file_id]) == 1:
            combined_features[file_id] = combined_features[file_id][0]
    
    print(f"📊 Multimodal features created:")
    print(f"  Files: {len(combined_features)}")
    print(f"  Acoustic features: {acoustic.shape[0]}")
    print(f"  Physical features: {len(numeric_columns)}")
    print(f"  Combined dimension: {combined.shape[0]}")
    print(f"  Used physical columns: {numeric_columns[:10]}...")  
    return combined_features

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # inputs: (batch_size, embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(inputs)
        
        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.squeeze(), perplexity
    
class ColdDetectionDataset(Dataset):
    def __init__(self,
                 embedding_dict: dict,
                 label_dict: dict,
                 label_ratio = 2,
                 seed = 42  # Add seed parameter
                ):
        
        # Set seed for reproducibility
        random.seed(seed)
        
        self.data_map = {} 
        self.all_samples = []
        
        for file_id, emb in embedding_dict.items():
            label_key = f"{file_id}.wav"
            if label_key not in label_dict:
                continue
                
            raw_label = label_dict[label_key]
            lab = 1 if raw_label == "C" else 0
            
            sample = (
                file_id,
                torch.tensor(emb, dtype=torch.float32),
                torch.tensor(lab, dtype=torch.long),
            ) 
            self.all_samples.append(sample)
        
        self.create_balanced_cold_dataset(label_ratio)
        
    def create_balanced_cold_dataset(self, label_ratio):
        samples_by_label = {0: [], 1: []}
        for sample in self.all_samples:
            label = sample[2].item()
            samples_by_label[label].append(sample)
                        
        num_healthy_samples = min(len(samples_by_label[1])*label_ratio, len(samples_by_label[0]))     
        self.balanced_samples = []
        
        for sample in samples_by_label[1]:
            self.balanced_samples.append(sample)
        
        # Use fixed seed for sampling
        healthy_samples = random.sample(samples_by_label[0], num_healthy_samples)
        self.balanced_samples.extend(healthy_samples)

        random.shuffle(self.balanced_samples)
        self.epoch_samples = self.balanced_samples.copy()

    def __getitem__(self, idx):
        sample = self.epoch_samples[idx]
        
        if len(sample) == 3:
            file_id, embedding, label = sample
            return embedding, label
        else:
            raise ValueError(f"Unexpected sample format with {len(sample)} elements")
        
    
    def __len__(self):
        return len(self.epoch_samples)
    
    
    def refresh_epoch_samples_balanced(self, label_ratio):
        self.create_balanced_cold_dataset(label_ratio)
    
    def get_statistics(self):
        stats = {
            'total_samples': len(self.all_samples),
            'epoch_samples': len(self.epoch_samples),
        }
        
        return stats
    
def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()

class SupervisedVQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings, num_classes, commitment_cost=0.25):
        super(SupervisedVQVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Quantize
        quantized, vq_loss, encoding_indices, perplexity = self.vq(encoded)
        
        # Decode
        decoded = self.decoder(quantized)
        
        # Classify
        class_logits = self.classifier(quantized)
        
        return decoded, class_logits, vq_loss, encoding_indices, perplexity

def train_supervised_vqvae(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda', save_path='best_model.pth', early_stopping_patience = 10):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0]).to(device))

    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_uars = []
    val_uars = []
    train_perplexities = []
    val_perplexities = []
    
    best_val_uar = 0.0
    early_stopping_counter = 0
    early_stopping_patience = early_stopping_patience
    best_model_state = None
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_labels = []
        train_all_preds = []
        train_perplexity_sum = 0.0
        train_perplexity_count = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Skip empty batches
            if data.size(0) == 0 or labels.size(0) == 0:
                print(f"⚠️ Skipping empty batch at index {batch_idx}")
                continue
                
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # Additional check after squeeze
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Check for dimension mismatch
            if data.size(0) != labels.size(0):
                print(f"⚠️ Dimension mismatch: data={data.size(0)}, labels={labels.size(0)}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            decoded, class_logits, vq_loss, _, perplexity = model(data)
            
            # Calculate losses
            recon_loss = criterion_recon(decoded, data)
            class_loss = criterion_class(class_logits, labels)
            total_loss = vq_loss + class_loss 
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(class_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Store perplexity
            train_perplexity_sum += perplexity.item()
            train_perplexity_count += 1
            
            # Store predictions and labels for UAR calculation
            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(predicted.cpu().numpy())
        
        # Skip epoch if no valid batches
        if train_total == 0:
            print(f"⚠️ No valid training batches in epoch {epoch+1}")
            continue
            
        # Calculate training UAR
        train_uar = calculate_uar(train_all_labels, train_all_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_preds = []
        val_perplexity_sum = 0.0
        val_perplexity_count = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                # Skip empty batches
                if data.size(0) == 0 or labels.size(0) == 0:
                    continue
                    
                data, labels = data.to(device), labels.to(device).squeeze()
                
                # Additional check after squeeze
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Check for dimension mismatch
                if data.size(0) != labels.size(0):
                    continue
                
                decoded, class_logits, vq_loss, _, perplexity = model(data)
                
                recon_loss = criterion_recon(decoded, data)
                class_loss = criterion_class(class_logits, labels)
                total_loss = vq_loss + class_loss
                
                val_loss += total_loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store perplexity
                val_perplexity_sum += perplexity.item()
                val_perplexity_count += 1
                
                # Store predictions and labels for UAR calculation
                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(predicted.cpu().numpy())
        
        # Skip epoch if no valid validation batches
        if val_total == 0:
            print(f"⚠️ No valid validation batches in epoch {epoch+1}")
            continue
            
        # Calculate validation UAR
        val_uar = calculate_uar(val_all_labels, val_all_preds)
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate average perplexities
        avg_train_perplexity = train_perplexity_sum / train_perplexity_count if train_perplexity_count > 0 else 0
        avg_val_perplexity = val_perplexity_sum / val_perplexity_count if val_perplexity_count > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_uars.append(train_uar)
        val_uars.append(val_uar)
        train_perplexities.append(avg_train_perplexity)
        val_perplexities.append(avg_val_perplexity)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train UAR: {train_uar:.4f}, Train Perplexity: {avg_train_perplexity:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val UAR: {val_uar:.4f}, Val Perplexity: {avg_val_perplexity:.4f}")
        
        # Early stopping check and save best model
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model to file
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_uar': best_val_uar,
                'train_uar': train_uar,
                'val_uar': val_uar,
                'model_config': {
                    'input_dim': model.encoder[0].in_features,
                    'hidden_dim': model.encoder[0].out_features,
                    'embedding_dim': model.vq.embedding_dim,
                    'num_embeddings': model.vq.num_embeddings,
                    'num_classes': model.classifier[-1].out_features,
                    'commitment_cost': model.vq.commitment_cost
                }
            }, save_path)
            
            print(f"  🎯 New best validation UAR: {best_val_uar:.4f}")
            print(f"  💾 Model saved to: {save_path}")
        else:
            early_stopping_counter += 1
            print(f"  ⚠️ No improvement for {early_stopping_counter} epochs")
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"  📊 Best validation UAR: {best_val_uar:.4f}")
            break
        
        print("-" * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with UAR: {best_val_uar:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_uars': train_uars,
        'val_uars': val_uars,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities,
        'best_val_uar': best_val_uar,
        'best_model_path': save_path
    }

def load_best_model(model_path, device='cuda'):
    """Load the best saved model"""
    # Fix for PyTorch 2.6 weights_only issue
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration
    config = checkpoint['model_config']
    
    # Create model with saved configuration
    model = SupervisedVQVAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_embeddings=config['num_embeddings'],
        num_classes=config['num_classes'],
        commitment_cost=config['commitment_cost']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ Loaded model from: {model_path}")
    print(f"📊 Best validation UAR: {checkpoint['best_val_uar']:.4f}")
    print(f"🔢 Epoch: {checkpoint['epoch']}")
    
    return model, checkpoint

def calculate_uar(y_true, y_pred):
    """Calculate Unweighted Average Recall (UAR)"""
    from sklearn.metrics import recall_score
    
    # Calculate recall for each class
    recalls = recall_score(y_true, y_pred, average=None)
    
    # Return the mean of all class recalls (UAR)
    return np.mean(recalls)

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_perplexities = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            
            _, class_logits, _, _, perplexity = model(data)
            _, predicted = torch.max(class_logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_perplexities.append(perplexity.item())
    
    # Calculate UAR as primary metric
    uar = calculate_uar(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['HC', 'PC'])
    avg_perplexity = np.mean(all_perplexities)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\n📊 Detailed Classification Results:")
    print(f"="*50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall UAR: {uar:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    print(f"\n🎯 Per-Class Accuracy:")
    print(f"  HC (Healthy): {per_class_acc[0]:.4f}")
    print(f"  PC (Pathological): {per_class_acc[1]:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"     Predicted")
    print(f"       HC   PC")
    print(f"HC   {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"PC   {cm[1,0]:4d} {cm[1,1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity (True Positive Rate) for PC class
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity (True Negative Rate) for HC class  
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Precision for PC class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # F1 score for PC class
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n📈 Additional Metrics:")
    print(f"  Sensitivity (PC Recall): {sensitivity:.4f}")
    print(f"  Specificity (HC Recall): {specificity:.4f}")
    print(f"  Precision (PC): {precision:.4f}")
    print(f"  F1-Score (PC): {f1:.4f}")
    
    return uar, accuracy, report, cm, per_class_acc

def plot_training_curves(history):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(history['train_losses'], label='Train Loss', color='blue')
    ax1.plot(history['val_losses'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_accs'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_accs'], label='Val Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # UAR curves
    ax3.plot(history['train_uars'], label='Train UAR', color='blue')
    ax3.plot(history['val_uars'], label='Val UAR', color='red')
    ax3.set_title('Training and Validation UAR')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('UAR')
    ax3.legend()
    ax3.grid(True)
    
    # Perplexity curves
    ax4.plot(history['train_perplexities'], label='Train Perplexity', color='blue')
    ax4.plot(history['val_perplexities'], label='Val Perplexity', color='red')
    ax4.set_title('Training and Validation Perplexity')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Perplexity')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('supervised_vqvae_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final perplexity statistics
    print(f"\n📊 Perplexity Statistics:")
    print(f"="*50)
    print(f"Final Train Perplexity: {history['train_perplexities'][-1]:.4f}")
    print(f"Final Val Perplexity: {history['val_perplexities'][-1]:.4f}")
    print(f"Max Train Perplexity: {max(history['train_perplexities']):.4f}")
    print(f"Max Val Perplexity: {max(history['val_perplexities']):.4f}")
    print(f"Average Train Perplexity: {np.mean(history['train_perplexities']):.4f}")
    print(f"Average Val Perplexity: {np.mean(history['val_perplexities']):.4f}")
    
    # Codebook utilization analysis
    codebook_size = 128  # From model configuration
    max_possible_perplexity = codebook_size
    final_train_utilization = history['train_perplexities'][-1] / max_possible_perplexity * 100
    final_val_utilization = history['val_perplexities'][-1] / max_possible_perplexity * 100
    
    print(f"\n📋 Codebook Utilization:")
    print(f"Codebook Size: {codebook_size}")
    print(f"Max Possible Perplexity: {max_possible_perplexity}")
    print(f"Final Train Utilization: {final_train_utilization:.2f}%")
    print(f"Final Val Utilization: {final_val_utilization:.2f}%")
    print(f"="*50)

def create_multimodal_features_with_addition(embeddings_dict, physical_features_df):    
    acoustic_data = []
    for file_id, embedding in embeddings_dict.items():
        filename = f"{file_id}.wav"
        
        acoustic_data.append({
            'filename': filename,
            'file_id': file_id,
            'embedding_idx': 0,
            'acoustic_features': embedding
        })
    
    acoustic_df = pd.DataFrame(acoustic_data)
    
    numeric_columns = physical_features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns_to_remove = ['split'] 
    for col in columns_to_remove:
        if col in numeric_columns:
            numeric_columns.remove(col)
            print(f"⚠️  Removed column: {col}")
    
    physical_subset = physical_features_df[['filename'] + numeric_columns].copy()
    
    merged_df = acoustic_df.merge(physical_subset, on='filename', how='left')
    
    merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)
    
    combined_features = {}
    
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Creating multimodal features with addition"):
        file_id = row['file_id']
        embedding_idx = row['embedding_idx']
        
        acoustic = row['acoustic_features']
        physical = row[numeric_columns].values.astype(np.float32)
        
        # Get dimensions
        acoustic_dim = acoustic.shape[0]
        physical_dim = len(physical)
        
        # Handle dimension mismatch
        if acoustic_dim > physical_dim:
            # Pad physical features to match acoustic dimension
            physical_padded = np.zeros(acoustic_dim, dtype=np.float32)
            physical_padded[:physical_dim] = physical
            combined = acoustic + physical_padded
        elif physical_dim > acoustic_dim:
            # Pad acoustic features to match physical dimension
            acoustic_padded = np.zeros(physical_dim, dtype=np.float32)
            acoustic_padded[:acoustic_dim] = acoustic
            combined = acoustic_padded + physical
        else:
            # Dimensions match, direct addition
            combined = acoustic + physical
        
        if file_id not in combined_features:
            combined_features[file_id] = []
        combined_features[file_id].append(combined)
    
    for file_id in combined_features:
        if len(combined_features[file_id]) == 1:
            combined_features[file_id] = combined_features[file_id][0]
    
    print(f"📊 Multimodal features created with addition:")
    print(f"  Files: {len(combined_features)}")
    print(f"  Acoustic features: {acoustic.shape[0]}")
    print(f"  Physical features: {len(numeric_columns)}")
    print(f"  Combined dimension: {combined.shape[0]}")
    print(f"  Fusion method: Element-wise addition")
    print(f"  Used physical columns: {numeric_columns[:10]}...")  
    return combined_features

def visualize_codebook_usage(model, data_loader, device='cuda', save_path='codebook_usage.png'):
    """
    Visualize the usage distribution of codebook entries
    """
    model.eval()
    all_encoding_indices = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _, _, _, encoding_indices, _ = model(data)
            all_encoding_indices.extend(encoding_indices.cpu().numpy())
    
    # Count frequency of each codebook entry
    codebook_size = model.vq.num_embeddings
    usage_counts = np.bincount(all_encoding_indices, minlength=codebook_size)
    usage_percentages = usage_counts / len(all_encoding_indices) * 100
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of usage counts
    ax1.bar(range(codebook_size), usage_counts, alpha=0.7, color='skyblue')
    ax1.set_title('Codebook Entry Usage Counts')
    ax1.set_xlabel('Codebook Index')
    ax1.set_ylabel('Usage Count')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of usage percentages
    ax2.hist(usage_percentages[usage_percentages > 0], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Distribution of Usage Percentages')
    ax2.set_xlabel('Usage Percentage (%)')
    ax2.set_ylabel('Number of Codebook Entries')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    used_entries = np.sum(usage_counts > 0)
    unused_entries = codebook_size - used_entries
    utilization_rate = used_entries / codebook_size * 100
    
    print(f"\n📊 Codebook Usage Analysis:")
    print(f"="*50)
    print(f"Total Codebook Entries: {codebook_size}")
    print(f"Used Entries: {used_entries}")
    print(f"Unused Entries: {unused_entries}")
    print(f"Utilization Rate: {utilization_rate:.2f}%")
    print(f"Most Used Entry: Index {np.argmax(usage_counts)} ({usage_percentages[np.argmax(usage_counts)]:.2f}%)")
    print(f"Average Usage (used entries): {np.mean(usage_percentages[usage_percentages > 0]):.2f}%")
    print(f"="*50)
    
    return usage_counts, usage_percentages

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Load data
    label_dict = load_labels_as_dict()
    csv_path = "audio_features.csv"
    physical_features_df = load_physical_features_as_df(csv_path, label_dict)
    
    # Load embeddings
    embedding_dir = "./embeddings_lhl/"
    val_embeddings_dict = load_acoustic_embeddings(os.path.join(embedding_dir, "devel_files"))
    train_embeddings_dict = load_acoustic_embeddings(os.path.join(embedding_dir, "train_files"))
    
    # Create multimodal features
    combined_train_features = create_multimodal_features_with_addition(train_embeddings_dict, physical_features_df)
    combined_val_features = create_multimodal_features_with_addition(val_embeddings_dict, physical_features_df)
    # combined_train_features = create_multimodal_features_with_addition(train_embeddings_dict, physical_features_df)
    # combined_val_features = create_multimodal_features_with_addition(val_embeddings_dict, physical_features_df)
    
    print(f"📊 Dataset Summary:")
    print(f"Combined train features: {len(combined_train_features)} files")
    print(f"Combined validation features: {len(combined_val_features)} files")  
    
    # Create datasets
    train_dataset = ColdDetectionDataset(combined_train_features, label_dict, label_ratio= 10 )
    val_dataset = ColdDetectionDataset(combined_val_features, label_dict, label_ratio = 15)
    
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"Train DataLoader size: {len(train_loader)} batches")
    print(f"Validation DataLoader size: {len(val_loader)} batches")
    print("-" * 50)

    # Model configuration
    input_dim = next(iter(train_loader))[0].shape[1]
    hidden_dim = 1024
    embedding_dim = 256
    num_embeddings = 2
    num_classes = 2  # Healthy (HC) and Pathological (PC)
    num_epochs = 200
    early_stopping_patience = 30

    print(f"🤖 Model Configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of embeddings: {num_embeddings}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")

    # Create model
    model = SupervisedVQVAE(input_dim, hidden_dim, embedding_dim, num_embeddings, num_classes)
    print(f"\n🏗️ Model Architecture:")
    print(model)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Using device: {device}")
    model.to(device)
    
    # Create save directory and path
    os.makedirs('./saved_models', exist_ok=True)
    save_path = './saved_models/best_supervised_vqvae.pth'
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = train_supervised_vqvae(
        model, 
        train_loader, 
        val_loader, 
        num_epochs, 
        lr=5e-5, 
        device=device, 
        save_path=save_path,
        early_stopping_patience=early_stopping_patience
    )
    
    # Plot training curves
    print(f"\n📈 Plotting training curves...")
    plot_training_curves(history)
    
    # Load the best model explicitly (this ensures we use the best saved model)
    print(f"\n🔄 Loading best saved model...")
    best_model, checkpoint = load_best_model(save_path, device=device)
    val_dataset = ColdDetectionDataset(combined_val_features, label_dict, label_ratio=15)

    # Visualize codebook usage
    print(f"\n📊 Analyzing codebook usage...")
    usage_counts, usage_percentages = visualize_codebook_usage(best_model, val_loader, device=device)

    # Evaluate on validation set using the best model
    print(f"\n🧪 Evaluating best model on validation set...")
    test_uar, test_accuracy, test_report, cm, per_class_acc = evaluate_model(best_model, val_loader, device=device)
    
    # Print final results
    print(f"\n" + "="*60)
    print(f"📊 FINAL VALIDATION RESULTS (Best Model):")
    print(f"="*60)
    print(f"🎯 Validation UAR: {test_uar:.4f}")
    print(f"📈 Validation Accuracy: {test_accuracy:.4f}")
    print(f"🎯 HC Accuracy: {per_class_acc[0]:.4f}")
    print(f"🎯 PC Accuracy: {per_class_acc[1]:.4f}")
    print(f"📋 Classification Report:")
    print(test_report)
    print(f"💾 Best model loaded from: {save_path}")
    print(f"🏆 Expected UAR: {checkpoint['best_val_uar']:.4f}")
    print(f"🏆 Actual UAR: {test_uar:.4f}")
    print(f"="*60)

    # Visualize codebook usage
    print(f"\n📊 Visualizing codebook usage...")
    visualize_codebook_usage(best_model, val_loader, device=device, save_path='codebook_usage_distribution.png')