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

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(EMAVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        
        # EMA parameters (not updated by gradient)
        self.register_buffer('ema_cluster_count', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer('ema_weight_sum', torch.zeros(num_embeddings, embedding_dim))
        
        # Track codebook usage statistics
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('total_steps', torch.tensor(0))
        self.usage_threshold = 0.01  # Threshold for unused entries
        
        # Initialize EMA weights
        self.ema_weight_sum.data.normal_()
        self.ema_cluster_count.data.fill_(1.0)
        
    def forward(self, inputs):
        # inputs: (batch_size, embedding_dim)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Update usage statistics during training
        if self.training:
            self.total_steps += 1
            # Count usage for each codebook entry
            usage_batch = encodings.sum(dim=0)
            self.usage_count += usage_batch
            
            # EMA update
            self.ema_cluster_count.mul_(self.decay).add_(usage_batch, alpha=1 - self.decay)
            
            # Calculate sum of embeddings assigned to each cluster
            embedding_sum = torch.matmul(encodings.t(), flat_input)
            self.ema_weight_sum.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
            
            # Update embedding weights using EMA
            # Laplace smoothing
            n = self.ema_cluster_count.sum()
            smoothed_cluster_count = (
                (self.ema_cluster_count + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # Update embedding weights
            self.embedding.weight.data.copy_(self.ema_weight_sum / smoothed_cluster_count.unsqueeze(1))
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # VQ loss (only commitment loss for EMA)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.squeeze(), perplexity
    
    def get_usage_statistics(self):
        """Get codebook usage statistics"""
        if self.total_steps == 0:
            return {
                'usage_rate': torch.zeros(self.num_embeddings),
                'unused_entries': list(range(self.num_embeddings)),
                'most_used_entries': [],
                'usage_entropy': 0.0,
                'collapse_ratio': 1.0,
                'ema_cluster_count': self.ema_cluster_count.clone(),
                'ema_usage_rate': torch.zeros(self.num_embeddings)
            }
        
        # Calculate usage rate from regular counting
        usage_rate = self.usage_count / self.total_steps
        
        # Calculate EMA-based usage rate  
        ema_usage_rate = self.ema_cluster_count / self.ema_cluster_count.sum()
        
        # Find unused entries (below threshold)
        unused_mask = usage_rate < self.usage_threshold
        unused_entries = torch.where(unused_mask)[0].tolist()
        
        # Find most used entries
        most_used_indices = torch.argsort(usage_rate, descending=True)[:10]
        most_used_entries = [(idx.item(), usage_rate[idx].item()) for idx in most_used_indices]
        
        # Calculate entropy (measure of diversity)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        normalized_usage = usage_rate / (usage_rate.sum() + eps)
        entropy = -torch.sum(normalized_usage * torch.log(normalized_usage + eps))
        
        # Calculate collapse ratio (percentage of unused entries)
        collapse_ratio = len(unused_entries) / self.num_embeddings
        
        return {
            'usage_rate': usage_rate,
            'unused_entries': unused_entries,
            'most_used_entries': most_used_entries,
            'usage_entropy': entropy.item(),
            'collapse_ratio': collapse_ratio,
            'total_steps': self.total_steps.item(),
            'ema_cluster_count': self.ema_cluster_count.clone(),
            'ema_usage_rate': ema_usage_rate
        }
    
    def reset_usage_statistics(self):
        """Reset usage statistics"""
        self.usage_count.zero_()
        self.total_steps.zero_()
        # Reset EMA parameters
        self.ema_cluster_count.fill_(1.0)
        self.ema_weight_sum.data.normal_()
    
    def reinitialize_unused_entries(self, threshold=0.01):
        """Reinitialize unused codebook entries to prevent collapse"""
        usage_rate = self.usage_count / (self.total_steps + 1e-8)
        unused_mask = usage_rate < threshold
        unused_indices = torch.where(unused_mask)[0]
        
        if len(unused_indices) > 0:
            # Reinitialize unused entries with random values
            with torch.no_grad():
                self.embedding.weight[unused_indices] = torch.randn_like(
                    self.embedding.weight[unused_indices]
                ) * 0.1
                # Reset usage count and EMA for reinitialized entries
                self.usage_count[unused_indices] = 0
                self.ema_cluster_count[unused_indices] = 1.0
                self.ema_weight_sum[unused_indices] = torch.randn_like(
                    self.ema_weight_sum[unused_indices]
                ) * 0.1
            
            return len(unused_indices)
        return 0

# Legacy alias for backward compatibility
VectorQuantizer = EMAVectorQuantizer
    
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
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings, num_classes, commitment_cost=0.25, label_embedding_dim=16):
        super(SupervisedVQVAE, self).__init__()
        
        # Store dimensions for conditioning
        self.base_embedding_dim = embedding_dim
        self.label_embedding_dim = label_embedding_dim
        self.condition_mode = True  # Can be toggled for train/eval
        
        # Encoder (outputs base embedding dimension)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.base_embedding_dim)
        )
        
        # Label embedding for conditioning
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        
        # Vector Quantizer (uses conditional embedding dimension with EMA)
        conditional_embedding_dim = self.base_embedding_dim + label_embedding_dim
        self.vq = EMAVectorQuantizer(num_embeddings, conditional_embedding_dim, commitment_cost)
        
        # Decoder (reconstructs from conditional quantized representation)
        self.decoder = nn.Sequential(
            nn.Linear(conditional_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classifier (works on conditional quantized representation)
        self.classifier = nn.Sequential(
            nn.Linear(conditional_embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, labels=None):
        # Encode to base embedding
        encoded = self.encoder(x)  # [B, base_embedding_dim]
        
        # Conditional encoding: concatenate label embedding if in condition mode and labels provided
        if self.condition_mode and labels is not None:
            label_embed = self.label_embedding(labels)  # [B, label_embedding_dim]
            conditional_encoded = torch.cat([encoded, label_embed], dim=1)  # [B, base_embedding_dim + label_embedding_dim]
        else:
            # For inference without labels, pad with zeros to match expected dimension
            batch_size = encoded.size(0)
            zero_padding = torch.zeros(batch_size, self.label_embedding_dim, device=encoded.device)
            conditional_encoded = torch.cat([encoded, zero_padding], dim=1)
        
        # Quantize using conditional representation
        quantized, vq_loss, encoding_indices, perplexity = self.vq(conditional_encoded)
        
        # Decode from quantized conditional representation
        decoded = self.decoder(quantized)
        
        # Classify using quantized conditional representation
        class_logits = self.classifier(quantized)
        
        return decoded, class_logits, vq_loss, encoding_indices, perplexity
    
    def set_condition_mode(self, mode: bool):
        """Enable/disable conditioning for train/eval phases"""
        self.condition_mode = mode

def train_supervised_vqvae(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda', save_path='best_model.pth', early_stopping_patience = 10):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss(weight=torch.tensor([1.0,8.0]).to(device))

    
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
            
            # Forward pass with labels for conditioning
            decoded, class_logits, vq_loss, _, perplexity = model(data, labels)
            
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
        
        # Validation - disable conditioning to simulate real inference
        model.eval()
        model.set_condition_mode(False)  # Disable conditioning for validation
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
                
                # Forward pass WITHOUT labels (inference mode)
                decoded, class_logits, vq_loss, _, perplexity = model(data, labels=None)
                
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
        
        # Re-enable conditioning for next training epoch
        model.set_condition_mode(True)
        
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
        
        # Monitor VQ usage every 5 epochs
        if (epoch + 1) % 5 == 0:
            stats = model.vq.get_usage_statistics()
            print(f"  🎲 VQ Usage: {stats['collapse_ratio']:.2%} unused, entropy: {stats['usage_entropy']:.3f}")
            
            # Check for collapse and reinitialize if needed
            reinitialized = monitor_vq_collapse(model, reinitialize_threshold=0.01)
            if reinitialized > 0:
                print(f"  🔄 Reinitialized {reinitialized} unused codebook entries")
        
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
                    'embedding_dim': model.base_embedding_dim,  # Store base embedding dim
                    'label_embedding_dim': model.label_embedding_dim,  # Store label embedding dim
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
    
    # Analyze VQ usage after training
    print(f"\n🎲 Final VQ Usage Analysis:")
    final_stats = analyze_vq_usage(model, "Final Training Analysis")
    
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
        'best_model_path': save_path,
        'vq_usage_stats': final_stats
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
        commitment_cost=config['commitment_cost'],
        label_embedding_dim=config.get('label_embedding_dim', 16)  # Default to 16 if not found
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
    model.set_condition_mode(False)  # Ensure no conditioning during evaluation
    all_preds = []
    all_labels = []
    all_perplexities = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # Forward pass WITHOUT labels (inference mode)
            _, class_logits, _, _, perplexity = model(data, labels=None)
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['val_accs'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

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
            print(f"📏 Padded physical features from {physical_dim} to {acoustic_dim}")
        elif physical_dim > acoustic_dim:
            # Pad acoustic features to match physical dimension
            acoustic_padded = np.zeros(physical_dim, dtype=np.float32)
            acoustic_padded[:acoustic_dim] = acoustic
            combined = acoustic_padded + physical
            print(f"📏 Padded acoustic features from {acoustic_dim} to {physical_dim}")
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

def analyze_vq_usage(model, title="VQ Codebook Usage Analysis"):
    """Analyze and visualize VQ codebook usage"""
    if not hasattr(model, 'vq'):
        print("❌ Model does not have VQ module")
        return None
    
    # Get usage statistics
    stats = model.vq.get_usage_statistics()
    
    print(f"\n🎲 {title}")
    print(f"="*60)
    print(f"📊 Usage Statistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Usage entropy: {stats['usage_entropy']:.4f}")
    print(f"  Collapse ratio: {stats['collapse_ratio']:.2%}")
    print(f"  Unused entries: {len(stats['unused_entries'])}/{model.vq.num_embeddings}")
    
    # Show EMA-specific information if available
    if 'ema_cluster_count' in stats:
        ema_total = stats['ema_cluster_count'].sum().item()
        print(f"  EMA cluster count sum: {ema_total:.2f}")
        print(f"  EMA decay rate: {model.vq.decay:.3f}")
    
    print(f"\n🔥 Top 10 Most Used Entries:")
    for i, (idx, rate) in enumerate(stats['most_used_entries']):
        ema_rate = stats['ema_usage_rate'][idx].item() if 'ema_usage_rate' in stats else 0
        print(f"  {i+1:2d}. Entry {idx:3d}: {rate:.4f} ({rate*100:.2f}%) | EMA: {ema_rate:.4f}")
    
    if len(stats['unused_entries']) > 0:
        print(f"\n❌ Unused Entries (< {model.vq.usage_threshold:.3f}):")
        unused_str = ", ".join(map(str, stats['unused_entries'][:20]))
        if len(stats['unused_entries']) > 20:
            unused_str += f" ... and {len(stats['unused_entries'])-20} more"
        print(f"  {unused_str}")        # Create visualization
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Usage rate histogram (regular + EMA)
            usage_rate = stats['usage_rate'].cpu().numpy()
            ax1.hist(usage_rate, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='Regular Usage')
            
            # Add EMA usage rate if available
            if 'ema_usage_rate' in stats:
                ema_usage_rate = stats['ema_usage_rate'].cpu().numpy()
                ax1.hist(ema_usage_rate, bins=50, alpha=0.5, color='orange', edgecolor='black', label='EMA Usage')
            
            ax1.set_xlabel('Usage Rate')
            ax1.set_ylabel('Number of Entries')
            ax1.set_title('Distribution of Usage Rates')
            ax1.axvline(model.vq.usage_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({model.vq.usage_threshold})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Usage rate by index (regular + EMA)
            indices = np.arange(len(usage_rate))
            colors = ['red' if rate < model.vq.usage_threshold else 'blue' for rate in usage_rate]
            ax2.bar(indices, usage_rate, color=colors, alpha=0.7, label='Regular Usage')
            
            # Add EMA usage rate if available
            if 'ema_usage_rate' in stats:
                ema_usage_rate = stats['ema_usage_rate'].cpu().numpy()
                ax2.plot(indices, ema_usage_rate, color='orange', linewidth=2, label='EMA Usage')
            
            ax2.set_xlabel('Codebook Entry Index')
            ax2.set_ylabel('Usage Rate')
            ax2.set_title('Usage Rate by Codebook Entry')
            ax2.axhline(model.vq.usage_threshold, color='red', linestyle='--', alpha=0.8)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Top 20 most used entries
            top_20 = stats['most_used_entries'][:20]
            if top_20:
                indices_top = [entry[0] for entry in top_20]
                rates_top = [entry[1] for entry in top_20]
                ax3.bar(range(len(indices_top)), rates_top, color='green', alpha=0.7)
                ax3.set_xlabel('Rank')
                ax3.set_ylabel('Usage Rate')
                ax3.set_title('Top 20 Most Used Entries')
                ax3.set_xticks(range(len(indices_top)))
                ax3.set_xticklabels([f'{idx}' for idx in indices_top], rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # 4. Collapse statistics over time (if available)
            ax4.pie([len(stats['unused_entries']), model.vq.num_embeddings - len(stats['unused_entries'])],
                   labels=['Unused', 'Used'], 
                   colors=['red', 'green'],
                   autopct='%1.1f%%',
                   startangle=90)
            ax4.set_title('Codebook Entry Usage (EMA VQ)')
            
            plt.tight_layout()
            plt.savefig(f'vq_usage_analysis_{title.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n📈 Visualization saved as 'vq_usage_analysis_{title.lower().replace(' ', '_')}.png'")
            
        except ImportError:
            print("📊 Matplotlib not available for visualization")
    
    return stats

def monitor_vq_collapse(model, reinitialize_threshold=0.01):
    """Monitor and prevent VQ collapse by reinitializing unused entries"""
    if not hasattr(model, 'vq'):
        return 0
    
    # Get usage statistics
    stats = model.vq.get_usage_statistics()
    
    # Check if collapse is happening
    if stats['collapse_ratio'] > 0.3:  # More than 30% unused
        print(f"⚠️  VQ Collapse detected! {stats['collapse_ratio']:.2%} entries unused")
        
        # Reinitialize unused entries
        reinitialized = model.vq.reinitialize_unused_entries(reinitialize_threshold)
        if reinitialized > 0:
            print(f"🔄 Reinitialized {reinitialized} unused codebook entries")
            return reinitialized
    
    return 0

def initialize_codebook_with_kmeans(model, train_loader, device='cuda', num_samples=1000):
    """
    Initialize VQ codebook using KMeans clustering on real data
    Args:
        model: The VQ-VAE model
        train_loader: Training data loader
        device: Device to run on
        num_samples: Number of samples to use for clustering
    """
    from sklearn.cluster import KMeans
    
    print(f"🎯 Initializing codebook with KMeans clustering...")
    print(f"  Using {num_samples} samples for clustering")
    
    model.eval()
    model.set_condition_mode(True)  # Enable conditioning to get better features
    
    # Collect features from real data
    all_features = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            if sample_count >= num_samples:
                break
                
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # Additional check after squeeze
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Forward pass through encoder to get base features
            encoded = model.encoder(data)  # [B, base_embedding_dim]
            
            # Add label embeddings for conditioning
            label_embed = model.label_embedding(labels)  # [B, label_embedding_dim]
            conditional_features = torch.cat([encoded, label_embed], dim=1)  # [B, base_embedding_dim + label_embedding_dim]
            
            all_features.append(conditional_features.cpu())
            sample_count += conditional_features.size(0)
            
            if sample_count >= num_samples:
                break
    
    if not all_features:
        print("⚠️ No features collected, skipping KMeans initialization")
        return
    
    # Concatenate all features
    features = torch.cat(all_features, dim=0)
    if features.size(0) > num_samples:
        features = features[:num_samples]
    
    print(f"  Collected {features.size(0)} feature vectors of dimension {features.size(1)}")
    
    # Convert to numpy for sklearn
    features_np = features.numpy()
    
    # Perform KMeans clustering
    num_clusters = model.vq.num_embeddings
    print(f"  Running KMeans with {num_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=num_clusters, 
        random_state=42, 
        n_init=10,
        max_iter=300,
        verbose=0
    ).fit(features_np)
    
    # Get cluster centers
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    # Initialize codebook with cluster centers
    with torch.no_grad():
        model.vq.embedding.weight.data.copy_(cluster_centers.to(device))
        
        # Reset EMA parameters with better initialization
        model.vq.ema_cluster_count.fill_(1.0)
        model.vq.ema_weight_sum.data.copy_(cluster_centers.to(device))
        
        # Reset usage statistics
        model.vq.usage_count.zero_()
        model.vq.total_steps.zero_()
    
    print(f"  ✅ Codebook initialized with KMeans cluster centers")
    print(f"  📊 Cluster centers shape: {cluster_centers.shape}")
    print(f"  📈 KMeans inertia: {kmeans.inertia_:.4f}")
    
    # Analyze initial cluster distribution
    labels_kmeans = kmeans.labels_
    unique_labels, counts = np.unique(labels_kmeans, return_counts=True)
    
    print(f"  📊 Initial cluster distribution:")
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        percentage = count / len(labels_kmeans) * 100
        print(f"    Cluster {label}: {count} samples ({percentage:.1f}%)")
    
    # Check if any clusters are empty (shouldn't happen with proper KMeans)
    if len(unique_labels) < num_clusters:
        print(f"  ⚠️ Warning: Only {len(unique_labels)} out of {num_clusters} clusters have samples")
    
    return kmeans

def analyze_codebook_initialization_quality(model, train_loader, device='cuda', num_samples=500):
    """
    Analyze the quality of codebook initialization by checking how well
    the initial codebook covers the data distribution
    """
    print(f"🔍 Analyzing codebook initialization quality...")
    
    model.eval()
    model.set_condition_mode(True)
    
    # Collect features and their nearest codebook assignments
    all_features = []
    all_assignments = []
    all_distances = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            if sample_count >= num_samples:
                break
                
            data, labels = data.to(device), labels.to(device).squeeze()
            
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Get conditional features
            encoded = model.encoder(data)
            label_embed = model.label_embedding(labels)
            conditional_features = torch.cat([encoded, label_embed], dim=1)
            
            # Get VQ assignments and distances
            _, vq_loss, encoding_indices, perplexity = model.vq(conditional_features)
            
            # Calculate distances to assigned codebook entries
            codebook_vectors = model.vq.embedding.weight[encoding_indices]
            distances = torch.norm(conditional_features - codebook_vectors, dim=1)
            
            all_features.append(conditional_features.cpu())
            all_assignments.append(encoding_indices.cpu())
            all_distances.append(distances.cpu())
            
            sample_count += conditional_features.size(0)
            
            if sample_count >= num_samples:
                break
    
    if not all_features:
        print("⚠️ No features collected for analysis")
        return
    
    # Concatenate all data
    features = torch.cat(all_features, dim=0)
    assignments = torch.cat(all_assignments, dim=0)
    distances = torch.cat(all_distances, dim=0)
    
    # Analyze assignment distribution
    unique_assignments, counts = torch.unique(assignments, return_counts=True)
    
    print(f"  📊 Codebook Assignment Analysis:")
    print(f"    Total samples analyzed: {len(assignments)}")
    print(f"    Codebook entries used: {len(unique_assignments)}/{model.vq.num_embeddings}")
    print(f"    Usage ratio: {len(unique_assignments)/model.vq.num_embeddings:.2%}")
    
    # Calculate assignment statistics
    mean_distance = distances.mean().item()
    std_distance = distances.std().item()
    
    print(f"  📏 Distance Statistics:")
    print(f"    Mean distance to assigned codebook: {mean_distance:.4f}")
    print(f"    Std distance to assigned codebook: {std_distance:.4f}")
    print(f"    Min distance: {distances.min().item():.4f}")
    print(f"    Max distance: {distances.max().item():.4f}")
    
    # Per-codebook analysis
    print(f"  📋 Per-Codebook Usage:")
    for i, (assignment, count) in enumerate(zip(unique_assignments, counts)):
        percentage = count.item() / len(assignments) * 100
        mask = assignments == assignment
        avg_dist = distances[mask].mean().item()
        print(f"    Codebook {assignment.item():2d}: {count.item():4d} samples ({percentage:5.1f}%) | Avg dist: {avg_dist:.4f}")
    
    # Check for unused codebook entries
    unused_entries = set(range(model.vq.num_embeddings)) - set(unique_assignments.tolist())
    if unused_entries:
        print(f"  ❌ Unused codebook entries: {sorted(unused_entries)}")
    else:
        print(f"  ✅ All codebook entries are being used")
    
    return {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'usage_ratio': len(unique_assignments)/model.vq.num_embeddings,
        'unused_entries': list(unused_entries),
        'assignment_counts': counts.tolist()
    }

def compare_random_vs_kmeans_initialization(model, train_loader, device='cuda', num_samples=1000):
    """
    Compare random initialization vs KMeans initialization for codebook
    """
    print(f"🔍 Comparing Random vs KMeans initialization...")
    
    # Save original codebook weights
    original_weights = model.vq.embedding.weight.data.clone()
    
    # Test random initialization
    print(f"  📊 Testing random initialization...")
    model.vq.embedding.weight.data.normal_(0, 1.0)
    random_quality = analyze_codebook_initialization_quality(
        model, train_loader, device=device, num_samples=num_samples//2
    )
    
    # Test KMeans initialization
    print(f"  📊 Testing KMeans initialization...")
    initialize_codebook_with_kmeans(
        model, train_loader, device=device, num_samples=num_samples
    )
    kmeans_quality = analyze_codebook_initialization_quality(
        model, train_loader, device=device, num_samples=num_samples//2
    )
    
    # Compare results
    print(f"\n📊 Initialization Comparison:")
    print(f"  Random Init:")
    print(f"    Mean distance: {random_quality['mean_distance']:.4f}")
    print(f"    Usage ratio: {random_quality['usage_ratio']:.2%}")
    print(f"    Unused entries: {len(random_quality['unused_entries'])}")
    
    print(f"  KMeans Init:")
    print(f"    Mean distance: {kmeans_quality['mean_distance']:.4f}")
    print(f"    Usage ratio: {kmeans_quality['usage_ratio']:.2%}")
    print(f"    Unused entries: {len(kmeans_quality['unused_entries'])}")
    
    # Determine which is better
    improvement = random_quality['mean_distance'] - kmeans_quality['mean_distance']
    usage_improvement = kmeans_quality['usage_ratio'] - random_quality['usage_ratio']
    
    print(f"\n📈 Improvement with KMeans:")
    print(f"  Distance reduction: {improvement:.4f} ({improvement/random_quality['mean_distance']*100:.1f}%)")
    print(f"  Usage improvement: {usage_improvement:.2%}")
    
    if improvement > 0 and usage_improvement > 0:
        print(f"  ✅ KMeans initialization is better!")
    else:
        print(f"  ⚠️ KMeans initialization may not be significantly better")
    
    # Keep the KMeans initialization (it's already applied)
    return {
        'random_quality': random_quality,
        'kmeans_quality': kmeans_quality,
        'improvement': improvement,
        'usage_improvement': usage_improvement
    }
    
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
    combined_train_features = create_multimodal_features_with_concatenate(train_embeddings_dict, physical_features_df)
    combined_val_features = create_multimodal_features_with_concatenate(val_embeddings_dict, physical_features_df)
    # combined_train_features = create_multimodal_features_with_addition(train_embeddings_dict, physical_features_df)
    # combined_val_features = create_multimodal_features_with_addition(val_embeddings_dict, physical_features_df)
    
    print(f"📊 Dataset Summary:")
    print(f"Combined train features: {len(combined_train_features)} files")
    print(f"Combined validation features: {len(combined_val_features)} files")  
    
    # Create datasets
    train_dataset = ColdDetectionDataset(combined_train_features, label_dict, label_ratio=8)
    val_dataset = ColdDetectionDataset(combined_val_features, label_dict, label_ratio=12)
    
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
    embedding_dim = 245 # Base embedding dimension
    label_embedding_dim = 16  # Label embedding dimension
    num_embeddings = 2  # Increased to prevent collapse - will be initialized with KMeans
    num_classes = 2  # Healthy (HC) and Pathological (PC)
    num_epochs = 100
    early_stopping_patience = 25

    print(f"🤖 Model Configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Base embedding dimension: {embedding_dim}")
    print(f"Label embedding dimension: {label_embedding_dim}")
    print(f"Conditional embedding dimension: {embedding_dim + label_embedding_dim}")
    print(f"Number of embeddings: {num_embeddings}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")

    # Create model with conditional support
    model = SupervisedVQVAE(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        embedding_dim=embedding_dim, 
        num_embeddings=num_embeddings, 
        num_classes=num_classes,
        label_embedding_dim=label_embedding_dim
    )
    print(f"\n🏗️ Model Architecture:")
    print(model)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Using device: {device}")
    model.to(device)
    
    # Create save directory and path
    os.makedirs('./saved_models', exist_ok=True)
    save_path = './saved_models/best_conditioned_vqvae.pth'
    
    # Initialize codebook with KMeans clustering and compare with random
    print(f"\n🎯 Initializing codebook with KMeans clustering on real data...")
    comparison_result = compare_random_vs_kmeans_initialization(
        model, 
        train_loader, 
        device=device, 
        num_samples=8000
    )
    
    # Train model
    print(f"\n🚀 Starting conditional VQ-VAE training with EMA quantizer...")
    print(f"🎯 Key features:")
    print(f"  - Labels used for conditioning during training")
    print(f"  - No conditioning during validation (inference mode)")
    print(f"  - EMA-based vector quantization (more stable training)")
    print(f"  - Enhanced usage statistics and collapse prevention")
    print(f"  - Codebook initialized with KMeans clustering on real data")
    print(f"  - Larger codebook ({num_embeddings} entries) for better representation")
    history = train_supervised_vqvae(
        model, 
        train_loader, 
        val_loader, 
        num_epochs, 
        lr=8e-6, 
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

    # Evaluate on validation set using the best model
    print(f"\n🧪 Evaluating best conditioned model on validation set...")
    print(f"Note: Evaluation performed WITHOUT condition labels (real-world scenario)")
    test_uar, test_accuracy, test_report, cm, per_class_acc = evaluate_model(best_model, val_loader, device=device)
    
    # Print final results
    print(f"\n" + "="*70)
    print(f"📊 FINAL VALIDATION RESULTS (Best Conditioned Model):")
    print(f"="*70)
    print(f"🎯 Validation UAR: {test_uar:.4f}")
    print(f"📈 Validation Accuracy: {test_accuracy:.4f}")
    print(f"🎯 HC Accuracy: {per_class_acc[0]:.4f}")
    print(f"🎯 PC Accuracy: {per_class_acc[1]:.4f}")
    print(f"📋 Classification Report:")
    print(test_report)
    print(f"💾 Best model loaded from: {save_path}")
    print(f"🏆 Expected UAR: {checkpoint['best_val_uar']:.4f}")
    print(f"🏆 Actual UAR: {test_uar:.4f}")
    print(f"\n🔬 Model Characteristics:")
    print(f"  - Conditioned training with label embeddings")
    print(f"  - Inference without condition labels")
    print(f"  - Enhanced vector quantization discriminability")
    print(f"="*70)
    
    # Final comprehensive VQ analysis
    print(f"\n🎲 Comprehensive VQ Analysis:")
    final_vq_stats = analyze_vq_usage(best_model, "Best Model Analysis")
    
    # Save VQ statistics
    if final_vq_stats:
        vq_stats_path = save_path.replace('.pth', '_vq_stats.json')
        import json
        
        # Convert tensors to lists for JSON serialization
        json_stats = {
            'usage_rate': final_vq_stats['usage_rate'].cpu().tolist(),
            'unused_entries': final_vq_stats['unused_entries'],
            'most_used_entries': final_vq_stats['most_used_entries'],
            'usage_entropy': final_vq_stats['usage_entropy'],
            'collapse_ratio': final_vq_stats['collapse_ratio'],
            'total_steps': final_vq_stats['total_steps']
        }
        
        # Add EMA-specific information if available
        if 'ema_cluster_count' in final_vq_stats:
            json_stats['ema_cluster_count'] = final_vq_stats['ema_cluster_count'].cpu().tolist()
            json_stats['ema_usage_rate'] = final_vq_stats['ema_usage_rate'].cpu().tolist()
            json_stats['ema_decay'] = best_model.vq.decay
            json_stats['ema_epsilon'] = best_model.vq.epsilon
        
        with open(vq_stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"💾 VQ usage statistics saved to: {vq_stats_path}")
    
    # Analyze codebook initialization quality
    analyze_codebook_initialization_quality(model, train_loader, device=device, num_samples=1000)
    
    print(f"\n🎯 Training Complete! Best UAR: {test_uar:.4f}")
    print(f"="*70)