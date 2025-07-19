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

def create_multimodal_features_with_addition(embeddings_dict, physical_features_df):    
    """Create multimodal features using addition fusion instead of concatenation"""
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
    
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Creating combined features with addition"):
        file_id = row['file_id']
        embedding_idx = row['embedding_idx']
        
        acoustic = row['acoustic_features']
        physical = row[numeric_columns].values.astype(np.float32)
        
        # 🔧 Use addition fusion instead of concatenation
        # Pad physical features to match acoustic dimension if needed
        acoustic_dim = len(acoustic)
        physical_dim = len(physical)
        
        if physical_dim < acoustic_dim:
            # Pad physical features with zeros
            physical_padded = np.pad(physical, (0, acoustic_dim - physical_dim), mode='constant')
            combined = acoustic + physical_padded
        elif physical_dim > acoustic_dim:
            # Truncate physical features to match acoustic dimension
            physical_truncated = physical[:acoustic_dim]
            combined = acoustic + physical_truncated
        else:
            # Same dimension, direct addition
            combined = acoustic + physical
        
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

class EMAQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(EMAQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize embeddings
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inputs):
        # inputs: (batch_size, embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to all embeddings
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embed**2, dim=0)
                    - 2 * torch.matmul(flat_input, self.embed))

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize using embeddings
        quantized = torch.matmul(encodings, self.embed.t()).view_as(inputs)

        # EMA Update (only during training)
        if self.training:
            # Update cluster sizes and embedding averages
            self.cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings, 0), alpha=1 - self.decay
            )
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data.add_(self.epsilon).div_(n + self.num_embeddings * self.epsilon).mul_(n)
            
            # Update embedding averages
            embed_sum = torch.matmul(flat_input.t(), encodings)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # Update embeddings
            self.embed.data.copy_(self.embed_avg / self.cluster_size.unsqueeze(0))

        # VQ loss (commitment loss only, since embedding updates are handled by EMA)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.squeeze()

    def get_codebook_usage(self):
        """获取codebook使用统计"""
        total_usage = self.cluster_size.sum()
        usage_ratio = (self.cluster_size > 0).float().mean()
        return {
            'total_usage': total_usage.item(),
            'usage_ratio': usage_ratio.item(),
            'cluster_sizes': self.cluster_size.cpu().numpy()
        }
    
class ColdDetectionDataset(Dataset):
    def __init__(self,
                 embedding_dict: dict,
                 label_dict: dict,
                 label_ratio = 2,
                 seed = 42
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=-100, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', 
            weight=self.weight, ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class SupervisedVQVAE(nn.Module):
    """
    VQ-VAE with Supervised Classification
    
    Architecture:
    Input → Encoder → VQ → [Decoder (reconstruction) + Classifier (supervision)]
    
    Loss: L_total = α*L_recon + β*L_VQ + γ*L_classification
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings, num_classes, commitment_cost=0.25):
        super(SupervisedVQVAE, self).__init__()
        
        # 🎯 Encoder: Input → Latent Representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 🎲 EMA Vector Quantizer: Continuous → Discrete Latent Code (More Stable)
        self.vq = EMAQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # 🔄 Decoder: Latent Code → Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # 🎯 Classifier: Latent Code → Cold/Normal Classification
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, x):
        # 🔍 Step 1: Encode input to continuous latent
        encoded = self.encoder(x)
        
        # 🎲 Step 2: Quantize to discrete latent codes
        quantized, vq_loss, encoding_indices = self.vq(encoded)
        
        # 🔄 Step 3: Decode for reconstruction
        decoded = self.decoder(quantized)
        
        # 🎯 Step 4: Classify using quantized latent codes
        class_logits = self.classifier(quantized)
        
        return {
            'decoded': decoded,
            'class_logits': class_logits,
            'vq_loss': vq_loss,
            'encoding_indices': encoding_indices,
            'quantized': quantized,
            'encoded': encoded
        }

def train_supervised_vqvae(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda', save_path='best_model.pth', early_stopping_patience=10):
    """
    训练VQ-VAE with监督分类 - 优化版本
    
    Loss = α*L_recon + β*L_VQ + γ*L_classification
    """
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8)
    
    # 损失函数
    criterion_recon = nn.MSELoss()
    # 🔥 使用Focal Loss替换CrossEntropyLoss，更好处理类别不平衡
    criterion_class = FocalLoss(
        alpha=2.0,  # 增加alpha，更关注难分类样本
        gamma=3.0,  # 增加gamma，更强调困难样本
        weight=torch.tensor([1.0, 5.0]).to(device)  # 大幅增加PC类权重
    )
    
    # 🔧 优化损失权重 - 进一步强调分类！
    alpha = 0.005  # L_recon weight - 重构损失权重 (0.01 → 0.005)
    beta = 0.5     # L_VQ weight - 向量量化损失权重 (1.0 → 0.5)
    gamma = 20.0   # L_classification weight - 分类损失权重 (10.0 → 20.0)
    
    print(f"🎯 Enhanced Loss Configuration:")
    print(f"  α (Reconstruction): {alpha} ⬇️ (Further reduced)")
    print(f"  β (VQ Loss): {beta} ⬇️ (Reduced)")
    print(f"  γ (Classification): {gamma} ⬆️ (Significantly increased)")
    print(f"  🔥 Focal Loss: α=2.0, γ=3.0, PC_weight=6.0")
    print(f"  Total Loss = α*L_recon + β*L_VQ + γ*L_focal")
    print(f"  🎯 Focus: Classification >> VQ > Reconstruction")
    
    # 训练历史记录
    history = {
        'train_losses': [], 'val_losses': [],
        'train_recon_losses': [], 'val_recon_losses': [],
        'train_vq_losses': [], 'val_vq_losses': [],
        'train_class_losses': [], 'val_class_losses': [],
        'train_accs': [], 'val_accs': [],
        'train_uars': [], 'val_uars': []
    }
    
    best_val_uar = 0.0
    early_stopping_counter = 0
    best_model_state = None
    
    # 🔧 数据标准化
    print(f"🔧 Applying input normalization...")
    model.to(device)
    
    # 计算训练集的均值和标准差用于标准化
    all_data = []
    for data, _ in train_loader:
        all_data.append(data)
    all_data = torch.cat(all_data, dim=0)
    data_mean = all_data.mean(dim=0).to(device)
    data_std = all_data.std(dim=0).to(device) + 1e-8  # 避免除零
    
    print(f"📊 Data statistics:")
    print(f"  Mean range: [{data_mean.min():.4f}, {data_mean.max():.4f}]")
    print(f"  Std range: [{data_std.min():.4f}, {data_std.max():.4f}]")
    
    for epoch in range(num_epochs):
        # =================== TRAINING ===================
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_vq_loss = 0.0
        train_class_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_labels = []
        train_all_preds = []
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            if data.size(0) == 0 or labels.size(0) == 0:
                continue
                
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # 🔧 数据标准化
            data = (data - data_mean) / data_std
            
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            if data.size(0) != labels.size(0):
                continue
            
            optimizer.zero_grad()
            
            # 🔄 Forward pass
            outputs = model(data)
            
            # 📊 Calculate losses
            l_recon = criterion_recon(outputs['decoded'], data)
            l_vq = outputs['vq_loss']
            l_class = criterion_class(outputs['class_logits'], labels)
            
            # 🎯 Total loss with optimized weights
            total_loss = alpha * l_recon + beta * l_vq + gamma * l_class
            
            # 🔧 梯度裁剪和检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ NaN/Inf loss detected at batch {batch_idx}, skipping...")
                continue
            
            # 🔙 Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
            optimizer.step()
            
            # 📈 Statistics
            train_total_loss += total_loss.item()
            train_recon_loss += l_recon.item()
            train_vq_loss += l_vq.item()
            train_class_loss += l_class.item()
            
            _, predicted = torch.max(outputs['class_logits'].data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(predicted.cpu().numpy())
        
        if train_total == 0:
            continue
            
        train_uar = calculate_uar(train_all_labels, train_all_preds)
        
        # =================== VALIDATION ===================
        model.eval()
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_vq_loss = 0.0
        val_class_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_preds = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                if data.size(0) == 0 or labels.size(0) == 0:
                    continue
                    
                data, labels = data.to(device), labels.to(device).squeeze()
                
                # 🔧 数据标准化
                data = (data - data_mean) / data_std
                
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                if data.size(0) != labels.size(0):
                    continue
                
                outputs = model(data)
                
                l_recon = criterion_recon(outputs['decoded'], data)
                l_vq = outputs['vq_loss']
                l_class = criterion_class(outputs['class_logits'], labels)
                
                total_loss = alpha * l_recon + beta * l_vq + gamma * l_class
                
                val_total_loss += total_loss.item()
                val_recon_loss += l_recon.item()
                val_vq_loss += l_vq.item()
                val_class_loss += l_class.item()
                
                _, predicted = torch.max(outputs['class_logits'].data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(predicted.cpu().numpy())
        
        if val_total == 0:
            continue
            
        val_uar = calculate_uar(val_all_labels, val_all_preds)
        
        # 📊 Calculate epoch metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        avg_train_total_loss = train_total_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_vq_loss = train_vq_loss / len(train_loader)
        avg_train_class_loss = train_class_loss / len(train_loader)
        
        avg_val_total_loss = val_total_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_vq_loss = val_vq_loss / len(val_loader)
        avg_val_class_loss = val_class_loss / len(val_loader)
        
        # 📝 Store history
        history['train_losses'].append(avg_train_total_loss)
        history['val_losses'].append(avg_val_total_loss)
        history['train_recon_losses'].append(avg_train_recon_loss)
        history['val_recon_losses'].append(avg_val_recon_loss)
        history['train_vq_losses'].append(avg_train_vq_loss)
        history['val_vq_losses'].append(avg_val_vq_loss)
        history['train_class_losses'].append(avg_train_class_loss)
        history['val_class_losses'].append(avg_val_class_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        history['train_uars'].append(train_uar)
        history['val_uars'].append(val_uar)
        
        # 📄 Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  📊 Total Loss    - Train: {avg_train_total_loss:.4f}, Val: {avg_val_total_loss:.4f}")
        print(f"  🔄 Recon Loss    - Train: {avg_train_recon_loss:.4f}, Val: {avg_val_recon_loss:.4f}")
        print(f"  🎲 VQ Loss       - Train: {avg_train_vq_loss:.4f}, Val: {avg_val_vq_loss:.4f}")
        print(f"  🎯 Class Loss    - Train: {avg_train_class_loss:.4f}, Val: {avg_val_class_loss:.4f}")
        print(f"  📈 Accuracy      - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")
        print(f"  🎯 UAR           - Train: {train_uar:.4f}, Val: {val_uar:.4f}")
        
        # 🔍 EMA Codebook usage statistics (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            codebook_stats = model.vq.get_codebook_usage()
            print(f"  📚 Codebook Usage: {codebook_stats['usage_ratio']:.2%} ({int(codebook_stats['usage_ratio'] * model.vq.num_embeddings)}/{model.vq.num_embeddings} vectors used)")
        
        # 📅 Learning rate scheduling
        scheduler.step(val_uar)
        
        # 💾 Early stopping and save best model
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_uar': best_val_uar,
                'train_uar': train_uar,
                'val_uar': val_uar,
                'history': history,
                'data_stats': {'mean': data_mean.cpu(), 'std': data_std.cpu()},  # 保存标准化参数
                'model_config': {
                    'input_dim': model.encoder[0].in_features,
                    'hidden_dim': model.encoder[1].num_features // 2,
                    'embedding_dim': model.vq.embedding_dim,
                    'num_embeddings': model.vq.num_embeddings,
                    'num_classes': model.classifier[-1].out_features,
                    'commitment_cost': model.vq.commitment_cost
                },
                'loss_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma}
            }, save_path)
            
            print(f"  🎯 New best validation UAR: {best_val_uar:.4f}")
            print(f"  💾 Model saved to: {save_path}")
        else:
            early_stopping_counter += 1
            print(f"  ⚠️ No improvement for {early_stopping_counter} epochs")
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 80)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with UAR: {best_val_uar:.4f}")
    
    history['best_val_uar'] = best_val_uar
    history['best_model_path'] = save_path
    return history

def load_best_model(model_path, device='cuda'):
    """Load the best saved model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['model_config']
    
    model = SupervisedVQVAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_embeddings=config['num_embeddings'],
        num_classes=config['num_classes'],
        commitment_cost=config['commitment_cost']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ Loaded model from: {model_path}")
    print(f"📊 Best validation UAR: {checkpoint['best_val_uar']:.4f}")
    print(f"🔢 Epoch: {checkpoint['epoch']}")
    
    return model, checkpoint

def calculate_uar(y_true, y_pred):
    """Calculate Unweighted Average Recall (UAR)"""
    from sklearn.metrics import recall_score
    recalls = recall_score(y_true, y_pred, average=None)
    return np.mean(recalls)

def evaluate_model(model, test_loader, device='cuda', data_stats=None):
    """Evaluate model with detailed metrics and proper normalization"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # 🔧 Apply the same normalization used during training
            if data_stats is not None:
                data_mean = data_stats['mean'].to(device)
                data_std = data_stats['std'].to(device)
                data = (data - data_mean) / data_std
            
            outputs = model(data)
            _, predicted = torch.max(outputs['class_logits'].data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    uar = calculate_uar(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 🔧 Fix classification_report error when only one class is predicted
    unique_pred_classes = np.unique(all_preds)
    unique_true_classes = np.unique(all_labels)
    
    try:
        if len(unique_pred_classes) == 1 and len(unique_true_classes) > 1:
            # Only one class predicted but multiple classes exist in true labels
            print(f"⚠️  Warning: Model only predicts class {unique_pred_classes[0]} ({'HC' if unique_pred_classes[0] == 0 else 'PC'})")
            report = f"Model only predicts class {unique_pred_classes[0]} - classification_report skipped"
        else:
            report = classification_report(all_labels, all_preds, target_names=['HC', 'PC'], zero_division=0)
    except Exception as e:
        print(f"⚠️  Warning: classification_report failed: {e}")
        report = f"Classification report failed: {e}"
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\n📊 Detailed Classification Results:")
    print(f"="*50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall UAR: {uar:.4f}")
    print(f"\n🎯 Per-Class Accuracy:")
    print(f"  HC (Healthy): {per_class_acc[0]:.4f}")
    print(f"  PC (Pathological): {per_class_acc[1]:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"     Predicted")
    print(f"       HC   PC")
    print(f"HC   {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"PC   {cm[1,0]:4d} {cm[1,1]:4d}")
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n📈 Additional Metrics:")
    print(f"  Sensitivity (PC Recall): {sensitivity:.4f}")
    print(f"  Specificity (HC Recall): {specificity:.4f}")
    print(f"  Precision (PC): {precision:.4f}")
    print(f"  F1-Score (PC): {f1:.4f}")
    
    return uar, accuracy, report, cm, per_class_acc

def plot_training_curves(history):
    """Plot comprehensive training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total Loss
    axes[0,0].plot(history['train_losses'], label='Train Total Loss', color='blue')
    axes[0,0].plot(history['val_losses'], label='Val Total Loss', color='red')
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Reconstruction Loss
    axes[0,1].plot(history['train_recon_losses'], label='Train Recon Loss', color='green')
    axes[0,1].plot(history['val_recon_losses'], label='Val Recon Loss', color='orange')
    axes[0,1].set_title('Reconstruction Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # VQ Loss
    axes[0,2].plot(history['train_vq_losses'], label='Train VQ Loss', color='purple')
    axes[0,2].plot(history['val_vq_losses'], label='Val VQ Loss', color='brown')
    axes[0,2].set_title('Vector Quantization Loss')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Loss')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Classification Loss
    axes[1,0].plot(history['train_class_losses'], label='Train Class Loss', color='pink')
    axes[1,0].plot(history['val_class_losses'], label='Val Class Loss', color='gray')
    axes[1,0].set_title('Classification Loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Accuracy
    axes[1,1].plot(history['train_accs'], label='Train Accuracy', color='blue')
    axes[1,1].plot(history['val_accs'], label='Val Accuracy', color='red')
    axes[1,1].set_title('Accuracy')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # UAR
    axes[1,2].plot(history['train_uars'], label='Train UAR', color='green')
    axes[1,2].plot(history['val_uars'], label='Val UAR', color='orange')
    axes[1,2].set_title('UAR (Unweighted Average Recall)')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('UAR')
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('supervised_vqvae_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("🚀 Starting Supervised VQ-VAE Training for Cold Detection")
    print("="*80)
    
    # Load data
    label_dict = load_labels_as_dict()
    csv_path = "audio_features_simplified.csv"
    physical_features_df = load_physical_features_as_df(csv_path, label_dict)
    
    # Load embeddings
    embedding_dir = "./embeddings_lhl/"
    val_embeddings_dict = load_acoustic_embeddings(os.path.join(embedding_dir, "devel_files"))
    train_embeddings_dict = load_acoustic_embeddings(os.path.join(embedding_dir, "train_files"))
    
    # Create multimodal features
    combined_train_features = create_multimodal_features_with_addition(train_embeddings_dict, physical_features_df)
    combined_val_features = create_multimodal_features_with_addition(val_embeddings_dict, physical_features_df)
    
    print(f"📊 Dataset Summary:")
    print(f"Combined train features: {len(combined_train_features)} files")
    print(f"Combined validation features: {len(combined_val_features)} files")  
    
    # Create datasets - 🔧 调整数据平衡比例
    train_dataset = ColdDetectionDataset(combined_train_features, label_dict, label_ratio=8)  # 15 → 8
    val_dataset = ColdDetectionDataset(combined_val_features, label_dict, label_ratio=8)     # 15 → 8
    
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"Train DataLoader size: {len(train_loader)} batches")
    print(f"Validation DataLoader size: {len(val_loader)} batches")
    print("-" * 50)

    # Model configuration
    input_dim = next(iter(train_loader))[0].shape[1]
    hidden_dim = 512
    embedding_dim = 256
    num_embeddings = 64 
    num_classes = 2
    num_epochs = 50
    early_stopping_patience = 15

    print(f"🤖 Supervised VQ-VAE Model Configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of embeddings (codebook size): {num_embeddings}")
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
    print(f"\n🚀 Starting Enhanced Supervised VQ-VAE training...")
    print(f"📝 Loss = α*L_recon + β*L_VQ + γ*L_focal")
    print(f"🔥 Key enhancements: Focal Loss, EMA Quantizer, Enhanced class weighting")
    history = train_supervised_vqvae(
        model, 
        train_loader, 
        val_loader, 
        num_epochs, 
        lr=1e-5,  # 5e-6 → 1e-4 (提高学习率)
        device=device, 
        save_path=save_path,
        early_stopping_patience=early_stopping_patience
    )
    
    # Plot training curves
    print(f"\n📈 Plotting training curves...")
    plot_training_curves(history)
    
    # Load the best model
    print(f"\n🔄 Loading best saved model...")
    best_model, checkpoint = load_best_model(save_path, device=device)
    
    # Evaluate on validation set
    print(f"\n🧪 Evaluating best model on validation set...")
    # 🔧 获取保存的数据标准化参数
    data_stats = checkpoint.get('data_stats', None)
    if data_stats is not None:
        print(f"✅ Using saved normalization parameters")
    else:
        print(f"⚠️ No normalization parameters found in checkpoint")
    
    test_uar, test_accuracy, test_report, cm, per_class_acc = evaluate_model(
        best_model, val_loader, device=device, data_stats=data_stats
    )
    
    # Print final results
    print(f"\n" + "="*80)
    print(f"📊 FINAL VALIDATION RESULTS (Supervised VQ-VAE):")
    print(f"="*80)
    print(f"🎯 Validation UAR: {test_uar:.4f}")
    print(f"📈 Validation Accuracy: {test_accuracy:.4f}")
    print(f"🎯 HC Accuracy: {per_class_acc[0]:.4f}")
    print(f"🎯 PC Accuracy: {per_class_acc[1]:.4f}")
    print(f"📋 Classification Report:")
    print(test_report)
    print(f"💾 Best model loaded from: {save_path}")
    print(f"🏆 Expected UAR: {checkpoint['best_val_uar']:.4f}")
    print(f"🏆 Actual UAR: {test_uar:.4f}")
    
    # 显示损失权重配置
    if 'loss_weights' in checkpoint:
        loss_weights = checkpoint['loss_weights']
        print(f"⚖️ Loss Weights: α={loss_weights['alpha']}, β={loss_weights['beta']}, γ={loss_weights['gamma']}")
    
    # 🔍 Final codebook analysis
    print(f"\n🔍 Final Codebook Analysis:")
    codebook_stats = best_model.vq.get_codebook_usage()
    print(f"  📚 Codebook Usage: {codebook_stats['usage_ratio']:.2%}")
    print(f"  📊 Used Vectors: {int(codebook_stats['usage_ratio'] * best_model.vq.num_embeddings)}/{best_model.vq.num_embeddings}")
    print(f"  📈 Total Usage: {codebook_stats['total_usage']:.0f}")
    
    print(f"="*80)