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
from sklearn.model_selection import KFold
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

def train_supervised_vqvae(model, train_loader, val_loader=None, num_epochs=100, lr=1e-4, device='cuda', save_path='best_model.pth', early_stopping_patience = 10):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7)
    
    # 如果没有验证集，使用基于损失的学习率调度器
    if val_loader is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  # 监控损失最小化
            factor=0.5,  # 学习率衰减因子
            patience=10,  # 等待10个epoch无改善再调整
            min_lr=1e-7  # 最小学习率
        )
    else:
        # 动态学习率调节器
        # 当验证UAR在patience个epoch内没有改善时，学习率乘以factor
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # 监控UAR最大化
            factor=0.5,  # 学习率衰减因子
            patience=8,  # 等待8个epoch无改善再调整
            min_lr=1e-7  # 最小学习率
        )
    
    # 可选：预热学习率调度器
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # 从10%的学习率开始
        total_iters=5      # 在前5个epoch进行预热
    )
    
    criterion_recon = nn.MSELoss()
    
    # 计算类别权重 - 降低权重以避免训练不稳定
    # 首先统计数据集中的类别分布
    total_samples = 0
    class_counts = {0: 0, 1: 0}  # HC: 0, PC: 1
    
    for data, labels in train_loader:
        labels = labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        for label in labels:
            class_counts[label.item()] += 1
            total_samples += 1
    
    # 计算平衡的类别权重
    if class_counts[0] > 0 and class_counts[1] > 0:
        weight_hc = total_samples / (2 * class_counts[0])  # HC权重
        weight_pc = total_samples / (2 * class_counts[1])  # PC权重
        # 限制权重范围，避免过大的权重导致训练不稳定
        max_weight = 5.0
        weight_hc = min(weight_hc, max_weight)
        weight_pc = min(weight_pc, max_weight)
        class_weights = torch.tensor([weight_hc, weight_pc]).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(device)
    
    print(f"📊 Training Data Distribution:")
    print(f"  HC (Healthy): {class_counts[0]} samples")
    print(f"  PC (Pathological): {class_counts[1]} samples")
    print(f"  Class weights: HC={class_weights[0]:.4f}, PC={class_weights[1]:.4f}")
    
    # 使用Focal Loss和加权交叉熵的组合
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    
    def combined_class_loss(logits, labels, alpha=0.25, gamma=2.0):
        """结合Focal Loss和加权交叉熵"""
        ce_loss = criterion_class(logits, labels)
        focal_loss_val = focal_loss(logits, labels, alpha=alpha, gamma=gamma)
        return 0.7 * ce_loss + 0.3 * focal_loss_val

    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_uars = []
    val_uars = []
    train_perplexities = []
    val_perplexities = []
    
    # 如果没有验证集，使用基于损失的早停
    if val_loader is None:
        best_train_loss = float('inf')
        loss_history = []
        loss_stability_threshold = 1e-4  # 损失稳定阈值
        stability_epochs = 15  # 连续稳定的epoch数
        stable_counter = 0
        print(f"🎯 Training without validation set")
        print(f"  Loss stability threshold: {loss_stability_threshold}")
        print(f"  Stability epochs required: {stability_epochs}")
    else:
        best_val_uar = 0.0
        early_stopping_counter = 0
        early_stopping_patience = early_stopping_patience
        print(f"🎯 Training with validation set")
        print(f"  Early stopping patience: {early_stopping_patience}")
    
    best_model_state = None
    
    model.to(device)
    
    print(f"🎯 Learning Rate Scheduling:")
    print(f"  Initial LR: {lr}")
    print(f"  Warmup epochs: 5")
    print(f"  ReduceLROnPlateau: factor=0.5, patience={'10' if val_loader is None else '8'}, min_lr=1e-7")
    print("-" * 50)
    
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
            class_loss = combined_class_loss(class_logits, labels)
            
            # 平衡各个损失项的权重
            recon_weight = 0.5  # 降低重建损失权重
            class_weight = 2.0   # 提高分类损失权重
            vq_weight = 0.25
            
            total_loss = (vq_weight * vq_loss + 
                         class_weight * class_loss ) 
            
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
        
        # Validation (只在有验证集时执行)
        if val_loader is not None:
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
                    class_loss = combined_class_loss(class_logits, labels)
                    
                    # 使用相同的权重计算验证损失
                    total_loss = (vq_weight * vq_loss + 
                                 class_weight * class_loss)
                    
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
            
            # Calculate validation metrics
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate average perplexities
            avg_val_perplexity = val_perplexity_sum / val_perplexity_count if val_perplexity_count > 0 else 0
            
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)
            val_uars.append(val_uar)
            val_perplexities.append(avg_val_perplexity)
        else:
            # 没有验证集时设置默认值
            val_uar = 0.0
            val_acc = 0.0
            avg_val_loss = 0.0
            avg_val_perplexity = 0.0
            val_all_labels = []
            val_all_preds = []
            
            val_losses.append(0.0)
            val_accs.append(0.0)
            val_uars.append(0.0)
            val_perplexities.append(0.0)
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Calculate average perplexities
        avg_train_perplexity = train_perplexity_sum / train_perplexity_count if train_perplexity_count > 0 else 0
        
        # 动态学习率调节
        current_lr = optimizer.param_groups[0]['lr']
        
        # 预热阶段（前5个epoch）
        if epoch < 5:
            warmup_scheduler.step()
        else:
            # 在预热之后使用ReduceLROnPlateau
            if val_loader is not None:
                scheduler.step(val_uar)
            else:
                scheduler.step(avg_train_loss)
        
        # 检查学习率是否发生变化
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = abs(current_lr - new_lr) > 1e-8
        
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        train_uars.append(train_uar)
        train_perplexities.append(avg_train_perplexity)
        
        # 打印训练信息
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{num_epochs} (LR: {new_lr:.2e}):")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train UAR: {train_uar:.4f}, Train Perplexity: {avg_train_perplexity:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val UAR: {val_uar:.4f}, Val Perplexity: {avg_val_perplexity:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} (LR: {new_lr:.2e}):")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train UAR: {train_uar:.4f}, Train Perplexity: {avg_train_perplexity:.4f}")
        
        if lr_changed:
            print(f"  📈 Learning rate adjusted: {current_lr:.2e} → {new_lr:.2e}")
        
        # 打印每个epoch的类别预测分布
        if epoch % 10 == 0 or epoch < 5:  # 前5个epoch和每10个epoch打印一次
            train_pred_dist = np.bincount(train_all_preds, minlength=2)
            train_true_dist = np.bincount(train_all_labels, minlength=2)
            
            print(f"  📊 Train - True: HC={train_true_dist[0]}, PC={train_true_dist[1]} | Pred: HC={train_pred_dist[0]}, PC={train_pred_dist[1]}")
            if val_loader is not None:
                val_pred_dist = np.bincount(val_all_preds, minlength=2)
                val_true_dist = np.bincount(val_all_labels, minlength=2)
                print(f"  📊 Val - True: HC={val_true_dist[0]}, PC={val_true_dist[1]} | Pred: HC={val_pred_dist[0]}, PC={val_pred_dist[1]}")
        
        # Early stopping check and save best model
        if val_loader is not None:
            # 有验证集时基于验证UAR
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
                print(f"  ⚠️ No improvement for {early_stopping_counter} epochs (LR: {new_lr:.2e})")
            
            # 如果学习率已经达到最小值且长时间没有改善，可以考虑提前停止
            if new_lr <= 1e-7 and early_stopping_counter >= early_stopping_patience // 2:
                print(f"  ⚠️ Learning rate reached minimum ({new_lr:.2e}) with no improvement")
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"  📊 Best validation UAR: {best_val_uar:.4f}")
                print(f"  📈 Final learning rate: {new_lr:.2e}")
                break
        else:
            # 没有验证集时基于训练损失稳定性
            loss_history.append(avg_train_loss)
            
            # 检查损失是否稳定
            if len(loss_history) >= stability_epochs:
                recent_losses = loss_history[-stability_epochs:]
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                
                if loss_std < loss_stability_threshold:
                    stable_counter += 1
                    print(f"  📊 Loss stable for {stable_counter} checks (std: {loss_std:.6f})")
                else:
                    stable_counter = 0
                    print(f"  📊 Loss not stable (std: {loss_std:.6f})")
                
                # 如果损失稳定或达到最小学习率，保存最终模型
                if stable_counter >= 3 or new_lr <= 1e-7:
                    best_model_state = model.state_dict().copy()
                    
                    # Save final model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'final_train_loss': avg_train_loss,
                        'train_uar': train_uar,
                        'model_config': {
                            'input_dim': model.encoder[0].in_features,
                            'hidden_dim': model.encoder[0].out_features,
                            'embedding_dim': model.vq.embedding_dim,
                            'num_embeddings': model.vq.num_embeddings,
                            'num_classes': model.classifier[-1].out_features,
                            'commitment_cost': model.vq.commitment_cost
                        }
                    }, save_path)
                    
                    print(f"  🎯 Loss stabilized or minimum LR reached")
                    print(f"  💾 Final model saved to: {save_path}")
                    print(f"  📊 Final training loss: {avg_train_loss:.4f}")
                    print(f"  📈 Final learning rate: {new_lr:.2e}")
                    break
            else:
                # 总是保存当前最佳模型（基于最低训练损失）
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    best_model_state = model.state_dict().copy()
                    
                    # Save best model to file
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_train_loss': best_train_loss,
                        'train_uar': train_uar,
                        'model_config': {
                            'input_dim': model.encoder[0].in_features,
                            'hidden_dim': model.encoder[0].out_features,
                            'embedding_dim': model.vq.embedding_dim,
                            'num_embeddings': model.vq.num_embeddings,
                            'num_classes': model.classifier[-1].out_features,
                            'commitment_cost': model.vq.commitment_cost
                        }
                    }, save_path)
                    
                    print(f"  🎯 New best training loss: {best_train_loss:.4f}")
                    print(f"  💾 Model saved to: {save_path}")
        
        print("-" * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if val_loader is not None:
            print(f"✅ Loaded best model with UAR: {best_val_uar:.4f}")
        else:
            print(f"✅ Loaded best model with train loss: {best_train_loss:.4f}")
    
    # 添加学习率历史记录
    lr_history = []
    for epoch in range(len(train_losses)):
        if epoch < 5:
            # 预热阶段的学习率
            warmup_lr = lr * (0.1 + 0.9 * epoch / 4)
            lr_history.append(warmup_lr)
        else:
            # 实际的学习率（这里简化处理，实际应该记录每个epoch的真实学习率）
            lr_history.append(optimizer.param_groups[0]['lr'])
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_uars': train_uars,
        'val_uars': val_uars,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities,
        'lr_history': lr_history,  # 添加学习率历史
        'best_val_uar': best_val_uar if val_loader is not None else 0.0,
        'best_train_loss': best_train_loss if val_loader is None else 0.0,
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
    
    # Learning Rate curve
    if 'lr_history' in history and len(history['lr_history']) > 0:
        ax4.plot(history['lr_history'], label='Learning Rate', color='green')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')  # 使用对数坐标显示学习率
        ax4.legend()
        ax4.grid(True)
    else:
        # Fallback to perplexity if no LR history
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
    
    # Print final statistics
    print(f"\n📊 Training Statistics:")
    print(f"="*50)
    print(f"Total Epochs: {len(history['train_losses'])}")
    print(f"Best Validation UAR: {history['best_val_uar']:.4f}")
    if 'lr_history' in history and len(history['lr_history']) > 0:
        print(f"Initial Learning Rate: {history['lr_history'][0]:.2e}")
        print(f"Final Learning Rate: {history['lr_history'][-1]:.2e}")
        print(f"LR Reduction Factor: {history['lr_history'][-1] / history['lr_history'][0]:.2e}")
    
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

def calculate_codebook_purity(model, data_loader, device='cuda'):
    """
    Calculate codebook purity for binary classification
    Purity measures how well each codebook entry corresponds to a specific class
    """
    model.eval()
    codebook_class_counts = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # Get encoding indices
            _, _, _, encoding_indices, _ = model(data)
            
            # Count class occurrences for each codebook entry
            for idx, label in zip(encoding_indices.cpu().numpy(), labels.cpu().numpy()):
                if idx not in codebook_class_counts:
                    codebook_class_counts[idx] = {0: 0, 1: 0}
                codebook_class_counts[idx][label] += 1
    
    # Calculate purity for each codebook entry
    codebook_purities = {}
    total_samples = 0
    weighted_purity = 0.0
    
    for idx, class_counts in codebook_class_counts.items():
        total_for_idx = sum(class_counts.values())
        if total_for_idx > 0:
            # Purity = max(class_count) / total_count for this codebook entry
            purity = max(class_counts.values()) / total_for_idx
            codebook_purities[idx] = {
                'purity': purity,
                'total_samples': total_for_idx,
                'class_distribution': class_counts,
                'dominant_class': max(class_counts, key=class_counts.get)
            }
            
            # Weighted average purity
            weighted_purity += purity * total_for_idx
            total_samples += total_for_idx
    
    # Overall purity (weighted by usage)
    overall_purity = weighted_purity / total_samples if total_samples > 0 else 0.0
    
    return codebook_purities, overall_purity

def print_codebook_purity_analysis(codebook_purities, overall_purity):
    """
    Print detailed codebook purity analysis
    """
    print(f"\n📊 Codebook Purity Analysis:")
    print(f"="*60)
    print(f"Overall Codebook Purity: {overall_purity:.4f}")
    print(f"Number of Used Codebook Entries: {len(codebook_purities)}")
    print(f"="*60)
    
    print(f"\n📋 Per-Entry Purity Details:")
    print(f"{'Index':<6} {'Purity':<8} {'Samples':<8} {'Dom.Class':<10} {'HC Count':<8} {'PC Count':<8}")
    print("-" * 60)
    
    for idx in sorted(codebook_purities.keys()):
        entry = codebook_purities[idx]
        purity = entry['purity']
        samples = entry['total_samples']
        dom_class = 'HC' if entry['dominant_class'] == 0 else 'PC'
        hc_count = entry['class_distribution'][0]
        pc_count = entry['class_distribution'][1]
        
        print(f"{idx:<6} {purity:<8.4f} {samples:<8} {dom_class:<10} {hc_count:<8} {pc_count:<8}")
    
    # Summary statistics
    purities = [entry['purity'] for entry in codebook_purities.values()]
    if purities:
        print(f"\n📈 Purity Statistics:")
        print(f"  Average Purity: {np.mean(purities):.4f}")
        print(f"  Max Purity: {max(purities):.4f}")
        print(f"  Min Purity: {min(purities):.4f}")
        print(f"  Std Purity: {np.std(purities):.4f}")
    
    print(f"="*60)

def create_kfold_datasets(combined_features, label_dict, n_splits=10, random_state=42):
    """
    Create K-fold cross-validation datasets
    """
    # Collect all samples
    all_samples = []
    for file_id, features in combined_features.items():
        label_key = f"{file_id}.wav"
        if label_key in label_dict:
            raw_label = label_dict[label_key]
            label = 1 if raw_label == "C" else 0
            all_samples.append((file_id, features, label))
    
    # Convert to arrays for sklearn
    file_ids = [sample[0] for sample in all_samples]
    features = [sample[1] for sample in all_samples]
    labels = [sample[2] for sample in all_samples]
    
    # Create K-fold splitter
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_datasets = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(features)):
        # Create train and validation dictionaries
        train_dict = {}
        val_dict = {}
        
        for idx in train_indices:
            train_dict[file_ids[idx]] = features[idx]
        
        for idx in val_indices:
            val_dict[file_ids[idx]] = features[idx]
        
        fold_datasets.append({
            'fold': fold_idx,
            'train_dict': train_dict,
            'val_dict': val_dict,
            'train_indices': train_indices,
            'val_indices': val_indices
        })
    
    return fold_datasets, all_samples

def train_kfold_vqvae(combined_features, label_dict, model_config, training_config, device='cuda'):
    """
    Train VQ-VAE with K-fold cross-validation
    """
    print(f"\n🔄 Starting K-fold Cross-Validation Training...")
    print(f"="*60)
    
    # Create K-fold datasets (9:1 split means 10 folds)
    fold_datasets, all_samples = create_kfold_datasets(
        combined_features, 
        label_dict, 
        n_splits=5, 
        random_state=42
    )
    
    fold_results = []
    overall_history = {
        'fold_uars': [],
        'fold_accuracies': [],
        'fold_purities': [],
        'fold_best_epochs': []
    }
    
    for fold_data in fold_datasets:
        fold_idx = fold_data['fold']
        train_dict = fold_data['train_dict']
        val_dict = fold_data['val_dict']
        
        print(f"\n🎯 Training Fold {fold_idx + 1}/10")
        print(f"Train samples: {len(train_dict)}")
        print(f"Validation samples: {len(val_dict)}")
        print("-" * 40)
        
        # Create datasets for this fold with more balanced sampling
        train_dataset = ColdDetectionDataset(train_dict, label_dict, label_ratio=1)  # Use all data
        val_dataset = ColdDetectionDataset(val_dict, label_dict, label_ratio=1)
        
        # 检查数据分布
        train_stats = train_dataset.get_statistics()
        val_stats = val_dataset.get_statistics()
        
        # 统计每个fold的类别分布
        train_hc_count = sum(1 for _, _, label in train_dataset.all_samples if label.item() == 0)
        train_pc_count = sum(1 for _, _, label in train_dataset.all_samples if label.item() == 1)
        val_hc_count = sum(1 for _, _, label in val_dataset.all_samples if label.item() == 0)
        val_pc_count = sum(1 for _, _, label in val_dataset.all_samples if label.item() == 1)
        
        print(f"  📊 Train distribution: HC={train_hc_count}, PC={train_pc_count}")
        print(f"  📊 Val distribution: HC={val_hc_count}, PC={val_pc_count}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        # Create model for this fold
        model = SupervisedVQVAE(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            embedding_dim=model_config['embedding_dim'],
            num_embeddings=model_config['num_embeddings'],
            num_classes=model_config['num_classes']
        )
        
        # Train model
        save_path = f'./saved_models/best_vqvae_fold_{fold_idx}.pth'
        history = train_supervised_vqvae(
            model,
            train_loader,
            val_loader,
            num_epochs=training_config['num_epochs'],
            lr=training_config['lr'],
            device=device,
            save_path=save_path,
            early_stopping_patience=training_config['early_stopping_patience']
        )
        
        # Load best model for this fold
        best_model, checkpoint = load_best_model(save_path, device=device)
        
        # Evaluate on validation set
        val_uar, val_accuracy, val_report, cm, per_class_acc = evaluate_model(
            best_model, val_loader, device=device
        )
        
        # Calculate codebook purity
        codebook_purities, overall_purity = calculate_codebook_purity(
            best_model, val_loader, device=device
        )
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'val_uar': val_uar,
            'val_accuracy': val_accuracy,
            'val_report': val_report,
            'confusion_matrix': cm,
            'per_class_acc': per_class_acc,
            'codebook_purity': overall_purity,
            'codebook_purities': codebook_purities,
            'best_epoch': checkpoint['epoch'],
            'history': history
        })
        
        # Update overall history
        overall_history['fold_uars'].append(val_uar)
        overall_history['fold_accuracies'].append(val_accuracy)
        overall_history['fold_purities'].append(overall_purity)
        overall_history['fold_best_epochs'].append(checkpoint['epoch'])
        
        print(f"✅ Fold {fold_idx + 1} completed:")
        print(f"  Validation UAR: {val_uar:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Codebook Purity: {overall_purity:.4f}")
        print(f"  Best Epoch: {checkpoint['epoch']}")
        
        # Print purity analysis for this fold
        print_codebook_purity_analysis(codebook_purities, overall_purity)
    
    return fold_results, overall_history

def print_kfold_summary(fold_results, overall_history):
    """
    Print K-fold cross-validation summary
    """
    print(f"\n🏆 K-Fold Cross-Validation Summary:")
    print(f"="*70)
    
    uars = overall_history['fold_uars']
    accuracies = overall_history['fold_accuracies']
    purities = overall_history['fold_purities']
    epochs = overall_history['fold_best_epochs']
    
    print(f"📊 Performance Metrics Across 10 Folds:")
    print(f"{'Fold':<6} {'UAR':<8} {'Accuracy':<10} {'Purity':<8} {'Best Epoch':<10}")
    print("-" * 50)
    
    for i, result in enumerate(fold_results):
        fold_idx = result['fold']
        uar = result['val_uar']
        acc = result['val_accuracy']
        purity = result['codebook_purity']
        epoch = result['best_epoch']
        
        print(f"{fold_idx + 1:<6} {uar:<8.4f} {acc:<10.4f} {purity:<8.4f} {epoch:<10}")
    
    print("-" * 50)
    print(f"📈 Summary Statistics:")
    print(f"  UAR - Mean: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    print(f"  UAR - Max: {np.max(uars):.4f}, Min: {np.min(uars):.4f}")
    print(f"  Accuracy - Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Accuracy - Max: {np.max(accuracies):.4f}, Min: {np.min(accuracies):.4f}")
    print(f"  Purity - Mean: {np.mean(purities):.4f} ± {np.std(purities):.4f}")
    print(f"  Purity - Max: {np.max(purities):.4f}, Min: {np.min(purities):.4f}")
    print(f"  Epochs - Mean: {np.mean(epochs):.1f} ± {np.std(epochs):.1f}")
    
    print(f"="*70)

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
    
    # Merge train and validation features for K-fold cross-validation
    print(f"📊 Original Dataset Summary:")
    print(f"Train features: {len(combined_train_features)} files")
    print(f"Validation features: {len(combined_val_features)} files")
    
    # Combine all features for K-fold
    all_combined_features = {}
    all_combined_features.update(combined_train_features)
    all_combined_features.update(combined_val_features)
    
    print(f"📊 Combined Dataset Summary:")
    print(f"Total combined features: {len(all_combined_features)} files")
    print("-" * 50)
    
    # Create a temporary dataset to get input dimension
    temp_dataset = ColdDetectionDataset(all_combined_features, label_dict, label_ratio=1)
    temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False, num_workers=0)
    input_dim = next(iter(temp_loader))[0].shape[1]
    
    # Model configuration
    model_config = {
        'input_dim': input_dim,
        'hidden_dim': 1024,
        'embedding_dim': 256,
        'num_embeddings': 2,
        'num_classes': 2
    }
    
    # Training configuration - 调整参数以更好地处理类别不平衡
    training_config = {
        'num_epochs': 50,      # 增加训练轮数
        'lr': 1e-4,             # 提高学习率，动态调节会自动降低
        'early_stopping_patience': 50  # 增加早停耐心，配合学习率调节
    }
    
    print(f"🤖 Model Configuration:")
    print(f"Input dimension: {model_config['input_dim']}")
    print(f"Hidden dimension: {model_config['hidden_dim']}")
    print(f"Embedding dimension: {model_config['embedding_dim']}")
    print(f"Number of embeddings: {model_config['num_embeddings']}")
    print(f"Number of classes: {model_config['num_classes']}")
    print(f"Number of epochs: {training_config['num_epochs']}")
    print(f"Early stopping patience: {training_config['early_stopping_patience']}")
    print(f"Learning rate: {training_config['lr']}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Using device: {device}")
    
    # Create save directory
    os.makedirs('./saved_models', exist_ok=True)
    
    # Run K-fold cross-validation
    fold_results, overall_history = train_kfold_vqvae(
        all_combined_features,
        label_dict,
        model_config,
        training_config,
        device=device
    )
    
    # Print K-fold summary
    print_kfold_summary(fold_results, overall_history)
    
    # Save results
    results_summary = {
        'model_config': model_config,
        'training_config': training_config,
        'fold_results': fold_results,
        'overall_history': overall_history,
        'device': device
    }
    
    import pickle
    with open('./saved_models/kfold_results_summary.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\n💾 Results saved to: ./saved_models/kfold_results_summary.pkl")
    print(f"🎉 K-fold cross-validation completed!")
    print(f"📊 Final Average UAR: {np.mean(overall_history['fold_uars']):.4f} ± {np.std(overall_history['fold_uars']):.4f}")
    print(f"📊 Final Average Purity: {np.mean(overall_history['fold_purities']):.4f} ± {np.std(overall_history['fold_purities']):.4f}")