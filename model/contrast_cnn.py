import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

import os
import random
import csv
import numpy as np
import glob

import time
from sklearn.metrics import accuracy_score, f1_score

def search_in_labels(filename, label_dict):
    base_name = os.path.splitext(filename)[0]
    
    if "_logmel" in base_name:
        base_name = base_name.replace("_logmel", "")
    if "_flipped" in base_name:
        base_name = base_name.replace("_flipped", "")
    
    parts = base_name.split("_")
    if len(parts) >= 2:
        audio_filename = f"{parts[0]}_{parts[1]}.wav"
    else:
        audio_filename = f"{base_name}.wav"
    
    return label_dict.get(audio_filename, None)

def collect_image_paths(split_name):
        sub_dir = os.path.join(img_dir, split_name)
        print(f"🔍 Looking for images in: {sub_dir}")
        
        if not os.path.exists(sub_dir):
            print(f"❌ Directory does not exist: {sub_dir}")
            return []
        
        png_files = glob.glob(os.path.join(sub_dir, "*.png"))
        print(f"📁 Found {len(png_files)} PNG files in {split_name}")
        
        return png_files

class SpectrogramDataset(Dataset):
        def __init__(self, image_paths, label_dict, transform=None):
            self.image_paths = image_paths
            self.label_dict = label_dict
            
            self.transform = transforms.Compose([
                transforms.Resize((128, 42)),
                transforms.ToTensor()
            ])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            filename = os.path.basename(image_path)
            label = search_in_labels(filename, self.label_dict)
            label_num = 1 if label == "C" else 0
            
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            
            return image, label_num

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class SupervisedContrastiveCNN(nn.Module):

    def __init__(self, input_channels=3, projection_dim=128, feature_dim=512):
        super(SupervisedContrastiveCNN, self).__init__()
        
        self.feature_extractor = self._build_feature_extractor(input_channels)
        
        self.feature_dim = self._get_feature_dim(input_channels)
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
            L2Norm(dim=1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
    def _build_feature_extractor(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            
            nn.Conv2d(128, 256, kernel_size=4, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _get_feature_dim(self, input_channels):
        with torch.no_grad():
            x = torch.randn(1, input_channels, 128, 42)
            features = self.feature_extractor(x)
            return features.shape[1]
    
    def forward(self, x, return_features=False):

        features = self.feature_extractor(x)
        
        logits = self.classifier(features)
        
        if return_features:
            projections = self.projection_head(features)
            return logits.squeeze(), projections, features
        else:
            return logits.squeeze()



class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature = 0.1, minority_weight = 2.0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.minority_weight = minority_weight
        
    def forward(self, projections, labels):
  
        device = projections.device
        batch_size = projections.shape[0]
        
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        mask = mask - torch.eye(batch_size).to(device)
        
        exp_sim = torch.exp(similarity_matrix)
        
        pos_sim = exp_sim * mask
        
        neg_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        all_sim = exp_sim * neg_mask
        
        losses = []
        for i in range(batch_size):
            if mask[i].sum() > 0: 
                pos_sum = pos_sim[i].sum()
                neg_sum = all_sim[i].sum()
                
                if neg_sum > 0:
                    loss_i = -torch.log(pos_sum / neg_sum)
                    
                    if labels[i] == 1:  
                        loss_i = loss_i * self.minority_weight
                    
                    losses.append(loss_i)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0).to(device)

class CombinedLoss(nn.Module):
    def __init__(self, classification_loss, contrastive_loss, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.classification_loss = classification_loss
        self.contrastive_loss = contrastive_loss
        self.alpha = alpha  
        
    def forward(self, logits, projections, labels):
        cls_loss = self.classification_loss(logits, labels.float())
        
        cont_loss = self.contrastive_loss(projections, labels)
        
        total_loss = (1 - self.alpha) * cls_loss + self.alpha * cont_loss
        
        return total_loss, cls_loss, cont_loss

    
class ContrastiveSpectrogramDataset(Dataset):
    def __init__(self, image_paths, label_dict, is_training=False, num_views=2, 
                 undersample_ratio=0.5, balance_classes=True):
        
        self.label_dict = label_dict
        self.is_training = is_training
        self.num_views = num_views
        
        self.image_paths = self._balance_dataset(
            image_paths, undersample_ratio, balance_classes
        )
        
        self.base_transform = transforms.Compose([
            transforms.Resize((128, 42)),
            transforms.ToTensor()
        ])
        
        self.strong_transform = transforms.Compose([
            transforms.Resize((128, 42)),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
        
        self.cold_transform = transforms.Compose([
            transforms.Resize((128, 42)),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08))
        ])
        
        self.healthy_transform = transforms.Compose([
            transforms.Resize((128, 42)),
            transforms.RandomAffine(degrees=3, translate=(0.05, 0.02)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05))
        ])
    
    def _balance_dataset(self, image_paths, undersample_ratio, balance_classes):
        cold_paths = []
        healthy_paths = []
        
        for path in image_paths:
            filename = os.path.basename(path)
            label = search_in_labels(filename, self.label_dict)
            
            if label == "C":
                cold_paths.append(path)
            elif label == "NC":
                healthy_paths.append(path)
        
        print(f"📊 原始数据分布:")
        print(f"   Cold: {len(cold_paths)}")
        print(f"   Healthy: {len(healthy_paths)}")
        print(f"   比例: {len(healthy_paths)/len(cold_paths):.2f}:1")
        
        if undersample_ratio < 1.0:
            target_healthy = int(len(healthy_paths) * undersample_ratio)
            healthy_paths = random.sample(healthy_paths, target_healthy)
            print(f"🔽 下采样后 Healthy: {len(healthy_paths)} (比例: {undersample_ratio:.1%})")
        
        balanced_paths = cold_paths + healthy_paths
        random.shuffle(balanced_paths)
        
        print(f"✅ 最终数据分布:")
        print(f"   Cold: {len(cold_paths)}")
        print(f"   Healthy: {len(healthy_paths)}")
        print(f"   总计: {len(balanced_paths)}")
        print(f"   新比例: {len(healthy_paths)/len(cold_paths):.2f}:1")
        
        return balanced_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        label = search_in_labels(filename, self.label_dict)
        label_num = 1 if label == "C" else 0
        
        image = Image.open(image_path).convert("RGB")
        
        if self.is_training:
            views = []
            
            views.append(self.base_transform(image))
            
            for _ in range(self.num_views - 1):
                if label == "C":  
                    augmented = self.cold_transform(image)
                else:  
                    augmented = self.healthy_transform(image)
                views.append(augmented)
            
            return views, label_num
        else:
            return self.base_transform(image), label_num
    
    def get_class_distribution(self):
        cold_count = 0
        healthy_count = 0
        
        for path in self.image_paths:
            filename = os.path.basename(path)
            label = search_in_labels(filename, self.label_dict)
            if label == "C":
                cold_count += 1
            else:
                healthy_count += 1
                
        return {
            'cold': cold_count,
            'healthy': healthy_count,
            'ratio': healthy_count / cold_count if cold_count > 0 else 0,
            'total': len(self.image_paths)
        }
        
if __name__ == "__main__":

    label_dict = {}
    with open("../ComParE2017_Cold_4students/lab/ComParE2017_Cold.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        for row in tqdm(rows, desc="Loading labels"):
            label_dict[row["file_name"]] = row["Cold (upper respiratory tract infection)"]

    data_split = ["train_files", "devel_files"]
    img_dir = "../spectrogram_images/log_mel"
    

    print("🚀 Collecting image paths...")
    train_image_paths = collect_image_paths("train_files")
    devel_image_paths = collect_image_paths("devel_files")

    contrastive_train_dataset = ContrastiveSpectrogramDataset(
        train_image_paths, 
        label_dict, 
        is_training=True, 
        num_views=2,
        undersample_ratio=0.6,  
        balance_classes=True    
    )

    distribution = contrastive_train_dataset.get_class_distribution()
    print(f"\n📈 对比学习训练集最终统计:")
    print(f"   Cold: {distribution['cold']}")
    print(f"   Healthy: {distribution['healthy']}")
    print(f"   比例: {distribution['ratio']:.2f}:1")
    print(f"   总样本: {distribution['total']}")

    devel_dataset = SpectrogramDataset(devel_image_paths, label_dict)
    devel_loader = DataLoader(devel_dataset, batch_size=128, shuffle=False)

    # Training  Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contrastive_model = SupervisedContrastiveCNN(
        input_channels=3, 
        projection_dim=128, 
        feature_dim=512
    ).to(device)

    focal_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))  
    contrastive_loss = SupervisedContrastiveLoss(temperature=0.1, minority_weight=3.0)
    combined_criterion = CombinedLoss(
        classification_loss=focal_loss,
        contrastive_loss=contrastive_loss,
        alpha=0.4 
    )

    optimizer = torch.optim.Adam(
        contrastive_model.parameters(), 
        lr=1e-5, 
        weight_decay=1e-8
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    print(f"🚀 对比学习模型参数: {sum(p.numel() for p in contrastive_model.parameters()):,}")


    
    # START TRAINING
    best_val_f1 = 0.0
    patience = 15
    early_stop_counter = 0
    num_epochs = 100

    train_losses = []
    val_losses = []

    print("🚀 开始监督对比学习训练...\n")

    for epoch in range(num_epochs):
        contrastive_model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_cont_loss = 0.0
        all_preds, all_labels = [], []
        
        print(f'\n{"="*80}')
        print(f'Epoch [{epoch+1}/{num_epochs}] - Contrastive Learning')
        print(f'{"="*80}\n')

        contrastive_train_loader = DataLoader(
            contrastive_train_dataset, batch_size=32, shuffle=True
        )
        
        progress_bar = tqdm(contrastive_train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_views, batch_labels in progress_bar:
            batch_size = len(batch_labels)
            
            all_views = []
            repeated_labels = []
            
            for i, views in enumerate(zip(*batch_views)):
                for view in views:
                    all_views.append(view)
                    repeated_labels.append(batch_labels[i])
            
            all_views = torch.stack(all_views).to(device)
            repeated_labels = torch.tensor(repeated_labels).to(device)
            
            optimizer.zero_grad()
            logits, projections, features = contrastive_model(
                all_views, return_features=True
            )
            
            loss, cls_loss, cont_loss = combined_criterion(
                logits, projections, repeated_labels
            )
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(repeated_labels.cpu().numpy())
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_cont_loss += cont_loss.item()
            
            progress_bar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}',
                'Cont': f'{cont_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(contrastive_train_loader)
        avg_cls_loss = total_cls_loss / len(contrastive_train_loader)
        avg_cont_loss = total_cont_loss / len(contrastive_train_loader)
        
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        
        contrastive_model.eval()
        val_preds, val_labels = [], []
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(devel_loader, desc="Validating"):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                logits = contrastive_model(batch_X)
                preds = (torch.sigmoid(logits) > 0.5).long()

                val_loss = focal_loss(logits, batch_y.float())
                total_val_loss += val_loss.item()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(devel_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch [{epoch+1}] Summary:")
        print(f"  🎯 Learning Rate: {current_lr:.2e}")
        print(f"  📈 Training   - Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Cont: {avg_cont_loss:.4f})")
        print(f"                 Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  📊 Validation - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            torch.save(contrastive_model.state_dict(), "best_contrastive_model_sampling_0.4.pth")
            print(f"🌟 New best F1: {best_val_f1:.4f}, saving model...")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    print(f"\n🎉 对比学习训练完成！最佳F1: {best_val_f1:.4f}")

    # plot training loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    epochs_completed = len(train_losses)
    plt.plot(range(1, epochs_completed + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs_completed + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('training_loss_plot.png')
