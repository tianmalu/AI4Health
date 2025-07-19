import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 添加joblib用于模型保存
import glob   # 添加glob用于文件搜索

from sklearn.utils import resample


def split_dataset_by_column(features_df, split_column='split'):
    train_data = features_df[features_df[split_column] == 'train_files']
    devel_data = features_df[features_df[split_column] == 'devel_files'] 
    test_data = features_df[features_df[split_column] == 'test_files']  # 修正：应该是test_files
    
    print(f"📊 Train samples: {len(train_data)}")
    print(f"📊 Devel samples: {len(devel_data)}")
    print(f"📊 Test samples: {len(test_data)}")
    
    return train_data, devel_data, test_data

def prepare_data_by_split(merged_df, feature_cols):
    
    train_data, devel_data, test_data = split_dataset_by_column(merged_df)
    
    X_train = train_data[feature_cols].values
    y_train = train_data['label'].values
    
    X_devel = devel_data[feature_cols].values
    y_devel = devel_data['label'].values
    
    X_test = test_data[feature_cols].values if len(test_data) > 0 else None
    y_test = test_data['label'].values if len(test_data) > 0 else None
    
    print(f"🚀 Training set: {X_train.shape}")
    print(f"🎯 Development set: {X_devel.shape}")
    if X_test is not None:
        print(f"🧪 Test set: {X_test.shape}")
    
    return X_train, y_train, X_devel, y_devel, X_test, y_test

def balance_dataset(X, y, ratio=1.0, random_state=42):
    
    print(f"Original class distribution: {np.bincount(y)}")
    
    class_0_mask = y == 0  
    class_1_mask = y == 1  
    
    X_class_0 = X[class_0_mask]
    y_class_0 = y[class_0_mask]
    X_class_1 = X[class_1_mask]
    y_class_1 = y[class_1_mask]
    
    n_class_0 = len(y_class_0)
    n_class_1 = len(y_class_1)
    

    if ratio == 1.0:
        n_samples_0 = n_class_1
    else:
        n_samples_0 = int(n_class_1 / ratio)
        
    n_samples_0 = min(n_samples_0, n_class_0)  
        
    X_class_0_resampled, y_class_0_resampled = resample(
        X_class_0, y_class_0,
        n_samples=n_samples_0,
        random_state=random_state,
        replace=False
    )
        
    X_balanced = np.vstack([X_class_0_resampled, X_class_1])
    y_balanced = np.hstack([y_class_0_resampled, y_class_1])
    
    indices = np.random.RandomState(random_state).permutation(len(y_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    print(f"Reduction: {len(y)} -> {len(y_balanced)} samples ({len(y_balanced)/len(y)*100:.1f}%)")
    
    return X_balanced, y_balanced

def train_svm_with_predefined_split(X_train, y_train, X_devel, y_devel, balance_ratio=1.0):
    
    X_train_balanced, y_train_balanced = balance_dataset(
            X_train, y_train, 
            ratio=balance_ratio
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_devel_scaled = scaler.transform(X_devel)
    
    param_grid = [
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10]
        },
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        {
            'kernel': ['poly'],
            'C': [0.1, 1, 10],
            'degree': [2, 3],
            'gamma': ['scale']
        }
    ]
    
    print("🔍 Grid search with UAR scoring on balanced data...")
    svm = SVC(random_state=42, class_weight='balanced')  
    
    uar_scorer = make_scorer(recall_score, average='macro')
    
    grid_search = GridSearchCV(
        svm, param_grid, 
        cv=3,  
        scoring=uar_scorer,
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train_balanced)
    
    print(f"🏆 Best parameters: {grid_search.best_params_}")
    print(f"📊 Best cross-validation UAR: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    y_devel_pred = best_model.predict(X_devel_scaled)
    
    accuracy = accuracy_score(y_devel, y_devel_pred)
    uar = recall_score(y_devel, y_devel_pred, average='macro')
    
    print(f"\n📈 Development Set Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   UAR: {uar:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_devel, y_devel_pred, target_names=['Healthy', 'Cold']))
    
    cm = confusion_matrix(y_devel, y_devel_pred)
    print(f"\n🎯 Confusion Matrix:")
    print(cm)
    
    return best_model, X_devel_scaled, y_devel, y_devel_pred, scaler

def save_model_and_scaler(model, scaler, model_path="best_svm_model.pkl", scaler_path="svm_scaler.pkl"):
    """
    保存训练好的模型和标准化器
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")

def load_model_and_scaler(model_path="best_svm_model.pkl", scaler_path="svm_scaler.pkl"):
    """
    加载保存的模型和标准化器
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler file not found!")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Scaler loaded from: {scaler_path}")
    return model, scaler

def load_test_features(test_feature_csv):
    """
    加载测试集特征数据
    """
    if not os.path.exists(test_feature_csv):
        print(f"❌ Test feature file not found: {test_feature_csv}")
        return None
    
    test_df = pd.read_csv(test_feature_csv)
    print(f"✅ Loaded test features: {test_df.shape}")
    print(f"📊 Sample columns: {test_df.columns.tolist()[:10]}")
    return test_df

def predict_on_test_data(model, scaler, test_features_df, feature_cols, output_csv="svm_test_predictions.csv"):
    """
    对测试数据进行预测并保存结果为CSV
    """
    # 提取特征
    X_test = test_features_df[feature_cols].values
    print(f"🧪 Test features shape: {X_test.shape}")
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
    # 进行预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.decision_function(X_test_scaled)  # 获取决策函数值
    
    # 获取文件名（假设第一列是文件名）
    file_names = test_features_df.iloc[:, 0].values
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'file_name': file_names,
        'Cold': ['C' if pred == 1 else 'NC' for pred in y_pred],
        'decision_score': y_pred_proba,
        'confidence': np.abs(y_pred_proba)  # 置信度（决策函数绝对值）
    })
    
    # 确保文件名格式正确（添加 .wav 扩展名）
    results_df['file_name'] = results_df['file_name'].apply(
        lambda x: f"{x}.wav" if not str(x).endswith('.wav') else x
    )
    
    # 保存预测结果
    final_results = results_df[['file_name', 'Cold']]  # 只保留必要的列
    final_results.to_csv(output_csv, index=False)
    
    # 打印结果摘要
    print(f"\n📊 Test Prediction Results:")
    print(f"   Total samples: {len(results_df)}")
    print(f"   Predicted Healthy (NC): {(results_df['Cold'] == 'NC').sum()}")
    print(f"   Predicted Cold (C): {(results_df['Cold'] == 'C').sum()}")
    print(f"   Results saved to: {output_csv}")
    
    print(f"\n📋 Sample Predictions:")
    print(results_df[['file_name', 'Cold', 'confidence']].head(10))
    
    # 可视化预测分布
    visualize_test_predictions(results_df)
    
    return results_df

def visualize_test_predictions(results_df):
    """
    可视化测试预测结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 决策函数分布
    ax1.hist(results_df['decision_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    ax1.set_title('SVM Decision Function Distribution')
    ax1.set_xlabel('Decision Function Value')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测标签分布
    pred_counts = results_df['Cold'].value_counts()
    colors = ['lightblue' if label == 'NC' else 'lightcoral' for label in pred_counts.index]
    ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', colors=colors)
    ax2.set_title('Test Set Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig('svm_test_predictions_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_results(model, X_test, y_test, y_pred):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Cold'],
                yticklabels=['Healthy', 'Cold'], ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    if hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(X_test)
        
        ax2.hist([decision_scores[y_test==0], decision_scores[y_test==1]], 
                bins=30, alpha=0.7, label=['Healthy', 'Cold'])
        ax2.axvline(0, color='red', linestyle='--', label='Decision Boundary')
        ax2.set_title('Decision Function Distribution')
        ax2.set_xlabel('Decision Function Value')
        ax2.set_ylabel('Count')
        ax2.legend()
    
    if model.kernel == 'linear':
        feature_importance = np.abs(model.coef_[0])
        top_features = np.argsort(feature_importance)[-20:]  
        
        ax3.barh(range(len(top_features)), feature_importance[top_features])
        ax3.set_title('Top 20 Feature Importance (Linear SVM)')
        ax3.set_xlabel('Absolute Coefficient Value')
        ax3.set_ylabel('Feature Index')
    else:
        ax3.text(0.5, 0.5, f'Feature importance not available\nfor {model.kernel} kernel', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Feature Importance')
    
    if X_test.shape[1] >= 2:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_test_2d = pca.fit_transform(X_test)
        
        colors = ['lightblue', 'lightcoral']
        for i in range(2):
            mask = y_test == i
            ax4.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
                       c=colors[i], label=f'True Class {i}', alpha=0.6)
        
        error_mask = y_test != y_pred
        ax4.scatter(X_test_2d[error_mask, 0], X_test_2d[error_mask, 1], 
                   c='red', marker='x', s=100, label='Misclassified')
        
        ax4.set_title('Test Data Visualization (PCA)')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('svm_results.png', dpi=300)


if __name__ == "__main__":
    feature_csv = "audio_features.csv"  
    features_df = pd.read_csv(feature_csv)
    print("Feature columns:", features_df.columns.tolist()[:10])  

    label_dict = "../ComParE2017_Cold_4students/lab/ComParE2017_Cold.tsv"
    print("📊 Loading labels...")
    label_df = pd.read_csv(label_dict, sep='\t', header=None, names=['filename', 'label'])
    
    if label_df.iloc[0]['filename'] == 'file_name':
        label_df = label_df.iloc[1:].reset_index(drop=True)
    
    label_df['label'] = label_df['label'].map({'C': 1, 'NC': 0}).astype(int)
    print(f"Loaded {len(label_df)} labels")
    print(label_df.head())

    merged_df = pd.merge(features_df, label_df[['filename', 'label']], 
                        left_on=features_df.columns[0], right_on='filename', how='inner')
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print("Label distribution:", merged_df['label'].value_counts())

    feature_cols = [col for col in merged_df.columns 
                   if col not in ['filename', 'label', 'split', 'duration'] and 'file' not in col]
    print(f"Number of features: {len(feature_cols)}")

    X_train, y_train, X_devel, y_devel, X_test, y_test = prepare_data_by_split(merged_df, feature_cols)

    if X_train is not None and len(X_train) > 0:
        print(f"✅ Data loaded successfully!")
        print(f"   Train Features: {X_train.shape}")
        print(f"   Train Labels: {y_train.shape}")
        
        y_train = y_train.astype(int)
        y_devel = y_devel.astype(int)
        
        print(f"   Train Class distribution: {np.bincount(y_train)}")
        print(f"   Devel Class distribution: {np.bincount(y_devel)}")
        
        model, X_eval, y_eval, y_pred, scaler = train_svm_with_predefined_split(
            X_train, y_train, X_devel, y_devel,
            balance_ratio=0.7
        )
        
        visualize_results(model, X_eval, y_eval, y_pred)
        
        # 保存模型和标准化器
        save_model_and_scaler(model, scaler)
        
        if X_test is not None and len(X_test) > 0:
            print(f"\n🧪 Testing on Test Set:")
            X_test_scaled = scaler.transform(X_test)
            y_test_pred = model.predict(X_test_scaled)
            
            y_test = y_test.astype(int)
            
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_uar = recall_score(y_test, y_test_pred, average='macro')
            

            test_recall_per_class = recall_score(y_test, y_test_pred, average=None)
            healthy_recall = test_recall_per_class[0] 
            cold_recall = test_recall_per_class[1]    
            
            test_cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = test_cm.ravel()
            
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Test UAR: {test_uar:.4f}")
            print(f"\n📊 Per-Class Recall:")
            print(f"   Healthy (Class 0) Recall: {healthy_recall:.4f}")
            print(f"   Cold (Class 1) Recall: {cold_recall:.4f}")
        
            
            print(f"\n🎯 Test Confusion Matrix:")
            print(test_cm)
            
            print(f"\n📊 Test Classification Report:")
            print(classification_report(y_test, y_test_pred, target_names=['Healthy', 'Cold']))
        
        print(f"\n🔄 Loading test data from same feature file...")

    # 从同一个特征文件中提取测试数据
    test_data = features_df[features_df['split'] == 'test_files']
    print(f"📊 Found {len(test_data)} test samples")

    if len(test_data) > 0:
        # 提取测试特征
        X_test_new = test_data[feature_cols].values
        print(f"🧪 Test features shape: {X_test_new.shape}")
        
        # 标准化特征
        X_test_new_scaled = scaler.transform(X_test_new)
        
        # 进行预测
        y_pred_new = model.predict(X_test_new_scaled)
        y_pred_proba_new = model.decision_function(X_test_new_scaled)
        
        # 获取文件名（假设第一列是文件名）
        file_names = test_data.iloc[:, 0].values
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'file_name': file_names,
            'Cold': ['C' if pred == 1 else 'NC' for pred in y_pred_new],
            'decision_score': y_pred_proba_new,
            'confidence': np.abs(y_pred_proba_new)
        })
        
        # 确保文件名格式正确（添加 .wav 扩展名如果需要）
        results_df['file_name'] = results_df['file_name'].apply(
            lambda x: f"{x}.wav" if not str(x).endswith('.wav') else x
        )
        
        # 保存预测结果（只保留必要的列）
        final_results = results_df[['file_name', 'Cold']]
        output_csv = "svm_test_predictions.csv"
        final_results.to_csv(output_csv, index=False)
        
        # 打印结果摘要
        print(f"\n📊 Test Prediction Results:")
        print(f"   Total samples: {len(results_df)}")
        print(f"   Predicted Healthy (NC): {(results_df['Cold'] == 'NC').sum()}")
        print(f"   Predicted Cold (C): {(results_df['Cold'] == 'C').sum()}")
        print(f"   Prediction ratio (C/Total): {(results_df['Cold'] == 'C').sum() / len(results_df):.3f}")
        print(f"   Results saved to: {output_csv}")
        
        print(f"\n📋 Sample Predictions:")
        print(results_df[['file_name', 'Cold', 'confidence']].head(10))
        
        # 可视化预测分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 决策函数分布
        ax1.hist(results_df['decision_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax1.set_title('SVM Decision Function Distribution (Test Set)')
        ax1.set_xlabel('Decision Function Value')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测标签分布
        pred_counts = results_df['Cold'].value_counts()
        colors = ['lightblue' if label == 'NC' else 'lightcoral' for label in pred_counts.index]
        ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Test Set Prediction Distribution')
        
        plt.tight_layout()
        plt.savefig('svm_test_predictions_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 验证输出格式
        print(f"\n📋 Output Format Verification:")
        print(f"Required columns: ['file_name', 'Cold']")
        print(f"Actual columns: {list(final_results.columns)}")
        print(f"Column types: {final_results.dtypes}")
        print(f"Unique labels: {final_results['Cold'].unique()}")
        print(f"Sample rows:")
        print(final_results.head())
        
    else:
        print("❌ No test data found in the feature file!")
        print("💡 Make sure your feature file has rows with split='test_files'")
