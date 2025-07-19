import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# SMOTE和贝叶斯优化库
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✅ SMOTE available (imbalanced-learn)")
except ImportError:
    SMOTE_AVAILABLE = False
    print("❌ SMOTE not available. Install with: pip install imbalanced-learn")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
    print("✅ Bayesian optimization available (scikit-optimize)")
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("❌ Bayesian optimization not available. Install with: pip install scikit-optimize")
    exit()

def split_dataset_by_column(features_df, split_column='split'):
    """根据split列划分数据集"""
    train_data = features_df[features_df[split_column] == 'train_files']
    devel_data = features_df[features_df[split_column] == 'devel_files'] 
    test_data = features_df[features_df[split_column] == 'devel_files']
    
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

def balance_dataset_smote(X, y, sampling_strategy='auto', random_state=42):
    """使用SMOTE进行数据平衡"""
    
    if not SMOTE_AVAILABLE:
        print("❌ SMOTE not available. Please install imbalanced-learn")
        return X, y
    
    print(f"\n🔄 Balancing dataset using SMOTE...")
    print(f"Original class distribution: {np.bincount(y)}")
    
    # 创建SMOTE对象
    smote = SMOTE(
        sampling_strategy=sampling_strategy,  # 'auto'表示平衡到多数类的数量
        random_state=random_state,
        k_neighbors=5  # 可以调整邻居数量
    )
    
    try:
        # 应用SMOTE
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        print(f"Expansion: {len(y)} -> {len(y_balanced)} samples ({len(y_balanced)/len(y)*100:.1f}%)")
        
        return X_balanced, y_balanced
        
    except Exception as e:
        print(f"❌ SMOTE failed: {e}")
        print("🔄 Falling back to original data...")
        return X, y

def bayesian_optimize_svm(X_train, y_train, n_calls=50):
    """使用贝叶斯优化进行SVM超参数优化"""
    
    print(f"🔬 Bayesian optimization with {n_calls} evaluations...")
    
    # 定义搜索空间
    dimensions = [
        Categorical(['linear', 'rbf', 'poly'], name='kernel'),
        Real(0.01, 100.0, prior='log-uniform', name='C'),
        Categorical(['scale', 'auto'], name='gamma_type'),
        Real(0.001, 1.0, prior='log-uniform', name='gamma_value'),
        Categorical([2, 3, 4], name='degree'),
        Categorical(['balanced', None], name='class_weight')
    ]
    
    # 定义目标函数
    @use_named_args(dimensions)
    def objective(**params):
        """目标函数：返回负UAR（因为gp_minimize最小化）"""
        
        # 构建参数字典
        svm_params = {
            'kernel': params['kernel'],
            'C': params['C'],
            'random_state': 42,
            'class_weight': params['class_weight']
        }
        
        # 根据kernel类型添加相应参数
        if params['kernel'] == 'rbf':
            if params['gamma_type'] == 'scale':
                svm_params['gamma'] = 'scale'
            elif params['gamma_type'] == 'auto':
                svm_params['gamma'] = 'auto'
            else:
                svm_params['gamma'] = params['gamma_value']
        elif params['kernel'] == 'poly':
            svm_params['degree'] = params['degree']
            if params['gamma_type'] == 'scale':
                svm_params['gamma'] = 'scale'
            elif params['gamma_type'] == 'auto':
                svm_params['gamma'] = 'auto'
            else:
                svm_params['gamma'] = params['gamma_value']
        
        try:
            # 创建SVM模型
            model = SVC(**svm_params)
            
            # 使用交叉验证评估
            uar_scorer = make_scorer(recall_score, average='macro')
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=3, scoring=uar_scorer, n_jobs=1)
            
            # 返回负平均UAR（用于最小化）
            mean_uar = cv_scores.mean()
            print(f"   Params: {svm_params} -> UAR: {mean_uar:.4f}")
            return -mean_uar
            
        except Exception as e:
            print(f"❌ Error in objective function: {e}")
            return 0  # 返回较差的分数
    
    # 执行贝叶斯优化
    print("🚀 Starting Bayesian optimization...")
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=10,  # 随机初始化点数
        acq_func='EI',        # Expected Improvement
        random_state=42
    )
    
    # 提取最佳参数
    best_params = {}
    param_names = ['kernel', 'C', 'gamma_type', 'gamma_value', 'degree', 'class_weight']
    for i, param_name in enumerate(param_names):
        best_params[param_name] = result.x[i]
    
    # 构建最终的SVM参数
    final_svm_params = {
        'kernel': best_params['kernel'],
        'C': best_params['C'],
        'random_state': 42,
        'class_weight': best_params['class_weight']
    }
    
    if best_params['kernel'] == 'rbf':
        if best_params['gamma_type'] == 'scale':
            final_svm_params['gamma'] = 'scale'
        elif best_params['gamma_type'] == 'auto':
            final_svm_params['gamma'] = 'auto'
        else:
            final_svm_params['gamma'] = best_params['gamma_value']
    elif best_params['kernel'] == 'poly':
        final_svm_params['degree'] = best_params['degree']
        if best_params['gamma_type'] == 'scale':
            final_svm_params['gamma'] = 'scale'
        elif best_params['gamma_type'] == 'auto':
            final_svm_params['gamma'] = 'auto'
        else:
            final_svm_params['gamma'] = best_params['gamma_value']
    
    print(f"🏆 Best parameters found: {final_svm_params}")
    print(f"📊 Best cross-validation UAR: {-result.fun:.4f}")
    
    # 训练最终模型
    best_model = SVC(**final_svm_params)
    best_model.fit(X_train, y_train)
    
    # 绘制优化过程
    plot_bayesian_optimization(result)
    
    return best_model, final_svm_params, -result.fun

def plot_bayesian_optimization(result):
    """绘制贝叶斯优化过程"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 优化过程曲线
    ax1.plot(range(len(result.func_vals)), -np.array(result.func_vals), 'bo-')
    ax1.axhline(-result.fun, color='red', linestyle='--', label=f'Best UAR: {-result.fun:.4f}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('UAR')
    ax1.set_title('Bayesian Optimization Progress')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 收敛曲线（累积最佳值）
    cumulative_best = []
    best_so_far = float('inf')
    for val in result.func_vals:
        if val < best_so_far:
            best_so_far = val
        cumulative_best.append(best_so_far)
    
    ax2.plot(range(len(cumulative_best)), -np.array(cumulative_best), 'ro-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best UAR So Far')
    ax2.set_title('Convergence Curve')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_smote.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_svm_with_predefined_split(X_train, y_train, X_devel, y_devel, 
                                  sampling_strategy='auto', n_calls=50):
    """训练SVM，使用SMOTE和贝叶斯优化"""
    
    # 使用SMOTE平衡数据
    X_train_balanced, y_train_balanced = balance_dataset_smote(
        X_train, y_train, sampling_strategy=sampling_strategy
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_devel_scaled = scaler.transform(X_devel)
    
    # 贝叶斯优化
    print("🔬 Using Bayesian Optimization with SMOTE-balanced data...")
    best_model, best_params, best_score = bayesian_optimize_svm(
        X_train_scaled, y_train_balanced, n_calls=n_calls
    )
    
    # 验证集评估
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
    
    # 计算每个类别的详细指标
    tn, fp, fn, tp = cm.ravel()
    healthy_recall = tn / (tn + fp)
    cold_recall = tp / (fn + tp)
    
    print(f"\n📊 Detailed Metrics:")
    print(f"   Healthy Recall: {healthy_recall:.4f}")
    print(f"   Cold Recall: {cold_recall:.4f}")
    print(f"   UAR: {(healthy_recall + cold_recall) / 2:.4f}")
    
    return best_model, X_devel_scaled, y_devel, y_devel_pred, scaler

def visualize_results(model, X_test, y_test, y_pred):
    """可视化SVM结果"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Cold'],
                yticklabels=['Healthy', 'Cold'], ax=ax1)
    ax1.set_title('Confusion Matrix (SMOTE + Bayesian Optimization)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. 决策函数分布
    if hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(X_test)
        
        ax2.hist([decision_scores[y_test==0], decision_scores[y_test==1]], 
                bins=30, alpha=0.7, label=['Healthy', 'Cold'])
        ax2.axvline(0, color='red', linestyle='--', label='Decision Boundary')
        ax2.set_title('Decision Function Distribution')
        ax2.set_xlabel('Decision Function Value')
        ax2.set_ylabel('Count')
        ax2.legend()
    
    # 3. 特征重要性
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
    
    # 4. PCA可视化
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
    plt.savefig('svm_smote_bayesian_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    feature_csv = "audio_features_simplified.csv"  
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
        
        print("\n🔧 SMOTE Configuration:")
        print("1. 'auto' - Balance to majority class")
        print("2. 'minority' - Only oversample minority class")
        print("3. Custom ratio (e.g., 0.8)")
        
        smote_choice = input("Choose SMOTE sampling strategy (default 'auto'): ").strip() or 'auto'
        
        if smote_choice not in ['auto', 'minority']:
            try:
                smote_choice = float(smote_choice)
            except:
                smote_choice = 'auto'
        
        n_calls = int(input("🔬 Enter number of Bayesian optimization evaluations (default 50): ") or "50")
        
        model, X_eval, y_eval, y_pred, scaler = train_svm_with_predefined_split(
            X_train, y_train, X_devel, y_devel, 
            sampling_strategy=smote_choice, 
            n_calls=n_calls
        )
        
        visualize_results(model, X_eval, y_eval, y_pred)
        
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
        
        print("✅ SVM training and evaluation completed successfully!")
    else:
        print("❌ No training data found!")