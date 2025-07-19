import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

def extract_audio_features_mp(file_info):
    file_path, filename, split = file_info
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_flat = librosa.feature.spectral_flatness(y=y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        features = {
            'filename': filename,
            'split': split,
            'duration': len(y) / sr,
        }
        
        # 简化的MFCC特征 - 只计算整体的均值和标准差
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        
        # 简化的Chroma特征 - 只计算整体的均值和标准差
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # 保留Spectral Contrast的分层处理（因为维度较少）
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
        
        # 其他特征保持不变
        features['spectral_flatness_mean'] = np.mean(spec_flat)
        features['spectral_flatness_std'] = np.std(spec_flat)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        features['zcr_mean'] = np.mean(zero_crossing_rate)
        features['zcr_std'] = np.std(zero_crossing_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    data_split = ['train_files', 'test_files', 'devel_files']
    folder_path = '../ComParE2017_Cold_4students/wav/'
    
    all_file_info = []
    for split in data_split:
        audio_dir = os.path.join(folder_path, split, "processed_files")
        if os.path.exists(audio_dir):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            for filename in audio_files:
                file_path = os.path.join(audio_dir, filename)
                all_file_info.append((file_path, filename, split))
    
    print(f"🎯 Total files: {len(all_file_info)}")
    
    num_processes = min(mp.cpu_count(), 8) 
    print(f"🚀 Using {num_processes} processes")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(
            executor.map(extract_audio_features_mp, all_file_info),
            total=len(all_file_info),
            desc="Processing files"
        ))
    
    all_features = [r for r in results if r is not None]
    
    print(f"✅ Successfully processed {len(all_features)} files")
    
    if all_features:
        df = pd.DataFrame(all_features)
        output_file = 'audio_features_simplified.csv'
        df.to_csv(output_file, index=False)
        print(f"✅ Features saved to {output_file}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Feature columns: {list(df.columns)}")
    else:
        print("❌ No features extracted!")