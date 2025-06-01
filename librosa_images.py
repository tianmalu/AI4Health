import librosa
import librosa.display
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

def compute_spectrogram_image(file_path, output_dir, filename):
    try:
        y, sr = librosa.load(file_path, sr=22050) 
        
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=128,       
            fmax=8000,        
            hop_length=512,    
            n_fft=2048       
        )
        
        log_S = librosa.power_to_db(S, ref=np.max)
        
        numpy_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_logmel.npy")
        np.save(numpy_output_path, log_S)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_S, 
            x_axis='time', 
            y_axis='mel', 
            sr=sr,
            fmax=8000,
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Log-Mel Spectrogram: {filename}')
        plt.tight_layout()
        
        image_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_logmel.png")
        plt.savefig(image_output_path, dpi=150, bbox_inches='tight')
        plt.close() 
        
        return {
            'image_path': image_output_path,
            'numpy_path': numpy_output_path,
            'shape': log_S.shape,
            'duration': len(y) / sr
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_audio_file(args):
    file_path, filename, split = args
    
    output_dir = os.path.join('spectrogram_images', 'log_mel', split)
    os.makedirs(output_dir, exist_ok=True)
    
    result = compute_spectrogram_image(file_path, output_dir, filename)
    if result:
        result['filename'] = filename
        result['split'] = split
        result['file_path'] = file_path
    
    return result

if __name__ == "__main__":
    data_split = ["train_files", "test_files", "devel_files"]  # 改为正确的文件夹名
    audio_dir = "./ComParE2017_Cold_4students/wav"
    
    for split in data_split:
        os.makedirs(f'spectrogram_images/log_mel/{split}', exist_ok=True)
    
    all_tasks = []
    file_counts = {}
    
    for split in data_split:
        audio_split_dir = os.path.join(audio_dir, split, "processed_files")
        print(f'Scanning {audio_split_dir}')
        
        if not os.path.exists(audio_split_dir):
            print(f"Directory {audio_split_dir} does not exist, skipping...")
            continue
            
        audio_files = [f for f in os.listdir(audio_split_dir) if f.endswith('.wav')]
        file_counts[split] = len(audio_files)
        print(f"Found {len(audio_files)} audio files in {split}")
        
        for filename in audio_files:
            file_path = os.path.join(audio_split_dir, filename)
            all_tasks.append((file_path, filename, split))
    
    print(f"\n🎯 Total files to process: {len(all_tasks)}")
    print(f"📊 File distribution: {file_counts}")
    
    max_workers = min(4, mp.cpu_count())
    print(f"🔧 Using {max_workers} workers for parallel processing")
    
    results = []
    failed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_audio_file, task): task 
            for task in all_tasks
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), 
                          total=len(all_tasks), 
                          desc="Generating Log-Mel spectrograms"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed_count += 1
            except Exception as e:
                task = future_to_task[future]
                print(f"Error processing task {task}: {e}")
                failed_count += 1
    
    print(f"\n✅ Successfully generated {len(results)} Log-Mel spectrograms")
    print(f"❌ Failed to process {failed_count} files")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv('log_mel_processing_results.csv', index=False)
        print(f"📊 Processing results saved to log_mel_processing_results.csv")
        
        print(f"\n📈 Statistics:")
        print(f"  Average duration: {df['duration'].mean():.2f}s")
        print(f"  Min duration: {df['duration'].min():.2f}s")
        print(f"  Max duration: {df['duration'].max():.2f}s")
        
        print(f"\n📊 Split distribution:")
        print(df['split'].value_counts())
        
        shapes = df['shape'].apply(lambda x: f"{x[0]}x{x[1]}").value_counts()
        print(f"\n🔲 Spectrogram shapes:")
        print(shapes.head())
    
    print(f"\n📁 Output saved to: {os.path.abspath('spectrogram_images/log_mel')}")
    print(f"🎯 Each file generated:")
    print(f"   - PNG image for visualization")
    print(f"   - NPY array for ML processing")