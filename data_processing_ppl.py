import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import noisereduce as nr
from pedalboard import *
from scipy.io import wavfile
import pyloudnorm as pyln  

import concurrent.futures


def load_audio_file(file_path, wav_file, path):
    full_path = os.path.join(file_path, path, wav_file)
    try:
        y, sr = librosa.load(full_path, sr=None)
        return (y, sr, wav_file)
    except Exception as e:
        print(f"❌ Error loading {wav_file}: {e}")
        return None

def denoise_audio(filename, file_path):
    y, sr = librosa.load(os.path.join(file_path, filename), sr=None)
    # Noise reduction
    reduced_noise = nr.reduce_noise(
        y=y, sr=sr, stationary=True,
        prop_decrease=0.82,
        time_mask_smooth_ms=50,
        freq_mask_smooth_hz=100
    )

    # Pedalboard enhancement
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-16, ratio=4),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
        Gain(gain_db=2)
    ])
    effected = board(reduced_noise, sr)
    return effected, sr

def pre_emphsis_filter(signal, alpha=0.9):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def loudness_normalization(y, sr, target_db=-23.0):
    meter    = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    y_norm   = pyln.normalize.loudness(y, loudness, target_db)
    peak = np.max(np.abs(y_norm))
    if peak > 1.0:
        y_norm = y_norm / peak
    return y_norm.astype(np.float32)


    
if __name__ == "__main__":
    file_path = './ComParE2017_Cold_4students/wav/train_files' # Path to the directory containing wav files
    folder_name = ''
    out_dir = './processed_files'  # Output directory for processed files

    wav_files = sorted([f for f in os.listdir(file_path) if f.endswith('.wav')])
    print(f"\n📂 Loading processed set: {len(wav_files)} files")
    max_workers = os.cpu_count() - 1 or 1 

    # Process files in parallel
    original_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_audio_file, file_path, wav_file, folder_name): wav_file 
                        for wav_file in wav_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                        total=len(future_to_file),
                        desc="Loading audio files"):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    original_data.append(filename)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    print(f"✅ Successfully loaded {len(original_data)} of {len(wav_files)} files")

    denoised_data = []
    for filename in tqdm(original_data, desc="Denoising audio files", unit="file"):
        y_denoised, sr = denoise_audio(filename, file_path)
        if y_denoised is not None:
            denoised_data.append((filename, y_denoised, sr))
        else:
            print(f"❌ Error denoising {filename}")
    print(f"✅ Successfully denoised {len(denoised_data)} files")

    pre_emphsis_filter_data = []
    for filename, y, sr in tqdm(denoised_data, desc="Applying pre-emphasis filter", unit="file"):
        y_pre_emph = pre_emphsis_filter(y)
        if y_pre_emph is not None:
            pre_emphsis_filter_data.append((filename, y_pre_emph, sr))
        else:
            print(f"❌ Error applying pre-emphasis filter to {filename}")
    print(f"✅ Successfully applied pre-emphasis filter to {len(pre_emphsis_filter_data)} files")


    os.makedirs(os.path.join(file_path, out_dir), exist_ok=True)
    for filename, y, sr in tqdm(pre_emphsis_filter_data, desc="Normalizing loudness", unit="file"):
        y_normalized = loudness_normalization(y, sr, target_db=-30.0)
        if y_normalized is not None:
            output_path = os.path.join(file_path, out_dir, filename)
            wavfile.write(output_path, sr, (y_normalized * 32767).astype(np.int16))
        else:
            print(f"❌ Error normalizing loudness for {filename}")
    print(f"✅ Successfully normalized loudness for {len(pre_emphsis_filter_data)} files")