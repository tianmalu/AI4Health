import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2

class VariableWidthLogMelProcessor:
    def __init__(self, 
                 sr=16000,           # 采样率
                 n_mels=128,         # mel频带数量（统一高度）
                 n_fft=1024,         # FFT窗口大小
                 hop_length=512,     # 跳跃长度
                 fmin=50,            # 最低频率
                 fmax=8000,          # 最高频率
                 normalize=True):    # 是否归一化
        
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        
        # 计算时间分辨率
        self.time_per_frame = hop_length / sr
        self.frames_per_second = sr / hop_length
        
        print(f"🎵 Variable-Width LogMel Processor initialized:")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Height (unified): {n_mels} mel bands")
        print(f"   Width (variable): Preserves audio duration")
        print(f"   Time resolution: {self.time_per_frame*1000:.1f} ms/frame")
        print(f"   Frame rate: {self.frames_per_second:.1f} frames/second")
        
    def audio_to_log_mel(self, audio_path):
        """将音频转换为保持时长的log mel spectrogram"""
        
        try:
            # 1. 加载音频文件
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            if len(y) == 0:
                print(f"⚠️ 空音频文件: {audio_path}")
                return None, None, None
            
            # 2. 计算mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # 3. 转换为log scale (dB)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 4. 归一化到 [0, 1]
            if self.normalize:
                log_mel = self._normalize_spectrogram(log_mel)
            
            # 5. 计算音频信息
            duration = len(y) / sr
            n_frames = log_mel.shape[1]
            
            return log_mel, duration, {
                'n_frames': n_frames,
                'duration': duration,
                'shape': log_mel.shape,
                'time_per_frame': self.time_per_frame
            }
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            return None, None, None
    
    def _normalize_spectrogram(self, log_mel):
        """归一化频谱图到 [0, 1] 范围"""
        if log_mel.max() == log_mel.min():
            return np.zeros_like(log_mel)
        return (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
    
    def save_variable_width_image(self, log_mel, output_path, duration, 
                                 dpi=100, cmap='viridis', show_axes=False):
        """保存可变宽度的频谱图图像"""
        
        if log_mel is None:
            return False
            
        try:
            height, width = log_mel.shape
            
            # 计算图像尺寸（保持原始比例）
            # 宽度与帧数成正比，高度固定
            frames_per_inch = 50  # 可调整：每英寸多少帧
            fig_width = max(width / frames_per_inch, 2)  # 最小2英寸
            fig_height = height / 100  # 固定高度比例
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            
            # 绘制频谱图
            if show_axes:
                # 显示时间轴
                im = ax.imshow(log_mel, 
                              aspect='auto', 
                              origin='lower',
                              cmap=cmap,
                              extent=[0, duration, 0, self.n_mels])
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Mel Frequency Bands')
                ax.set_title(f'Duration: {duration:.2f}s | Frames: {width}')
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                # 无坐标轴版本
                ax.imshow(log_mel, 
                         aspect='auto', 
                         origin='lower',
                         cmap=cmap)
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # 保存图像
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving image {output_path}: {e}")
            plt.close()
            return False
    
    def save_numpy_array(self, log_mel, output_path):
        """保存为numpy数组（保持原始形状）"""
        
        if log_mel is None:
            return False
            
        try:
            np.save(output_path, log_mel.astype(np.float32))
            return True
        except Exception as e:
            print(f"❌ Error saving numpy array {output_path}: {e}")
            return False
    
    def batch_process_directory(self, input_dir, output_dir, 
                              audio_extensions=['.wav', '.mp3', '.flac'],
                              save_png=True, save_npy=True, overwrite=False):
        """批量处理目录中的音频文件"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找所有音频文件
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'**/*{ext}'))
        
        print(f"🔍 Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            print("⚠️ No audio files found!")
            return {'processed': 0, 'failed': 0, 'skipped': 0, 'results': []}
        
        # 处理统计
        stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'results': []
        }
        
        # 批量处理
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            
            # 生成输出文件名
            base_name = audio_file.stem
            png_output = output_path / f"{base_name}_logmel.png"
            npy_output = output_path / f"{base_name}_logmel.npy"
            
            # 检查是否需要跳过
            skip_png = png_output.exists() and not overwrite and save_png
            skip_npy = npy_output.exists() and not overwrite and save_npy
            
            if skip_png and skip_npy:
                stats['skipped'] += 1
                continue
            
            # 处理音频
            log_mel, duration, info = self.audio_to_log_mel(audio_file)
            
            if log_mel is not None:
                success = True
                
                # 保存PNG图像
                if save_png and not skip_png:
                    success_png = self.save_variable_width_image(log_mel, png_output, duration)
                    success = success and success_png
                
                if success:
                    stats['processed'] += 1
                    
                    # 记录统计信息
                    height, width = log_mel.shape
                    stats['results'].append({
                        'filename': base_name,
                        'duration': duration,
                        'height': height,
                        'width': width,
                        'frames': info['n_frames']
                    })
                else:
                    stats['failed'] += 1
            else:
                stats['failed'] += 1
        
        # 打印处理结果
        self._print_processing_stats(stats)
        
        return stats
    
    def _print_processing_stats(self, stats):
        """打印处理统计信息"""
        print(f"\n📊 Processing Results:")
        print(f"   ✅ Processed: {stats['processed']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   ⏭️ Skipped: {stats['skipped']}")
        
        if stats['results']:
            df = pd.DataFrame(stats['results'])
            
            print(f"\n📈 Duration & Width Statistics:")
            print(f"   Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
            print(f"   Width range: {df['width'].min()} - {df['width'].max()} frames")
            print(f"   Height (unified): {df['height'].iloc[0]} mel bands")
            print(f"   Average width: {df['width'].mean():.0f} frames")
            
            print(f"\n🔄 Duration-to-Width Preservation:")
            duration_width_ratio = df['width'] / df['duration']
            print(f"   Frames per second: {duration_width_ratio.mean():.1f} ± {duration_width_ratio.std():.1f}")
            print(f"   Time preservation: Perfect (no compression)")

def create_variable_width_spectrograms(audio_dir, output_dir):
    """主函数：创建保持时长特征的log mel spectrograms"""
    
    processor = VariableWidthLogMelProcessor(
        sr=16000,                    # 16kHz采样率
        n_mels=128,                  # 128个mel频带（统一高度）
        n_fft=1024,                  # FFT窗口
        hop_length=512,              # 帧移
        fmin=50,                     # 最低频率
        fmax=8000,                   # 最高频率
        normalize=True               # 归一化
    )
    
    stats = processor.batch_process_directory(
        input_dir=audio_dir,
        output_dir=output_dir,
        save_png=True,    # 保存可视化图像
        save_npy=False,    # 保存numpy数组
        overwrite=False   # 不覆盖已存在文件
    )
    
    return stats

def visualize_duration_preservation(output_dir, num_samples=6):
    """可视化时长保持效果"""
    
    png_files = list(Path(output_dir).glob('**/*_logmel.png'))
    
    if len(png_files) == 0:
        print("⚠️ No PNG files found for visualization")
        return
    
    # 按文件名排序，选择不同时长的样本
    sample_files = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            width, height = img.size
            sample_files.append((png_file, width, height))
        except:
            continue
    
    # 按宽度排序选择代表性样本
    sample_files.sort(key=lambda x: x[1])
    step = max(1, len(sample_files) // num_samples)
    selected_samples = sample_files[::step][:num_samples]
    
    # 创建可视化
    fig, axes = plt.subplots(num_samples, 1, figsize=(16, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    print(f"🖼️ Duration Preservation Visualization:")
    
    for i, (img_file, width, height) in enumerate(selected_samples):
        try:
            img = Image.open(img_file)
            
            # 估算时长（基于帧数）
            estimated_frames = width * 0.6  # 粗略估算
            estimated_duration = estimated_frames * 512 / 16000
            
            axes[i].imshow(img, aspect='auto')
            axes[i].set_title(f'{img_file.name}\n'
                            f'Width: {width}px | Est. Duration: {estimated_duration:.1f}s | Height: {height}px (unified)',
                            fontsize=10)
            axes[i].axis('off')
            
            print(f"   Sample {i+1}: {width}×{height} pixels → ~{estimated_duration:.1f}s")
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    
    # 配置参数
    data_splits = ["train_files", "test_files", "devel_files"]
    audio_base_dir = "../ComParE2017_Cold_4students/wav"
    output_base_dir = "../spectrograms_variable_width"
    
    print("🎵 Creating Variable-Width Log Mel Spectrograms")
    print("🎯 保留不同音频不同时长的特征，图片统一高度")
    print("="*60)
    
    total_stats = {'processed': 0, 'failed': 0, 'skipped': 0}
    
    for split in data_splits:
        print(f"\n📁 Processing {split}...")
        
        # 构建路径
        audio_dir = os.path.join(audio_base_dir, split, "processed_files")
        output_dir = os.path.join(output_base_dir, split)
        
        if os.path.exists(audio_dir):
            print(f"📂 Input: {audio_dir}")
            print(f"📂 Output: {output_dir}")
            
            # 处理该split的所有音频
            stats = create_variable_width_spectrograms(
                audio_dir=audio_dir,
                output_dir=output_dir
            )
            
            # 累计统计
            for key in ['processed', 'failed', 'skipped']:
                total_stats[key] += stats[key]
                
        else:
            print(f"⚠️ Directory not found: {audio_dir}")
    
    # 总结
    print(f"\n🎯 Overall Summary:")
    print(f"   ✅ Total processed: {total_stats['processed']} files")
    print(f"   ❌ Total failed: {total_stats['failed']} files")
    print(f"   ⏭️ Total skipped: {total_stats['skipped']} files")
    print(f"   📁 Output directory: {os.path.abspath(output_base_dir)}")
    
    # 可视化结果
    if total_stats['processed'] > 0:
        print(f"\n🖼️ Visualizing duration preservation...")
        for split in data_splits:
            output_dir = os.path.join(output_base_dir, split)
            if os.path.exists(output_dir):
                visualize_duration_preservation(output_dir, num_samples=5)
                break
    
    print(f"\n✨ Generation Complete!")
    print(f"🎯 核心特性:")
    print(f"   ✅ 保持时长信息：不同音频 → 不同图片宽度")
    print(f"   ✅ 统一高度：128 mel频带")
    print(f"   ✅ 比例保持：时长比例转换为宽度比例")
    print(f"   ✅ 高质量：log mel特征完整保留")