import librosa
import numpy as np
import parselmouth
import matplotlib.pyplot as plt


class Praat:
    @staticmethod
    def extract_pif(wav_path, time_step=0.01):
        """
        高效同步提取音高(Pitch/F0)、声强(Intensity)、前3个共振峰(F1/F2/F3)
        返回: 时间轴 (T,), pitch (T,), intensity (T,), f1/f2/f3 (T,)
        """
        snd = parselmouth.Sound(wav_path)

        pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
        intensity = snd.to_intensity(time_step=time_step, minimum_pitch=75)
        formant = snd.to_formant_burg(
            time_step=time_step,
            max_number_of_formants=5,
            maximum_formant=5500,
            window_length=0.025
        )

        times = intensity.xs()
        int_vals = intensity.values[0]
        pitch_vals = np.array([pitch.get_value_at_time(t) for t in times])
        f1_vals = np.array([formant.get_value_at_time(1, t) for t in times])
        f2_vals = np.array([formant.get_value_at_time(2, t) for t in times])
        f3_vals = np.array([formant.get_value_at_time(3, t) for t in times])

        # 不强制替换 nan（让 matplotlib 自动断线更真实）
        # 但为了兼容性，仍可选保留；这里改为保留 nan
        return times, pitch_vals, int_vals, f1_vals, f2_vals, f3_vals

    @staticmethod
    def plot_pitch_formants(wav_path, time_step=0.01, figsize=(14, 12)):
        # 加载音频文件
        y, sr = librosa.load(wav_path, sr=None)
        duration = len(y) / sr
        time_wave = np.linspace(0, duration, len(y))

        # 提取声学特征
        times, pitch, intensity, f1, f2, f3 = Praat.extract_pif(wav_path, time_step=time_step)

        # 处理无效值
        def mask_invalid(data):
            return np.where(data == 0, np.nan, data)

        pitch = mask_invalid(pitch)
        f1, f2, f3 = map(mask_invalid, [f1, f2, f3])

        # 创建图形
        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # 波形
        axs[0].plot(time_wave, y, color='k', linewidth=0.5)
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Waveform')
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # Pitch (F0)
        axs[1].plot(times, pitch, color='b')
        axs[1].set_ylabel('Pitch (Hz)')
        axs[1].set_title('Fundamental Frequency (F0)')
        axs[1].grid(True)

        # Formants
        axs[2].plot(times, f1, label='F1', color='r')
        axs[2].plot(times, f2, label='F2', color='orange')
        axs[2].plot(times, f3, label='F3', color='purple')
        axs[2].set_ylabel('Frequency (Hz)')
        axs[2].set_title('Formants (F1, F2, F3)')
        axs[2].legend()
        axs[2].grid(True)

        # Delta Formants: 修改了 ΔF3 的定义以避免混淆
        axs[3].plot(times, f2 - f1, label='ΔF2=F2−F1', color='r')
        axs[3].plot(times, f3 - f1, label='ΔF3=F3−F1', color='orange')
        axs[3].set_ylabel('Frequency Difference (Hz)')
        axs[3].set_title('Delta Formants')
        axs[3].legend()
        axs[3].grid(True)
        axs[3].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pif(wav_path, time_step=0.01, figsize=(14, 12)):
        """
        提取并绘制：波形 + Pitch + Intensity + Formants (F1/F2/F3)
        参数:
            wav_path (str): 音频文件路径（.wav）
            time_step (float): 分析时间步长（秒）
            figsize (tuple): 图像大小
        """
        # 1. 加载原始音频用于波形
        y, sr = librosa.load(wav_path, sr=None)  # 保持原采样率
        duration = len(y) / sr
        time_wave = np.linspace(0, duration, len(y))

        # 2. 提取声学特征
        times, pitch, intensity, f1, f2, f3 = Praat.extract_pif(wav_path, time_step=time_step)

        # 3. 创建图形（5 行：波形 + pitch + intensity + formants x2）
        fig, axs = plt.subplots(5, 1, figsize=figsize, sharex=True)

        # 波形
        axs[0].plot(time_wave, y, color='k', linewidth=0.5)
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Waveform')
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # Pitch (F0)
        axs[1].plot(times, pitch, color='b')
        axs[1].set_ylabel('Pitch (Hz)')
        axs[1].set_title('Fundamental Frequency (F0)')
        axs[1].grid(True)

        # Intensity
        axs[2].plot(times, intensity, color='g')
        axs[2].set_ylabel('Intensity (dB)')
        axs[2].set_title('Intensity')
        axs[2].grid(True)

        # Formants - 单独图
        axs[3].plot(times, f1, label='F1', color='r')
        axs[3].plot(times, f2, label='F2', color='orange')
        axs[3].plot(times, f3, label='F3', color='purple')
        axs[3].set_ylabel('Frequency (Hz)')
        axs[3].set_title('Formants (F1, F2, F3)')
        axs[3].legend()
        axs[3].grid(True)

        # Formants - 与波形时间对齐的完整视图（可选冗余，但清晰）
        axs[4].plot(times, f1, label='F1', color='r')
        axs[4].plot(times, f2, label='F2', color='orange')
        axs[4].plot(times, f3, label='F3', color='purple')
        axs[4].set_xlabel('Time (s)')
        axs[4].set_ylabel('Frequency (Hz)')
        axs[4].set_title('Formants Over Time (Aligned with Waveform)')
        axs[4].legend()
        axs[4].grid(True)

        plt.tight_layout()
        plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    audiopath = r"D:\download\BV1cPSsBhEU5.wav"
    Praat.plot_pitch_formants(audiopath, time_step=0.01)