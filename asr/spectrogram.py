import librosa
import librosa.display
from gammatone.gtgram import gtgram
import matplotlib.pyplot as plt
import numpy as np


class AudioSpectrogram:
    def __init__(self, sr=16000, window=0.025, hop=0.01, nchannel=64, f_min=50, n_mels=64):
        self.sr = sr
        self.window = window      # 窗长（秒）
        self.hop = hop            # 帧移（秒）
        self.nchannel = nchannel  # Gammatone 通道数
        self.f_min = f_min        # Gammatone 最低频率
        self.n_mels = n_mels      # Mel 频带数（用于公平对比）

        # 预计算帧长和帧移（以采样点为单位）
        self.win_length = int(self.window * self.sr)
        self.hop_length = int(self.hop * self.sr)

    def _build_gt_params(self):
        return {
            'fs': self.sr,
            'window_time': self.window,
            'hop_time': self.hop,
            'channels': self.nchannel,
            'f_min': self.f_min
        }

    def compute_gammatone(self, y):
        """计算 Gammatone spectrogram (dB)"""
        params = self._build_gt_params()
        gt_spec = gtgram(y, **params)  # (nchannel, time_frames)
        #归一化
        ref_power = np.mean(y ** 2) + 1e-12
        return librosa.power_to_db(gt_spec, ref=ref_power)

    def compute_mel(self, y):
        """计算 Mel spectrogram (dB)，使用与 Gammatone 相同的时间分辨率"""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.sr // 2,
            window='hann'
        )
        #归一化
        ref_power = np.mean(y ** 2) + 1e-12
        return librosa.power_to_db(mel_spec, ref=ref_power)

    def load_audio(self, audiofile):
        y, sr = librosa.load(audiofile, sr=self.sr)
        if sr != self.sr:
            raise ValueError(f"Expected sample rate {self.sr}, got {sr}")
        return y

    def compare_visualization(self, audiofile):
        """并排可视化 Gammatone vs Mel spectrogram"""
        y = self.load_audio(audiofile)

        gt_db = self.compute_gammatone(y)
        mel_db = self.compute_mel(y)

        # 共享 dB 范围
        vmin = -60  # 或 min(gt_db.min(), mel_db.min())
        vmax = 20  # 因为 ref=np.max，最大值为 0 dB

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Gammatone
        img1 = librosa.display.specshow(gt_db, sr=self.sr, hop_length=self.hop_length,
                                        x_axis='time', y_axis=None, cmap='magma',
                                        vmin=vmin, vmax=vmax, ax=ax1)
        ax1.set_ylabel('Gammatone Channel')
        ax1.set_title(f'Gammatone Spectrogram ({self.nchannel} channels, f_min={self.f_min} Hz)')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

        # Mel
        img2 = librosa.display.specshow(mel_db, sr=self.sr, hop_length=self.hop_length,
                                        x_axis='time', y_axis='mel', cmap='magma',
                                        vmin=vmin, vmax=vmax, ax=ax2)
        ax2.set_ylabel('Mel Frequency')
        ax2.set_title(f'Mel Spectrogram ({self.n_mels} bands, f_min={self.f_min} Hz)')
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

        plt.tight_layout()
        plt.show()


# ===== 使用示例 =====
if __name__ == "__main__":
    import config as cfg
    audiofile = cfg.audiofile

    spec = AudioSpectrogram(
        sr=16000,
        window=0.025,
        hop=0.01,
        nchannel=64,
        f_min=80,
        n_mels=64  # 保持通道数一致，便于对比
    )

    spec.compare_visualization(audiofile)