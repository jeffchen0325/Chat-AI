import pyaudio
import torch
import numpy as np
import wave
import silero_vad
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

class Recorder:
    def __init__(self, **kwargs):
        # 音频参数
        self.sampling_rate = kwargs.get('sampling_rate', 16000)
        self.chunk_size = kwargs.get('chunk_size', 512)
        self.silence_timeout = kwargs.get('silence_timeout', 1.5)
        self.min_duration = kwargs.get('min_duration', 0.5)
        self.speech_threshold = kwargs.get('speech_threshold', 0.3)
        self.energy_threshold = kwargs.get('energy_threshold', 0.01)

        # 预计算帧数阈值
        self.silence_frames_threshold = int(self.silence_timeout * self.sampling_rate / self.chunk_size)
        self.min_duration_frames = int(self.min_duration * self.sampling_rate / self.chunk_size)

        self.enable_record = False

        # 初始化资源
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        self.vad_model = silero_vad.load_silero_vad()
        self.vad_model.reset_states()

    def set_record_enable(self, enable: bool):
        self.enable_record = enable

    def record_audio(self, file_name=None):
        if not self.enable_record:
            return False

        recording = False
        audio_buffer = []
        pre_buffer = []
        consecutive_silence_frames = 0  # 改用帧数计数

        try:
            while True:
                if not self.enable_record:
                    return False

                chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)
                energy = np.mean(audio_data ** 2)

                vad_ok = False
                if energy > self.energy_threshold * 0.5:
                    with torch.no_grad():
                        vad_ok = self.vad_model(torch.from_numpy(audio_data),
                                                self.sampling_rate).item() > self.speech_threshold

                voice_detected = (energy > self.energy_threshold) and vad_ok

                if not recording:
                    if voice_detected:
                        recording = True
                        audio_buffer = pre_buffer + [chunk]
                        consecutive_silence_frames = 0  # 重置静音计数
                    else:
                        pre_buffer.append(chunk)
                        if len(pre_buffer) > 3:
                            pre_buffer.pop(0)
                else:
                    audio_buffer.append(chunk)

                    if voice_detected:
                        consecutive_silence_frames = 0  # 检测到语音，重置计数
                    else:
                        consecutive_silence_frames += 1  # 静音帧数+1

                    # 静音检测
                    if consecutive_silence_frames >= self.silence_frames_threshold:
                        if len(audio_buffer) >= self.min_duration_frames:
                            self._save_recording(file_name, audio_buffer)
                            return True
                        else:
                            print("录音过短，已丢弃")
                            return False
        finally:
            self.vad_model.reset_states()

    def _save_recording(self, file_name, audio_buffer):
        with wave.open(file_name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sampling_rate)
            wf.writeframes(b''.join(audio_buffer))

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def play_audio_file(filename: str):
    """播放音频文件 - 阻塞调用"""
    try:
        data, fs = sf.read(filename)
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"播放失败: {e}")


def play_audio_data(audio_data, sample_rate=24000):
    """最简单的音频播放函数"""
    if audio_data is None:
        return

    try:
        # 自动处理所有格式转换
        audio_data = np.array(audio_data, dtype=np.float32).flatten()

        # 归一化到[-1, 1]范围
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # 播放并等待完成
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()

    except Exception as e:
        print(f"音频播放错误: {e}")


def split_audio(
    input_file,
    output_dir,
    min_duration=10,
    max_duration=30,
    skip_fragment=0.5,
    min_silence_len=300,
    silence_thresh=-38,
    keep_silence=150,
    sr = 16000
):
    # 参数验证
    if min_duration <= 0 or max_duration <= 0:
        raise ValueError("时长参数必须大于0")
    if min_duration > max_duration:
        raise ValueError("最小时长不能大于最大时长")
    if skip_fragment < 0:
        raise ValueError("skip_fragment 不能为负数")
    if skip_fragment > min_duration:
        raise ValueError("skip_fragment不能大于最小时长")
    if min_silence_len <= 0:
        raise ValueError("min_silence_len 必须大于0")
    if keep_silence < 0:
        raise ValueError("keep_silence 不能为负数")

    # 检查输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    min_dur_ms = int(min_duration * 1000)
    max_dur_ms = int(max_duration * 1000)
    drop_ms = int(skip_fragment * 1000)

    # 加载并标准化音频
    audio = AudioSegment.from_file(input_file)
    if audio.frame_rate != sr:
        audio = audio.set_frame_rate(sr)
    if audio.channels > 1:
        audio = audio.set_channels(1)

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    # 过滤太短的碎片（保留 ≥ skip_fragment 秒）
    chunks = [c for c in chunks if len(c) >= drop_ms]

    # 智能合并片段
    segments = []
    current = None

    for chunk in chunks:
        if current is None:
            current = chunk
            continue

        if len(current) + len(chunk) <= max_dur_ms:
            current += chunk
        else:
            # 保存 current（如果足够长或不低于 drop_ms）
            if len(current) >= min_dur_ms:
                segments.append(current)
            current = chunk

    # 处理最后一段
    if current is not None and (len(current) >= min_dur_ms):
        segments.append(current)

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = []

    for i, seg in enumerate(segments):
        duration_sec = len(seg) / 1000.0
        filename = f"{Path(input_file).stem}_part_{i+1:03d}_{duration_sec:.1f}s.wav"
        out_file = output_path / filename
        seg.export(out_file, format="wav")  # 已是单声道
        output_files.append(str(out_file))

    print(f"\n分割完成！共生成 {len(segments)} 个片段, 保存在{output_path}")
    if segments:
        durations = [len(s) / 1000.0 for s in segments]
        avg_dur = sum(durations) / len(durations)
        print(f"平均时长: {avg_dur:.1f} 秒 | 范围: {min(durations):.1f} ~ {max(durations):.1f} 秒")

    return output_files


# 使用示例
if __name__ == "__main__":
    recorder = Recorder()
    filename = "../../temp/input.wav"
    try:
        print("对话开始")
        while True:
            recorder.set_record_enable(True)
            result = recorder.record_audio(filename)
            if result:
                recorder.set_record_enable(False)
                # 播放固定的音频文件，而不是录音文件
                play_audio_file(r"../../temp/output.wav")
            else:
                print("未检测到有效录音")

    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        recorder.close()
