import os
from queue import Queue
from typing import List
import soundfile as sf
from kokoro import KPipeline, KModel


class Kokoro:
    def __init__(self, model_dir, config_dir, repo_id, voice_path, voice_id:str='af_maple', sr=24000):
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.voice_path = voice_path
        self.voice_id = voice_id
        self.target_sample_rate = sr

        print(f"加载 Kokoro 模型...")
        try:
            # Load model with safetensors to avoid the vulnerability
            self.model = KModel(model=model_dir, config=config_dir, repo_id=repo_id).to(self.device).eval()
            self.pipe = KPipeline(lang_code='z', repo_id=repo_id, model=self.model)
            print(f"Kokoro 模型加载成功")
        except Exception as e:
            print(f"加载模型时出错: {e}")


    def tts(self, file_name, text, voice_id=None):
        """生成音频文件"""
        voice_id = voice_id or self.voice_id
        voice_tensor = f"{os.path.join(self.voice_path, voice_id)}.pt"
        if not os.path.exists(voice_tensor):
            raise FileNotFoundError(f"语音文件不存在: {voice_tensor}")

        generator = self.pipe(text, voice=voice_tensor, speed=self._speed_callable)
        wav = next(generator).audio

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        sf.write(file_name, wav, self.target_sample_rate)

    def tts_stream(self, output_queue: Queue, text:List[str], voice_id=None):
        """流式生成音频数据"""

        voice_id = voice_id or self.voice_id
        voice_tensor = f"{os.path.join(self.voice_path, voice_id)}.pt"

        if not os.path.exists(voice_tensor):
            raise FileNotFoundError(f"语音文件不存在: {voice_tensor}")

        for sentence in text:
            if not sentence.strip():  # 跳过空句子
                continue
            try:
                # 生成音频
                generator = self.pipe(sentence, voice=voice_tensor, speed=self._speed_callable)
                wav = next(generator).audio

                # 将音频数据放入队列
                output_queue.put({'audio':wav, 'text':sentence})

            except Exception as e:
                print(f"处理句子时出错: '{sentence}', 错误: {e}")
                # 可以选择继续处理下一个句子或抛出异常
                continue

        # 放入结束标记
        output_queue.put(None)

    def _speed_callable(self, len_ps):
        """根据文本长度调整语速"""
        if len_ps <= 83: return 1.1
        if len_ps < 183: return (1 - (len_ps - 83) / 500) * 1.1
        return 0.8 * 1.1



if __name__ == "__main__":
    from Common.audio import play_audio_file

    # 配置参数
    voice_zf = "zf_001"
    voice_af = 'af_maple'
    sentence = '你好，这是一个语音合成测试。'
    file_path = r"C:\Users\Administrator\PycharmProjects\ChatAI\temp\output.wav"

    # 模型路径
    model_id = "hexgrad/Kokoro-82M-v1.1-zh"
    model_path = '../model/models--hexgrad--Kokoro-82M/kokoro-v1_1-zh.pth'
    config_path = '../model/models--hexgrad--Kokoro-82M/config.json'
    voice_path = "../model/models--hexgrad--Kokoro-82M/voice"

    # run model
    kokoro = Kokoro(model_path, config_path, model_id, voice_path, voice_af)
    kokoro.tts(file_path, sentence)
    play_audio_file(file_path)
