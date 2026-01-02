from typing import Callable

import torch
import json
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class Whisper:
    def __init__(self, model_id, cache_dir=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = None

        print(f"加载 Whisper 模型...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, cache_dir=cache_dir, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=torch_dtype,
            device=self.device,
        )

    def transcribe_audio(self, file_path):
        """
        使用 whisper 识别音频文件
        """
        result = self.pipe(file_path)
        return result["text"]

    def transcribe_directory(self, directory_path, output_file=None, user_id=1001, data_id:Callable[[],int]=None):
        """
        转录目录中的所有WAV文件

        参数:
            directory_path: 目录路径
            output_file: 输出文件路径（可选）
        """
        directory = Path(directory_path)

        # 查找WAV文件
        wav_files = list(directory.glob("*.wav"))

        if not wav_files:
            print(f"Error：在 {directory} 中未找到WAV文件")
            return []

        # 设置输出文件
        if output_file is None:
            output_file = directory / "whisper_transcriptions.txt"

        results = []

        for i, wav_file in enumerate(wav_files, 1):
            try:
                print(f"\n[{i}/{len(wav_files)}] 处理: {wav_file.name}")

                # 转录
                text = self.transcribe_audio(str(wav_file))

                # 保存结果
                result = {
                    "id": data_id(),
                    "file": str(wav_file),
                    "user_id": user_id,
                    "text": text,
                    "length": len(text)
                }
                results.append(result)

            except Exception as e:
                print(f"❌ 失败: {e}")

        # 保存到文件
        self._save_batch_results(results, output_file)
        return results

    def _save_batch_results(self, results, output_file):
        """保存批量处理结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


if __name__ == "__main__":
    import config as cfg
    from dataset.utils import get_absolute_path # 获取测试用数据绝对路径

    test_file_path = get_absolute_path(cfg.test_audio, cfg)
    print(test_file_path)

    # run model
    whisper = Whisper(cfg.asr_model_id)
    text = whisper.transcribe_audio(test_file_path)
    print("识别结果:")
    print(text)



