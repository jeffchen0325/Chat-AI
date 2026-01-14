from typing import Callable, Optional, List, Dict, Any
import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class Whisper:
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        language: Optional[str] = None,   # ← 新增：控制语言
        task: str = "transcribe",         # ← 新增："transcribe" 或 "translate"
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.language = language
        self.task = task

        print(f"加载 Whisper 模型: {model_id} (device={self.device})")

        # 加载模型
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        # 创建 pipeline（不传 dtype，已由 model.to() 控制）
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device,
        )

    def transcribe_audio(self, file_path: str) -> str:
        """使用 whisper 识别音频文件"""
        # ✅ 关键：通过 generate_kwargs 传递新参数
        generate_kwargs = {}
        if self.language is not None:
            generate_kwargs["language"] = self.language
        if self.task is not None:
            generate_kwargs["task"] = self.task

        result = self.pipe(file_path, generate_kwargs=generate_kwargs)
        return result["text"]

    def transcribe_directory(
        self,
        directory_path: str,
        output_file: Optional[str] = None,
        user_id: int = 1001,
        data_id: Optional[Callable[[], int]] = None,
    ) -> List[Dict[str, Any]]:
        """转录目录中的所有WAV文件"""
        directory = Path(directory_path)
        wav_files = list(directory.glob("*.wav"))

        if not wav_files:
            print(f"Error：在 {directory} 中未找到WAV文件")
            return []

        if output_file is None:
            output_file = directory / "whisper_transcriptions.jsonl"  # ← 改为 .jsonl 更合理

        results = []
        for i, wav_file in enumerate(wav_files, 1):
            try:
                print(f"\n[{i}/{len(wav_files)}] 处理: {wav_file.name}")
                text = self.transcribe_audio(str(wav_file))

                result = {
                    "id": data_id() if data_id else i,
                    "file": str(wav_file),
                    "user_id": user_id,
                    "text": text.strip(),  # ← 去除首尾空白
                    "length": len(text),
                }
                results.append(result)

            except Exception as e:
                print(f"❌ 处理 {wav_file.name} 失败: {e}")

        self._save_batch_results(results, output_file)
        return results

    def _save_batch_results(self, results: List[Dict], output_file: str):
        """保存为 JSONL 格式（每行一个 JSON 对象）"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"✅ 转录结果已保存至: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


# ======================
# 示例用法
# ======================
if __name__ == "__main__":
    from configs import config as cfg
    from src.dataset.utils import get_absolute_path

    test_file_path = get_absolute_path(cfg.test_audio, cfg)
    print("测试文件:", test_file_path)

    # ✅ 示例 1: 自动检测语言并转录（默认）
    whisper = Whisper(cfg.asr_model_id)

    # ✅ 示例 2: 强制翻译成英文（避免行为变更）
    # whisper = Whisper(cfg.asr_model_id, language="en", task="translate")

    text = whisper.transcribe_audio(test_file_path)
    print("\n识别结果:")
    print(text)