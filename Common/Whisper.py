import torch
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


if __name__ == "__main__":
    audio_file_path = r"C:\Users\Administrator\PycharmProjects\ChatAI\temp\output.wav"

    # download
    #asr_model_id = "openai/whisper-large-v3-turbo"
    #cache_dir = "../model"

    # deployed in local
    asr_model_id = "C:/Users/Administrator/PycharmProjects/ChatAI/model/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    cache_dir = None

    # run model
    whisper = Whisper(asr_model_id, cache_dir)
    text = whisper.transcribe_audio(audio_file_path)
    print("识别结果:")
    print(text)



