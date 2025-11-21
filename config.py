sr = 16000
window = 512
silence = 2
minDuration = 0.5

models_root = "D:/models/"
# ASR模型配置
asr_model_id = models_root + "models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

# LLM模型配置
llm_base_url = "https://api.deepseek.com"
llm_model_id = "deepseek-chat"
token_file = models_root + "mytoken.txt"

# TTS模型配置
kokoro_model_id = "hexgrad/Kokoro-82M-v1.1-zh"
kokoro_model_path = models_root + 'models--hexgrad--Kokoro-82M/kokoro-v1_1-zh.pth'
kokoro_config_path = models_root + 'models--hexgrad--Kokoro-82M/config.json'
kokoro_voice_path = models_root + "models--hexgrad--Kokoro-82M/voice/"
