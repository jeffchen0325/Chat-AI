sr = 16000
window = 512
silence = 2
minDuration = 0.5

models_root = "D:/models/"
# ASR模型配置
asr_model_id = models_root + "models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

# LLM模型配置
# deepseek
ds_base_url = "https://api.deepseek.com"
ds_model_id = "deepseek-chat"
ds_token    = models_root + "ds_token.txt"
ds_config = (ds_token, ds_base_url, ds_model_id)

# qianwen
qw_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
qw_model_id = "qwen3-max"
qw_token    = models_root + "qw_token.txt"
qw_config = (qw_token, qw_base_url, qw_model_id)

# TTS模型配置
kokoro_model_id = "hexgrad/Kokoro-82M-v1.1-zh"
kokoro_model_path = models_root + 'models--hexgrad--Kokoro-82M/kokoro-v1_1-zh.pth'
kokoro_config_path = models_root + 'models--hexgrad--Kokoro-82M/config.json'
kokoro_voice_path = models_root + "models--hexgrad--Kokoro-82M/voice/"
kokoro_config = (kokoro_model_path, kokoro_config_path, kokoro_model_id, kokoro_voice_path)

coqui_model_id = "coqui/XTTS-v2"
coqui_model_path = ""

# DataSet配置
download_dir = r"D:\download"
slice_dir_root = r"D:\audiodata"
test_audio   = r"C:\Users\Administrator\PycharmProjects\ChatAI\temp\example.wav"

# 语音切片设置
min_duration = 10.0         # 片段最短长度，单位：s
max_duration = 30.0         # 片段最短长度，单位：s
skip_fragment = 1.0         # 丢弃长度，单位：s
min_silence_len = 300       # 中文自然停顿多在 200–500ms，300ms 是合理切分点。
silence_thresh = -38        # 比默认 -35 更敏感，能切出更多片段，但不会误伤轻声。
keep_silence = 150          # 保证“你好”不会变成“好
split_config = (min_duration, max_duration, skip_fragment, min_silence_len, silence_thresh, keep_silence)

# 测试用audiopath
audiofile = r"D:\download\BV1cPSsBhEU5.wav"
