import sys
import os
import warnings
from queue import Queue, Empty
from Common.audio import Recorder, play_audio_data
from Common.Whisper import Whisper
from Common.Kokoro import Kokoro
from Common.DeepSeek import Deepseek
from Common.Sentense import split_paragraph
import config as cfg

sys.path.extend('.')
warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("./temp", exist_ok=True)

# load models
output_queue = Queue()
recorder = Recorder()
whisper = Whisper(cfg.asr_model_id)
deepseek = Deepseek(cfg.token_file, cfg.llm_base_url, cfg.llm_model_id)
kokoro = Kokoro(cfg.kokoro_model_path, cfg.kokoro_config_path, cfg.kokoro_model_id, cfg.kokoro_voice_path)

test_audio    = "./temp/test.wav"
input_speech  = "./temp/input.wav"
output_speech = "./temp/output.wav"

# 模型预热
_ = whisper.transcribe_audio(test_audio)
_ = deepseek.chat(content='你好', padding=False)
kokoro.tts(output_speech, '你好，有什么可以帮您的？')
print('对话开始...')

# 开始对话
try:
    while True:
        # 开始录音
        recorder.set_record_enable(True)
        print(f'用户:')
        result = recorder.record_audio(input_speech)
        if result:
            recorder.set_record_enable(False)

            # ASR
            user_input = whisper.transcribe_audio(input_speech)
            print(user_input)
            print()

            # LLM
            response = deepseek.chat(user_input)

            # 分句
            sentences = split_paragraph(response)

            # 清空队列，避免旧数据干扰
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except Empty:
                    break

            # TTS生成
            kokoro.tts_stream(output_queue, sentences)

            print(f'系统:')
            while True:
                try:
                    item = output_queue.get(timeout=2.0)
                    if item is None:  # 结束标记
                        print()
                        break

                    # 获取音频数据和对应文本
                    audio_data = item['audio']
                    text = item['text']

                    # 显示文本并播放音频
                    print(text)
                    play_audio_data(audio_data)

                except Empty:
                    print("\n⚠️ TTS生成超时，跳过当前回复")

except KeyboardInterrupt:
    print("\n\n对话结束，再见！")
    # 清理资源
    if 'whisper' in locals():
        del whisper
    if 'kokoro' in locals():
        del kokoro
    if 'deepseek' in locals():
        del deepseek
    sys.exit(0)  # 正常退出
except Exception as e:
    print(f"\n程序出错: {e}")
    import traceback

    traceback.print_exc()
    # 清理资源
    if 'whisper' in locals():
        del whisper
    if 'kokoro' in locals():
        del kokoro
    if 'deepseek' in locals():
        del deepseek
    sys.exit(1)  # 异常退出
