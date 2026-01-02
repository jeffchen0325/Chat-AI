import sys
import os
from dataset.download import download_from_url
from common.audio import split_audio
import config as cfg

sys.path.extend('.')

download_dir = cfg.download_dir
slice_dir_root = cfg.slice_dir_root
try:
    while True:
        url = input("\n请输入视频URL: ").strip()
        if not url:
            print("URL不能为空，请重新输入")
            continue
        if url.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break

        try:
            # download audio
            bv_id, filepath = download_from_url(url, download_dir, "audio")

            # split audio
            slice_dir = os.path.join(slice_dir_root, bv_id)
            slice_files = split_audio(filepath, slice_dir, *cfg.split_config)


        except Exception as e:
            print(f"❌ 发生错误: {e}")

except KeyboardInterrupt:
    sys.exit(0)

