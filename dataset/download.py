import subprocess
import os
import re


def _download_media(filename, url, media_type="audio", quality=None, timeout=300):
    """
    高级下载函数，支持更多选项

    参数:
        filename: 输出文件名（包含路径）
        url: 视频URL
        media_type: 媒体类型 "audio" 或 "video"
        quality: 质量设置
    """
    try:
        # 确保输出目录存在
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            print(f"❌ 存储目录不存在！")
            return False

        # 构建命令参数
        cmd_args = ["yt-dlp"]

        # 设置输出文件名（确保包含扩展名占位符）
        if not filename.endswith(".wav"):
            filename += ".wav"
        cmd_args.extend(["-o", filename])

        # 设置媒体类型和质量
        if media_type == "audio":
            cmd_args.extend(["-x", "--audio-format", "wav"])
            if quality:
                if quality == "high":
                    cmd_args.extend(["--audio-quality", "0"])
                elif quality == "medium":
                    cmd_args.extend(["--audio-quality", "5"])
                elif quality == "low":
                    cmd_args.extend(["--audio-quality", "9"])
        else:
            if quality:
                if quality == "high":
                    cmd_args.extend(["-f", "bestvideo+bestaudio"])
                elif quality == "medium":
                    cmd_args.extend(["-f", "mp4"])
                else:
                    cmd_args.extend(["-f", "best"])
            else:
                cmd_args.extend(["-f", "best"])

        cmd_args.append(url)

        # 执行命令
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=timeout
        )

        if result.returncode == 0:
            print(f"✅ 下载成功！ 文件保存为{filename}")
            return filename
        else:
            print(f"❌ 下载失败！")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 下载超时！")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False


def download_from_bv(bv_id, save_dir, media_type="audio", quality=None, timeout=300):
    """
    BV号下载

    参数:
        bv_id: BV号（如 "BV1U9CrBzEgL"）
        save_dir: 保存目录
        media_type: 媒体类型
        quality: 质量设置
    """
    # 确保目录路径以斜杠结尾
    if not save_dir.endswith(os.path.sep):
        save_dir += os.path.sep

    url = f"https://www.bilibili.com/video/{bv_id}/"

    # 构建文件名
    if media_type == "audio":
        filename = f"{save_dir}{bv_id}"
    else:
        filename = f"{save_dir}{bv_id}"

    try:
        return _download_media(
            filename=filename,
            url=url,
            media_type=media_type,
            quality=quality,
            timeout=timeout
            )
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False


def download_from_url(url, save_dir, media_type="audio", quality=None, timeout=300):
    """
    URL直接下载

    参数:
        url: 视频URL
        save_dir: 保存目录
        media_type: 媒体类型
        quality: 质量设置
    """
    # 确保目录路径以斜杠结尾
    if not save_dir.endswith(os.path.sep):
        save_dir += os.path.sep

    # 从URL提取BV号作为文件名
    bv_match = re.search(r'/video/(BV[0-9A-Za-z]{10})', url)
    if bv_match:
        bv_id = bv_match.group(1)
        try:
            return bv_id, download_from_bv(bv_id, save_dir, media_type, quality, timeout)
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            return False
    else:
        # 如果没有BV号
        print(f"❌ Link不正确！")
        return False


# 使用示例
if __name__ == "__main__":
    # 示例1：直接使用URL下载
    url = "https://www.bilibili.com/video/BV19o2sBVEPU/?spm_id_from=333.337.search-card.all.click&vd_source=11a1fbde4b457a1090ace2fc8038a74e"
    download_from_url(
        url=url,
        save_dir=r"D:\download",
        media_type="audio",
        quality="medium"
    )
"""
    # 示例2：直接使用BV号下载
    bv = "BV1b9CrBzEga"
    download_from_bv(
        bv_id=bv,
        save_dir=r"D:\download",
        media_type="video",
        quality="high"
    )
"""