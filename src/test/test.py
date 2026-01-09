import parselmouth
import numpy as np


def extract_f0_and_formants(audio_path, max_formant=5500.0):
    """
    同时提取 F0 和前三个共振峰（F1, F2, F3）。

    参数:
        audio_path (str): 音频路径
        max_formant (float): 最大共振峰频率（Hz）
            - 男性: 5000
            - 女性/儿童: 5500

    返回:
        dict: 包含 F0_median, F1, F2, F3
    """
    sound = parselmouth.Sound(audio_path)

    # === 提取 F0 ===
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=600)
    f0_vals = pitch.selected_array['frequency']
    valid_f0 = f0_vals[f0_vals != 0]
    f0_median = np.median(valid_f0) if len(valid_f0) >= 10 else None

    # === 提取共振峰 ===
    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
        pre_emphasis_from=50.0
    )

    f1_list, f2_list, f3_list = [], [], []
    for t in np.arange(0.01, sound.duration, 0.01):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        if f1 > 0 and f2 > 0 and f3 > 0:
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)

    return {
        'f0_median': f0_median,
        'F1': np.median(f1_list) if f1_list else None,
        'F2': np.median(f2_list) if f2_list else None,
        'F3': np.median(f3_list) if f3_list else None
    }


def classify_speaker_with_formants(features):
    """
    基于 F0 + 共振峰联合判断说话人类型。
    """
    f0 = features['f0_median']
    f1, f2, f3 = features['F1'], features['F2'], features['F3']

    if f0 is None or f1 is None:
        return "Unknown (insufficient voiced speech)"

    # 规则 1: 先用 F0 初筛
    if f0 < 150:
        speaker_type = "Male"
    elif f0 < 220:
        speaker_type = "Female"
    else:
        speaker_type = "Child"

    # 规则 2: 用共振峰验证（尤其区分高音男声 vs 女声）
    if speaker_type == "Male" and f2 > 1600:
        # 男声 F2 很少 >1600，若高，可能是女声
        speaker_type = "Female"
    elif speaker_type in ["Female", "Child"] and f2 > 2200:
        # F2 > 2200 极可能是儿童
        speaker_type = "Child"
    elif speaker_type == "Child" and f2 < 1800:
        # 儿童 F2 通常 >1800，若低，可能是女声
        speaker_type = "Female"

    return speaker_type


def analyze_speaker(audio_path, gender_hint=None):
    """
    主函数：分析音频并输出结果。

    参数:
        audio_path: 音频文件路径
        gender_hint: 可选提示（"male", "female", "child"），用于自动选择 max_formant
    """
    # 自动选择 max_formant（提升共振峰估计精度）
    if gender_hint == "male":
        max_formant = 5000.0
    else:
        max_formant = 5500.0  # 默认用于女/儿童

    features = extract_f0_and_formants(audio_path, max_formant)
    speaker_type = classify_speaker_with_formants(features)

    print(f"🔊 音频: {audio_path}")
    print("-" * 50)
    print(f"{'特征':<10} | {'值 (Hz)':<10}")
    print("-" * 50)
    print(f"{'F0 (中位)':<10} | {features['f0_median']:<10.1f}" if features['f0_median'] else "F0         | N/A")
    print(f"{'F1':<10} | {features['F1']:<10.1f}" if features['F1'] else "F1         | N/A")
    print(f"{'F2':<10} | {features['F2']:<10.1f}" if features['F2'] else "F2         | N/A")
    print(f"{'F3':<10} | {features['F3']:<10.1f}" if features['F3'] else "F3         | N/A")
    print("-" * 50)
    print(f"🎯 预测说话人类型: {speaker_type}")

    return speaker_type, features


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 分析一段音频
    from configs.config import audiofile
    result, feats = analyze_speaker(audiofile)  # 若知道大致类别，可传 hint