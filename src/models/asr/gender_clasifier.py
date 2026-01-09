import os
import numpy as np
import parselmouth
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. 特征提取函数
# ----------------------------
def extract_acoustic_features(wav_path, time_step=0.01):
    """
    从音频中提取鲁棒的声学统计特征（12维）
    """
    try:
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=50, pitch_ceiling=600)
        intensity = snd.to_intensity(time_step=time_step, minimum_pitch=50)
        formant = snd.to_formant_burg(
            time_step=time_step,
            max_number_of_formants=5,
            maximum_formant=5500,
            window_length=0.025
        )

        times = intensity.xs()
        int_vals = intensity.values[0]
        pitch_vals = np.array([pitch.get_value_at_time(t) for t in times])
        f1_vals = np.array([formant.get_value_at_time(1, t) for t in times])
        f2_vals = np.array([formant.get_value_at_time(2, t) for t in times])
        f3_vals = np.array([formant.get_value_at_time(3, t) for t in times])

        # 有效帧：pitch > 0 且共振峰 > 0
        valid = (pitch_vals > 0) & (f1_vals > 0) & (f2_vals > 0) & (f3_vals > 0)
        if np.sum(valid) == 0:
            return None  # 全静音或无效

        p = pitch_vals[valid]
        f1 = f1_vals[valid]
        f2 = f2_vals[valid]
        f3 = f3_vals[valid]
        i = int_vals[:len(valid)][valid]  # 对齐长度

        # 构造12维特征向量
        features = [
            np.mean(p), np.std(p), np.ptp(p),      # F0: 均值、标准差、极差
            np.median(p),
            np.mean(f1), np.mean(f2), np.mean(f3),
            np.mean(i),
            np.mean(f2 / (f1 + 1e-6)),             # F2/F1 比值（声道长度指标）
            np.mean(f3 / (f2 + 1e-6)),
            np.sum(valid) / len(pitch_vals),       # 有声帧比例
            np.percentile(p, 90) - np.percentile(p, 10)  # F0 动态范围
        ]
        return np.array(features)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


# ----------------------------
# 2. 数据加载与准备
# ----------------------------
def load_dataset(data_dir):
    """
    假设目录结构：
    data_dir/
        ├── male/
        │   ├── 001.wav
        │   └── ...
        ├── female/
        │   ├── 001.wav
        │   └── ...
        └── child/
            ├── 001.wav
            └── ...

    返回: X (n_samples, 12), y (n_samples,)
    """
    label_map = {'male': 0, 'female': 1, 'child': 2}
    X, y = [], []

    for label_name, label_id in label_map.items():
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                path = os.path.join(folder, file)
                feat = extract_acoustic_features(path)
                if feat is not None:
                    X.append(feat)
                    y.append(label_id)

    return np.array(X), np.array(y)


# ----------------------------
# 3. 训练模型
# ----------------------------
def train_model(data_dir, model_save_path="gender_rf_model.joblib"):
    X, y = load_dataset(data_dir)
    print(f"Loaded {len(X)} samples.")

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练随机森林
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # 多核加速（CPU友好）
    )
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    labels = ['male', 'female', 'child']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(clf, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    return clf


# ----------------------------
# 4. 预测单条音频
# ----------------------------
def predict_gender(wav_path, model_path="gender_rf_model.joblib"):
    clf = joblib.load(model_path)
    feat = extract_acoustic_features(wav_path)
    if feat is None:
        return "unknown"
    proba = clf.predict_proba([feat])[0]
    pred = clf.predict([feat])[0]
    label_map = {0: 'male', 1: 'female', 2: 'child'}
    confidence = np.max(proba)
    return label_map[pred], confidence


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--predict", type=str, help="Predict gender of a single .wav file")
    parser.add_argument("--model", type=str, default="gender_rf_model.joblib", help="Model path")

    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir, args.model)
    elif args.predict:
        label, conf = predict_gender(args.predict, args.model)
        print(f"Prediction: {label} (confidence: {conf:.2f})")
    else:
        print("Please specify --train or --predict <wav_file>")

