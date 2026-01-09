import numpy as np
import matplotlib.pyplot as plt

def erb(f):
    return 24.7 * (4.37 * f / 1000 + 1)

def gammachirp_ir(cf, sr, order=4, c=-2.5, dur=0.2):
    """生成 Gammachirp 冲激响应"""
    erb_bw = erb(cf)
    b = erb_bw / 2.0
    t = np.arange(1, int(dur * sr)) / sr
    env = (t ** (order - 1)) * np.exp(-2 * np.pi * b * t)
    phase = 2 * np.pi * cf * t + c * np.log(t)
    ir = env * np.cos(phase)
    return np.concatenate(([0], ir)), t[0]

def compressive_gammatone(signal, cf, sr):
    """近似压缩 Gammatone 滤波"""
    ir, dt = gammachirp_ir(cf, sr)
    # 线性滤波
    from scipy.signal import fftconvolve
    linear_out = fftconvolve(signal, ir, mode='same')
    # 压缩非线性（幂律）
    compressed = np.sign(linear_out) * np.abs(linear_out) ** (1/3)
    return compressed

# 测试
sr = 16000
t = np.arange(0, 0.1, 1/sr)
# 小声信号
sig_soft = 0.01 * np.sin(2*np.pi*1000*t)
# 大声信号
sig_loud = 0.5 * np.sin(2*np.pi*1000*t)

out_soft = compressive_gammatone(sig_soft, 1000, sr)
out_loud = compressive_gammatone(sig_loud, 1000, sr)

# 比较包络
plt.plot(t*1000, np.abs(out_soft), label='Soft input (0.01)')
plt.plot(t*1000, np.abs(out_loud)/10, label='Loud input (0.5) / 10')  # 归一化对比
plt.xlabel('Time (ms)')
plt.ylabel('Compressed envelope')
plt.title('Compression: Loud signal not 50x larger!')
plt.legend()
plt.grid(True)
plt.show()