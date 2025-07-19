import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def simulate_amplitude_shape(total_time_us, fwhm_us, N, Delta_MHz):
    # 単位変換
    total_time_s = total_time_us * 1e-6
    fwhm_s = fwhm_us * 1e-6
    Delta_Hz = Delta_MHz * 1e6

    # βの算出（sech関数のFWHM ≈ 1.76 / β）
    beta = 1.76 / fwhm_s

    # 時間軸の設定
    num_points = 5000
    t = np.linspace(-total_time_s / 2, total_time_s / 2, num_points)

    # 振幅部分の計算
    sech = lambda x: 1 / np.cosh(x)
    pi = np.pi
    amp = sech(beta * t) * np.sin(N * pi * Delta_Hz * t) / np.sin(pi * Delta_Hz * t)

    # プロット①：時間領域
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e6, amp)
    plt.title("Time-domain amplitude shape (sech × sin comb)")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # FFTの計算
    dt = t[1] - t[0]
    amp_fft = fftshift(fft(amp))
    freq = fftshift(fftfreq(num_points, dt)) / 1e6  # 単位 [MHz]
    power_spectrum = np.abs(amp_fft)**2

    # プロット②：周波数領域
    plt.figure(figsize=(10, 4))
    plt.plot(freq, power_spectrum)
    plt.title("Power Spectrum of Amplitude (|FFT|^2)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power")
    plt.xlim(-3,3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 使用例（数値は自由に変更してね！）
simulate_amplitude_shape(
    total_time_us = 200,   # 波形全体の長さ [μs]
    fwhm_us = 30,          # パルスのFWHM [μs]
    N = 5,                # 櫛の本数
    Delta_MHz = 0.4         # 櫛の間隔 [MHz]
)
