# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 22:18:19 2025

@author: kakoi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def simulate_full_field_shape(total_time_us, fwhm_us, N, Delta_MHz, Gamma_MHz, f0_MHz, sampling_rate):
    # 単位変換
    total_time_s = total_time_us * 1e-6
    fwhm_s = fwhm_us * 1e-6
    Delta_Hz = Delta_MHz * 1e6
    Gamma_Hz = Gamma_MHz * 1e6
    f0_Hz = f0_MHz * 1e6

    # Δf の定義（Δf ≈ Δ - Γ）
    Delta_f = Delta_Hz - Gamma_Hz

    # βの算出（sech関数のFWHM ≈ 1.76 / β）
    beta = 1.76 / fwhm_s

    # 時間軸の設定
    num_points = sampling_rate
    t = np.linspace(-total_time_s / 2, total_time_s / 2, num_points)

    # 振幅部分の計算
    sech = lambda x: 1 / np.cosh(x)
    pi = np.pi
    N = N + 1   #　AFCの櫛の本数　→　パルスのピーク数( = 櫛本数 + 1 )に変換
    amp = sech(beta * t) * np.sin(N * pi * Delta_Hz * t) / np.sin(pi * Delta_Hz * t)
    amp = amp / np.max(amp)

    # 位相項の計算（cos の中身）
    phi = 2 * pi * f0_Hz * t + 2 * pi * (Delta_f / (2 * beta)) * np.log(np.cosh(beta * t))

    # 電場波形（振幅 × cos(位相)）
    field = amp * np.cos(phi)

    # プロット①：時間領域
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e6, field)
    plt.title("Time-domain field f(t) = amplitude × cos(phase)")
    plt.xlabel("Time [μs]")
    plt.ylabel("Field Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # FFTの計算
    dt = t[1] - t[0]
    field_fft = fftshift(fft(field))
    freq = fftshift(fftfreq(num_points, dt)) / 1e6  # 単位 [MHz]
    power_spectrum = np.abs(field_fft)**2

    # プロット②：周波数領域
    plt.figure(figsize=(10, 4))
    plt.plot(freq, power_spectrum)
    plt.title("Power Spectrum of Full Field (|FFT|^2)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power")
    plt.xlim(157,163)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 使用例（数値はあとで自由に変更可能）
simulate_full_field_shape(
    total_time_us=200,         # 波形全体の長さ [μs]
    fwhm_us=30,                # パルスのFWHM [μs]
    N=5,                       # AFC櫛の本数
    Delta_MHz=0.4,             # 櫛の間隔 [MHz]
    Gamma_MHz=0.3,             # 櫛のピーク幅 [MHz]
    f0_MHz=160,                # 中心周波数 [MHz]
    sampling_rate=100000       # サンプリング点数
)
