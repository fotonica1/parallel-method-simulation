# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 23:17:07 2025

@author: kakoi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def simulate_aom_double_pass_signal(total_time_us, fwhm_us, N, Delta_MHz, Gamma_MHz, f0_MHz, sampling_rate):
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
    amp = amp / np.max(amp)  # 規格化

    # 位相項の計算（cos の中身）
    phi = 2 * pi * f0_Hz * t + 2 * pi * (Delta_f / (2 * beta)) * np.log(np.cosh(beta * t))

    # AOMダブルパス用信号（振幅の平方根 × cos(位相の1/2)）
    # 安全に振幅の符号に応じてg(t)を定義  
    g = np.where(
        amp >= 0,
        np.sqrt(abs(amp)) * np.cos(phi / 2),
        np.sqrt(abs(amp)) * np.cos(phi / 2 + np.pi / 2)
    )
    
    # gの2乗の直流電源成分を抜いた項
    gg = np.where(
        amp >= 0,
        amp * np.cos(phi) / 2,
        amp * np.cos(phi) / 2
    )

    # プロット①：AOM入力信号 g(t)
    plt.figure(figsize=(8, 4))
    plt.plot(t * 1e6, g)
    plt.title("AOM Input Signal g(t)")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # プロット②：AOM信号 g(t)^2（二乗）
    g_squared = g ** 2
    plt.figure(figsize=(8, 4))
    plt.plot(t * 1e6, g_squared)
    plt.title("Squared AOM Signal g(t)^2 (Expected f(t))")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude Squared")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # プロット③： g(t)^2（二乗）の直流電源成分を抜いた項
    plt.figure(figsize=(8, 4))
    plt.plot(t * 1e6, gg)
    plt.title("Squared AOM Signal g(t)^2 (Expected f(t))")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude Squared")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # FFT の準備
    dt = t[1] - t[0]
    freq = fftshift(fftfreq(num_points, dt)) / 1e6  # 単位 MHz

    # FFT①：g(t) のスペクトル
    g_fft = fftshift(fft(g))
    power_g = np.abs(g_fft)**2
    plt.figure(figsize=(8, 4))
    plt.plot(freq, power_g)
    plt.title("Power Spectrum of g(t)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power")
    plt.xlim(77,83)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # FFT②：g(t)^2 のスペクトル（f(t)に対応）
    g2_fft = fftshift(fft(g_squared))
    power_g2 = np.abs(g2_fft)**2
    plt.figure(figsize=(8, 4))
    plt.plot(freq, power_g2)
    plt.title("Power Spectrum of g(t)^2 (Expected f(t))")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power")
    #plt.xlim(156,164)
    plt.xlim(0,180)
    plt.ylim(-1e8,1.6e9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # FFT②：g(t)^2の直流電源成分を抜いた項 のスペクトル（f(t)に対応）
    gg_fft = fftshift(fft(gg))
    power_gg = np.abs(gg_fft)**2
    plt.figure(figsize=(8, 4))
    plt.plot(freq, power_gg)
    plt.title("Power Spectrum of g(t)^2 (Expected f(t))")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power")
    plt.xlim(156,164)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

simulate_aom_double_pass_signal(
    total_time_us=200,         # 波形全体の長さ [μs]
    fwhm_us=30,                # パルスのFWHM [μs]
    N=5,                       # AFC櫛の本数
    Delta_MHz=0.4,             # 櫛の間隔 [MHz]
    Gamma_MHz=0.3,             # 櫛のピーク幅 [MHz]
    f0_MHz=160,                # 中心周波数 [MHz]
    sampling_rate=10000000       # サンプリング点数
)
