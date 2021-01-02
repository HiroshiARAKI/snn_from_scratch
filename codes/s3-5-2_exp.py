"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


def single_exp(t):
    return np.exp(-t / 20)


if __name__ == '__main__':
    # 観測時間とタイムステップ
    time = 100
    dt = 0.5

    h = 1  # 初期値
    result = []

    # 微分法定式を解く
    for t in range(int(time / dt)):
        result.append(h)

        dh = -h * dt   # dH = -H · dt
        h += dh / 20   # h ← h + dh / τ

    t = np.arange(0, time, dt)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, single_exp(t))
    plt.title(r'$H(t) = \exp(-t / \tau)$')
    plt.xlabel('time [ms]')

    plt.subplot(1, 2, 2)
    plt.plot(t, result)
    plt.title(r'$\tau dH(t) / dt = -H(t)$')
    plt.xlabel('time [ms]')

    plt.show()
