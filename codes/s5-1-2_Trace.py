"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    time = 300
    dt = 0.5

    # Spike Traceを適当に作る
    spikes = np.zeros(int(time/dt))
    # 5本適当にスパイクを立てる
    for _ in range(5):
        spikes[np.random.randint(0, int(time/dt))] = 1

    # Firing Traceを作成
    firing = []
    fire = 0
    tc = 20  # 時定数

    for t in range(int(time/dt)):
        if spikes[t]:  # 発火していれば1を立てる
            fire = 1
        else:  # 発火していなければ時間的減衰
            fire -= fire / tc
        firing.append(fire)

    t = np.arange(0, time, dt)
    plt.subplot(2, 1, 1)
    plt.plot(t, spikes, label='Spike Trace')
    plt.ylabel('Spike Trace')

    plt.subplot(2, 1, 2)
    plt.plot(t, firing)
    plt.ylabel('Firing Trace')
    plt.xlabel('time [ms]')

    plt.show()
