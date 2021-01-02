"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


def lif(currents, time: int, dt: float = 1.0, rest=-65, th=-40, ref=3, tc_decay=100):
    """ simple LIF neuron """
    time = int(time / dt)

    # initialize
    tlast = 0  # 最後に発火した時刻
    vpeak = 20  # 膜電位のピーク(最大値)
    spikes = np.zeros(time)
    v = rest  # 静止膜電位

    monitor = []  # monitor voltage

    # Core of LIF
    for t in range(time):
        dv = ((dt * t) > (tlast + ref)) * (-v + rest + currents[t]) / tc_decay  # 微小膜電位増加量
        v = v + dt * dv  # 膜電位を計算

        tlast = tlast + (dt * t - tlast) * (v >= th)  # 発火したら発火時刻を記録
        v = v + (vpeak - v) * (v >= th)  # 発火したら膜電位をピークへ

        monitor.append(v)

        spikes[t] = (v >= th) * 1  # スパイクをセット

        v = v + (rest - v) * (v >= th)  # 静止膜電位に戻す

    return spikes, monitor


if __name__ == '__main__':
    duration = 500  # ms
    dt = 0.5  # time step

    time = int(duration / dt)

    # Input data
    # 適当なサインカーブの足し合わせで代用
    input_data_1 = 10 * np.sin(0.1 * np.arange(0, duration, dt)) + 50
    input_data_2 = -10 * np.cos(0.05 * np.arange(0, duration, dt)) - 10

    input_data = input_data_1 + input_data_2

    spikes, voltage = lif(input_data, duration, dt)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.ylabel('Input Current')
    plt.plot(np.arange(0, duration, dt), input_data)

    plt.subplot(2, 1, 2)
    plt.ylabel('Membrane Voltage')
    plt.xlabel('time [ms]')
    plt.plot(np.arange(0, duration, dt), voltage)

    plt.show()
