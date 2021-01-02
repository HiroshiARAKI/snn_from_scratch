"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


class LIF:
    def __init__(self, rest: float = -65, ref: float = 3, th: float = -40, tc: float = 20, peak: float = 20):
        """
        Leaky integrate-and-fire neuron
        :param rest: 静止膜電位 [mV]
        :param ref:  不応期 [ms]
        :param th:   発火閾値 [mV]
        :param tc:   膜時定数 [ms]
        :param peak: ピーク電位 [mV]
        """
        self.rest = rest
        self.ref = ref
        self.th = th
        self.tc = tc
        self.peak = peak

    def calc(self, inputs, weights, time=300, dt=0.5, tci=10):
        """
        膜電位を計算する．
        本来はスパイク時刻(発火時刻)も保持しておいてそれを出力データとする．
        :param inputs:
        :param weights:
        :param time:
        :param dt:
        :param tci:
        :return:
        """
        i = 0           # 初期入力電流
        v = self.rest   # 初期膜電位
        tlast = 0       # 最後に発火した時刻
        monitor = []    # 膜電位の記録

        for t in range(int(time/dt)):
            # 入力電流の計算
            di = ((dt * t) > (tlast + self.ref)) * (-i)
            i += di * dt / tci + np.sum(inputs[:, t] * weights)

            # 膜電位の計算
            dv = ((dt * t) > (tlast + self.ref)) * ((-v + self.rest) + i)
            v += dv * dt / self.tc

            # 発火処理
            tlast = tlast + (dt * t - tlast) * (v >= self.th)  # 発火したら発火時刻を記録
            v = v + (self.peak - v) * (v >= self.th)           # 発火したら膜電位をピークへ

            monitor.append(v)

            v = v + (self.rest - v) * (v >= self.th)   # 発火したら静止膜電位に戻す

        return monitor


if __name__ == '__main__':
    time = 300  # 実験時間 (観測時間)
    dt = 0.5    # 時間分解能

    image = np.random.random((20, 20))  # 適当な画像

    max_freq = 128  # 最大スパイク周波数 [Hz] = 1秒間 (1000 ms) に何本のスパイクを最大生成するか
    freq_img = image * max_freq      # pixels to Hz
    norm_img = 1000. / freq_img      # Hz to ms
    norm_img = norm_img.reshape(-1)  # 2次元だと扱いが面倒なので1次元に

    fires = np.array([
        # 周期が抽出されるのでスパイク発生時間としては累積させなければならない
        # とりあえず，今回は約300msに収まる分だけ抽出
        np.cumsum(np.random.poisson(cell / dt, (int(time / cell + 1)))) * dt
        for cell in norm_img
    ])

    # 発火時刻→スパイク列
    spikes = np.zeros((norm_img.size, int(time/dt)))

    for s, f in zip(spikes, fires):
        f = f[f < time]  # 300msからはみ出た発火時刻は除く
        s[np.array(f / dt, dtype=int)] = 1    # {0,1} spikesへ変換する

    # 重みの初期化 (適当に)
    weights = np.random.normal(0.1, 0.4, norm_img.size)

    # LIFニューロンの生成および膜電位計算
    neuron = LIF()
    v = neuron.calc(spikes, weights, time, dt)

    # 結果の描画
    plt.figure(figsize=(16, 4))

    # 入力画像
    plt.subplot(1, 3, 1)
    plt.title('Original Input Image')
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # 入力データ
    plt.subplot(1, 3, 2)
    t = np.arange(0, time, dt)

    for i, f in enumerate(fires):
        plt.scatter(f, [i for _ in range(len(f))], s=2.0, c='tab:blue')
    plt.xlim(0, time)
    plt.xlabel('time [ms]')
    plt.ylabel('Neuron index')
    plt.title('Poisson Spike Trains (Inputs)')

    # 膜電位
    plt.subplot(1, 3, 3)
    plt.plot(t, v)
    plt.xlabel('time [ms]')
    plt.ylabel('Membrane potential [mV]')
    plt.title('Neuron Internal Status')
    plt.show()
