"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


class HodgkinHuxley:
    def __init__(self, time, dt, rest=-65., Cm=1.0, gNa=120., gK=36., gl=0.3, ENa=50., EK=-77., El=-54.387):
        """
        Initialize Neuron parameters

        :param time: experimental time
        :param dt:   time step
        :param rest: resting potential
        :param Cm:   membrane capacity
        :param gNa:  Na+ channel conductance
        :param gK:   K+ channel conductance
        :param gl:   other (Cl) channel conductance
        :param ENa:  Na+ equilibrium potential
        :param EK:   K+ equilibrium potential
        :param El:   other (Cl) equilibrium potentials
        """
        self.time = time
        self.dt = dt
        self.rest = rest
        self.Cm = Cm
        self.gNa = gNa
        self.gK = gK
        self.gl = gl
        self.ENa = ENa
        self.EK = EK
        self.El = El

    def calc(self, i):
        """ compute membrane potential """

        # initialize parameters
        v = self.rest
        n = 0.32
        m = 0.05
        h = 0.6

        v_monitor = []
        n_monitor = []
        m_monitor = []
        h_monitor = []

        time = int(self.time / self.dt)

        # update time
        for t in range(time):
            # calc channel gating kinetics
            n += self.dn(v, n)
            m += self.dm(v, m)
            h += self.dh(v, h)

            # calc tiny membrane potential
            dv = (i[t] -
                  self.gK * n**4 * (v - self.EK) -         # K+ current
                  self.gNa * m**3 * h * (v - self.ENa) -   # Na+ current
                  self.gl * (v - self.El)) / self.Cm       # other current

            # calc new membrane potential
            v += dv * self.dt

            # record
            v_monitor.append(v)
            n_monitor.append(n)
            m_monitor.append(m)
            h_monitor.append(h)

        return v_monitor, n_monitor, m_monitor, h_monitor

    def dn(self, v, n):
        return (self.alpha_n(v) * (1 - n) - self.beta_n(v) * n) * self.dt

    def dm(self, v, m):
        return (self.alpha_m(v) * (1 - m) - self.beta_m(v) * m) * self.dt

    def dh(self, v, h):
        return (self.alpha_h(v) * (1 - h) - self.beta_h(v) * h) * self.dt

    def alpha_n(self, v):
        return 0.01 * (10 - (v - self.rest)) / (np.exp((10 - (v - self.rest))/10) - 1)

    def alpha_m(self, v):
        return 0.1 * (25 - (v - self.rest)) / (np.exp((25 - (v - self.rest))/10) - 1)

    def alpha_h(self, v):
        return 0.07 * np.exp(-(v - self.rest) / 20)

    def beta_n(self, v):
        return 0.125 * np.exp(-(v - self.rest) / 80)

    def beta_m(self, v):
        return 4 * np.exp(-(v - self.rest) / 18)

    def beta_h(self, v):
        return 1 / (np.exp((30 - (v - self.rest))/10) + 1)


if __name__ == '__main__':
    # init experimental time and time-step
    time = 300  # 実験時間 (観測時間)
    dt = 2**-4  # 時間分解能 (HHモデルは結構小さめでないと上手く計算できない)

    # Hodgkin-Huxley Neuron
    neuron = HodgkinHuxley(time, dt)

    # 入力データ (面倒臭いので適当な矩形波とノイズを合成して作った)
    input_data = np.sin(0.5 * np.arange(0, time, dt))
    input_data = np.where(input_data > 0, 20, 0) + 10 * np.random.rand(int(time/dt))
    input_data_2 = np.cos(0.4 * np.arange(0, time, dt) + 0.5)
    input_data_2 = np.where(input_data_2 > 0, 10, 0)
    input_data += input_data_2

    # 膜電位などを計算
    v, m, n, h = neuron.calc(input_data)

    # plot
    plt.figure(figsize=(12, 6))
    x = np.arange(0, time, dt)
    plt.subplot(3, 1, 1)
    plt.plot(x, input_data)
    plt.ylabel('I [μA/cm2]')

    plt.subplot(3, 1, 2)
    plt.plot(x, v)
    plt.ylabel('V [mV]')

    plt.subplot(3, 1, 3)
    plt.plot(x, n, label='n')
    plt.plot(x, m, label='m')
    plt.plot(x, h, label='h')
    plt.xlabel('time [ms]')
    plt.ylabel('Conductance param')
    plt.legend()

    plt.show()
