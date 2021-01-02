"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


def stdp_ltp(dt, a=1.0, tc=20):
    """ Long-term Potentiation """
    return a * np.exp(-dt / tc)


def stdp_ltd(dt, a=-1.0, tc=20):
    """ Long-term Depression """
    return a * np.exp(dt / tc)


def stdp(dt, pre=-1.0, post=1.0, tc_pre=20, tc_post=20):
    """ STDP rule """
    return stdp_ltd(dt[dt<0], pre, tc_pre), stdp_ltp(dt[dt>=0], post, tc_post)


if __name__ == '__main__':
    # 発火時刻差集合
    dt = np.arange(-50, 50, 0.5)

    # LTD, LTP
    ltd, ltp = stdp(dt)

    plt.plot(dt[dt<0], ltd, label=r'LTD: $\Delta t < 0$')
    plt.plot(dt[dt>=0], ltp, label=r'LTP: $\Delta t \leq 0$')

    plt.xlabel(r'$\Delta t = t_{post} - t_{pre}$')
    plt.ylabel(r'$\Delta w$')
    plt.grid()
    plt.legend()
    plt.show()
