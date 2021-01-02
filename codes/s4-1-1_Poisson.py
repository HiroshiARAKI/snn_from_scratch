"""
ゼロから学ぶスパイキングニューラルネットワーク
- Spiking Neural Networks from Scratch

Copyright (c) 2020 HiroshiARAKI. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    num = 100000  # 抽出回数
    lam = [4, 8, 16, 32, 64]  # lambda

    res = [np.random.poisson(l, num) for l in lam]  # Poisson dist. から抽出

    plt.hist(res, bins=100, histtype='stepfilled', alpha=0.7)

    plt.xlabel('k')
    plt.legend([f'λ={l}' for l in reversed(lam)])
    plt.show()
