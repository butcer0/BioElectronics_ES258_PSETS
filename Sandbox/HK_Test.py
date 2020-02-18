# import pylab as plt
# import numpy as np
# from scipy.integrate import odeint
#
#
# class HodgkinHuxley():
#     _Vm0 = -70  # mV
#     _Cm = 200e-9  # in uF
#     _gK = 36.0  # mS
#     _gNA = 120.0  # mS
#     _gL = 0.3  # mS
#     _V_Na = 50  # mV
#     _V_K = -77  # mV
#     _V_L = 50  # mV
#     _t = np.arange(0.0, 5.0, 0.001)
#
#     alphaNf = lambda self, Vm: (0.01 * (10 - Vm)) / (np.exp((10 - Vm) / 10) - 1)
#     alphaMf = lambda self, Vm: (0.1 * (25 - Vm) / (np.exp((25 - Vm) / 10)) - 1)
#     alphaHf = lambda self, Vm: 0.07 * np.exp((-1 * Vm) / 20)
#     betaNf = lambda self, Vm: 0.125 * np.exp((-1 * Vm) / 80)
#     betaMf = lambda self, Vm: 4 * np.exp((-1 * Vm) / 18)
#     betaHf = lambda self, Vm: 1 / (np.exp((30 - Vm) / 10) + 1)
#     INaf = lambda self, Vm, m, h: self._gNA * m ** 3 * h * (Vm - self._gNA)
#     IKf = lambda self, Vm, n: self._gK * n ** 4 * (Vm - self._V_K)
#     ILf = lambda self, Vm: self._gL * (Vm - self._V_L)
#     Issf = lambda self, t: 10 * (t > 2) - 10 * (t > 3) + 35 * (t > 4) - 35 * (t > 5)
#
#     @staticmethod
#     def model(z, t, self):
#         Vm, m, n, h = z
#         dVdt = (self.Issf(t) - self.INaf(Vm, m, h) - self.IKf(Vm, n) - self.ILf(Vm)) / self._Cm
#         dndt = self.alphaNf(Vm) * (1.0 - n) - self.betaNf(Vm) * n
#         dmdt = self.alphaMf(Vm) * (1.0 - m) - self.betaNf(Vm) * m
#         dhdt = self.alphaHf(Vm) * (1.0 - h) - self.betaNf(Vm) * h
#         return dVdt, dndt, dmdt, dhdt
#
#     def __init__(self):
#         z0 = [self._Vm0, 0.05, 0.6, 0.32]
#         s = odeint(self.model, z0, self._t, args=(self,))
#         sol = {'V': s[:, 0], 'N': s[:, 1], 'M': s[:, 2], 'H': s[:, 3]}
#         INa = self.INaf(sol['V'], sol['M'], sol['H'])
#         IK = self.IKf(sol['V'], sol['N'])
#         IL = self.ILf(sol['V'])
#
#         plt.figure()
#
#         plt.subplot(4, 1, 1)
#         plt.title('Hodgkin-Huxley')
#         plt.plot(self._t, sol['V'], 'k')
#         plt.ylabel('V (mV)')
#
#         plt.subplot(4, 1, 2)
#         plt.plot(self._t, INa, 'c', label='$I_{Na}$')
#         plt.plot(self._t, IK, 'y', label='$I_{K}$')
#         plt.plot(self._t, IL, 'm', label='$I_{L}$')
#         plt.ylabel('Current')
#         plt.legend()
#
#         plt.subplot(4, 1, 3)
#         plt.plot(self._t, sol['N'], 'b', label='n')
#         plt.plot(self._t, sol['M'], 'r', label='m')
#         plt.plot(self._t, sol['H'], 'g', label='h')
#         plt.ylabel('Gating Value')
#         plt.legend()
#
#         plt.subplot(4, 1, 4)
#         i_inj_values = [self.Issf(t) for t in self._t]
#         plt.plot(self._t, i_inj_values, 'k')
#         plt.xlabel('t (ms)')
#         plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
#         plt.ylim(-1, 40)
#
#         plt.show()
#
#
# if __name__ == '__main__':
#     runner = HodgkinHuxley()
#
#
# class HodgkinHuxley:
# #     _Cm = 200e-9  # in uF
# #     _gK = 36.0  # mS
# #     _gNA = 120.0  # mS
# #     _gL = 0.3  # mS
# #     _V_Na = 50  # mV
# #     _V_K = -77 # mV
# #     _V_L = -50 # mV
# #
# #     t = np.linspace(0.0, 50.0, 10000)
# #     _I_m = None
# #
# #
# #     def i_ssf(self, t):
# #         if 0.0 < t < 1.0:
# #             return 150.0
# #         elif 10.0 < t < 11.0:
# #             return 50.0
# #         elif 20.0 < t < 21.0:
# #             return 150.0
# #         elif 30.0 < t < 31.0:
# #             return 50.0
# #         elif 40.0 < t < 41.0:
# #             return 150.0
# #         elif 50.0 < t < 51.0:
# #             return 50.0
# #         return 0.0
# #
# #     @staticmethod
# #     def model(z, t, self):
# #         Vm, m, n, h = z
# #         alphaNf = lambda Vm: (0.01 * (10 - Vm)) / (np.exp((10.0 - Vm) / 10.0) - 1)
# #         alphaMf = lambda Vm: (0.1 * (25 - Vm) / (np.exp((25.0 - Vm) / 10.0)) - 1)
# #         alphaHf = lambda Vm: 0.07 * np.exp((-1 * Vm) / 20.0)
# #         betaNf = lambda Vm: 0.125 * np.exp((-1 * Vm) / 80.0)
# #         betaMf = lambda Vm: 4.0 * np.exp((-1.0 * Vm) / 18.0)
# #         betaHf = lambda Vm: 1.0 / (np.exp((30.0 - Vm) / 10.0) + 1)
# #         dVdt = (self.i_ssf(t) - (self._gK * n**4 * (Vm - self._V_K))
# #                 - (self._gNA * m**3 * h * (Vm- self._V_Na))
# #                 + (self._gL * (Vm - self._V_L))) / self._Cm
# #         dndt = alphaNf(Vm) * (1.0 - n) - betaNf(Vm) * n
# #         dmdt = alphaMf(Vm) * (1.0 - m) - betaMf(Vm) * m
# #         dhdt = alphaHf(Vm) * (1.0 - h) - betaHf(Vm) * h
# #         return dVdt, dndt, dmdt, dhdt
# #
# #     def __init__(self):
# #         alphaNf = lambda Vm: (0.01 * (10 - Vm)) / (np.exp((10 - Vm) / 10) - 1)
# #         alphaMf = lambda Vm: (0.1 * (25 - Vm) / (np.exp((25 - Vm) / 10)) - 1)
# #         alphaHf = lambda Vm: 0.07 * np.exp((-1 * Vm) / 20)
# #         betaNf = lambda Vm: 0.125 * np.exp((-1 * Vm) / 80)
# #         betaMf = lambda Vm: 4 * np.exp((-1 * Vm) / 18)
# #         betaHf = lambda Vm: 1 / (np.exp((30 - Vm) / 10) + 1)
# #         z0 = np.array([0.0, (1 + alphaNf(0.0)/betaNf(0.0)), (1 + alphaMf(0.0)/betaMf(0.0)), (1 + alphaHf(0.0)/betaHf(0.0))])
# #         s = odeint(self.model, z0, self.t, args=(self,))
# #         sol = {'V': s[:, 0], 'N': s[:, 1], 'M': s[:, 2], 'H': s[:, 3]}
# #         plt.figure()
# #
# #         plt.subplot(4, 1, 1)
# #         plt.title('Hodgkin-Huxley')
# #         plt.plot(self.t, sol['V'], 'k')
# #         plt.ylabel('V (mV)')
# #
# #         plt.subplot(4, 1, 3)
# #         plt.plot(self.t, sol['N'], 'b', label='n')
# #         plt.plot(self.t, sol['M'], 'r', label='m')
# #         plt.plot(self.t, sol['H'], 'g', label='h')
# #         plt.ylabel('Gating Value')
# #         plt.legend()
# #
# #
# # if __name__ == '__main__':
# #     runner = HodgkinHuxley()