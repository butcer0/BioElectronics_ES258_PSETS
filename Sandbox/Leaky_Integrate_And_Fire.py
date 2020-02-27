import scipy as sp
import pylab as plt
import numpy as np
from scipy.integrate import odeint


class IntegrateAndFire():
    """Standard Integrate-and-Fire Model"""

    _R_M = 50e6  # MO
    _C_M = 200e-12  # pF
    _V_E = -70e-3  # mV
    _V_TH = -40e-3  # mV
    _V_RESET = -80e-3  # mV
    _T_TOT = 1e-1/3
    _DELTA_T = 10e-6  # uS
    _T_REF = 3e-3  # mS

    t = np.arange(0.0, _T_TOT, _DELTA_T)
    _I_m = None
    i = np.arange(0.0, 10e-9, 100e-12)
    iff = lambda self, t: (t/100e-12)

    t_2 = np.linspace(0, 10e-9, 1000)
    VMF = lambda self, v: (v % (self._V_TH - self._V_RESET)) * (self._V_TH - self._V_RESET) + self._V_RESET

    def __init__(self, i_m=None):
        self._I_m = i_m

    @staticmethod
    def voltage_Model(vm, t, self):
        dVdt = (-1 * (1 / self._R_M) * (vm - self._V_E) + self._I_m) / self._C_M
        return dVdt

    getVoltageModel = lambda self: odeint(self.voltage_Model, self._V_E, self.t, args=(self,))

    getVoltageModelThresholded = lambda self: list(map(lambda v_m: self.VMF(v_m), self.getVoltageModel()))

    @staticmethod
    def voltage_Model_Sweep(vm, t, self):
        dVdt = (-1 * (1 / self._R_M) * (vm - self._V_E) + t / self._C_M)
        return dVdt

    getVoltageModel_Sweep = lambda self: odeint(self.voltage_Model_Sweep, self._V_E, self.t_2, args=(self,))

    getVoltageModelThresholded_Sweep = lambda self: list(map(lambda v_m: self.VMF(v_m), self.getVoltageModel_Sweep()))

    @staticmethod
    def frequency_Model(vm, i, self):
        i_gt_i_th = lambda i: (self._T_REF - self._R_M * self._C_M * np.log(1 - (self._V_TH / (i * self._R_M)))) ** -1
        i_f = 0 if i <= ((self._V_TH - self._V_E) / self._R_M) else i_gt_i_th(i)
        return i_f

    getFrequencyModel = lambda self: odeint(self.frequency_Model, self._V_E, self.i, args=(self,))


if __name__ == '__main__':
    ii = np.arange(0.0, 10.0e-9, 100e-12)
    i_s = np.zeros(len(ii) * 2)
    for index in range(len(ii)):
        i_s[index * 2] = ii[index]

    plot_100pA, plot_1nA, plot_10nA = np.array(
        list(map(lambda i_m: IntegrateAndFire(i_m).getVoltageModelThresholded(), [100e-12, 1e-9, 10e-9])))

    plot_sweep_I = IntegrateAndFire(IntegrateAndFire.i).getVoltageModelThresholded_Sweep()

    print(i_s)
    axis_color = 'lightgoldenrodyellow'
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(IntegrateAndFire.t, plot_100pA, 'r')
    plt.ylabel('V (mV)')
    plt.xlabel('t (ms)')
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(IntegrateAndFire.t, plot_1nA, 'g')
    plt.ylabel('V (mV)')
    plt.xlabel('t (ms)')
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(IntegrateAndFire.t, plot_10nA, 'b')
    plt.ylabel('${V} \;(mV)$')
    plt.xlabel('t (ms)')
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(IntegrateAndFire.t_2, plot_sweep_I, 'b')
    plt.ylabel('${V} \;(mV)$')
    plt.xlabel('t (ms)')
    plt.grid()

    # plt.subplot(3, 1, 2)
    # plt.plot(IntegrateAndFire.t, plot_voltage_m, 'r')
    # plt.ylabel('V (mV)')
    # plt.grid()

    # plot_frequency = IntegrateAndFire(0.1).getFrequencyModel()
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(IntegrateAndFire.i, plot_frequency, 'g')
    # plt.ylabel('f(I)')
    # plt.grid()
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(IntegrateAndFire.t, i_s, 'k')
    # plt.xlabel('t (ms)')
    # plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    # plt.subplot(3, 1, 3)
    # plt.plot(IntegrateAndFire.t, i_s, 'k')
    # plt.xlabel('t (ms)')
    # plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
    plt.show()
