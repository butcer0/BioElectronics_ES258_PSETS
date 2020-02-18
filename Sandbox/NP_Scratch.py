import pylab as plt
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

_R_M = 50e6  # MO
_C_M = 200e-12  # pF
_V_E = -70e-3  # mV
_V_TH = -40e-3  # mV
_V_RESET = -80e-3  # mV
_T_TOT = 1e-1
_DELTA_T = 10e-6  # uS
_T_REF = 3e-3  # mS
_NUM_STEPS = 10000
_I_Max = 10e-9
_I_Step = 100e-12
_DELTA_I = _I_Max - _I_Step

_i = np.arange(0.0, _I_Max, _I_Step)
t = np.arange(0.0, _T_TOT, _DELTA_T)
# print(list(map(lambda t: _i[t/_DELTA_T], t)))
print('% _i[t/delta_t]', (t[5] % _DELTA_T) * _DELTA_T)
curIndex = int(t[5] / _DELTA_T)
print('/ _i[t/delta_t]', curIndex)
print('cur I:', _i[int(t[5] / _DELTA_T)])


# print(_i)
# print(_i[2])
