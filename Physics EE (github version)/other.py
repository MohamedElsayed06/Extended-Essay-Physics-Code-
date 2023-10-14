import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

_t, _g = smp.symbols('t g')
_m1, _m2 = smp.symbols('m1 m2')
_L1, _L2 = smp.symbols('L1, L2')

_the1, _the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)

_the1 = _the1(_t)
_the2 = _the2(_t)

_the1_d = smp.diff(_the1, _t)
_the2_d = smp.diff(_the2, _t)
_the1_dd = smp.diff(_the1_d, _t)
_the2_dd = smp.diff(_the2_d, _t)

_x1 = _L1*smp.sin(_the1)
_y1 = -_L1*smp.cos(_the1)
_x2 = _L1*smp.sin(_the1)+_L2*smp.sin(_the2)
_y2 = -_L1*smp.cos(_the1)-_L2*smp.cos(_the2)

# Kinetic
_T1 = 1/2 * _m1 * (smp.diff(_x1, _t)**2 + smp.diff(_y1, _t)**2)
_T2 = 1/2 * _m2 * (smp.diff(_x2, _t)**2 + smp.diff(_y2, _t)**2)
_T = _T1+_T2
# Potential
_V1 = _m1*_g*_y1
_V2 = _m2*_g*_y2
_V = _V1 + _V2
# Lagrangian
_L = _T-_V

_LE1 = smp.diff(_L, _the1) - smp.diff(smp.diff(_L, _the1_d), _t).simplify()
_LE2 = smp.diff(_L, _the2) - smp.diff(smp.diff(_L, _the2_d), _t).simplify()

_sols = smp.solve([_LE1, _LE2], (_the1_dd, _the2_dd),
                  simplify=False, rational=False)

_dz1dt_f = smp.lambdify(
    (_t, _g, _m1, _m2, _L1, _L2, _the1, _the2, _the1_d, _the2_d), _sols[_the1_dd])
_dz2dt_f = smp.lambdify(
    (_t, _g, _m1, _m2, _L1, _L2, _the1, _the2, _the1_d, _the2_d), _sols[_the2_dd])
_d_the1dt_f = smp.lambdify(_the1_d, _the1_d)
_d_the2dt_f = smp.lambdify(_the2_d, _the2_d)


def dSdt(A, _t, _g, _m1, _m2, _L1, _L2):
    _the1, _z1, _the2, _z2 = A
    return [
        _d_the1dt_f(_z1),
        _dz1dt_f(_t, _g, _m1, _m2, _L1, _L2, _the1, _the2, _z1, _z2),
        _d_the2dt_f(_z2),
        _dz2dt_f(_t, _g, _m1, _m2, _L1, _L2, _the1, _the2, _z1, _z2),
    ]


_t = np.linspace(0, 40, 1000)
_g = 9.81
_m1 = 1
_m2 = 1
_L1 = 1
_L2 = 1
# initial condition    _the1, omega1, _the2, omega2
_ans = odeint(dSdt, y0=[(np.pi)-0.3, 0, 0, 0],
              t=_t, args=(_g, _m1, _m2, _L1, _L2))

_the1 = _ans.T[0]
_the2 = _ans.T[2]

#result = [1]
#for i in range(len(_the2)):
#    if _the2[i] > 0:
#        result.append((_the2[i] % np.pi) - np.pi)
#    if _the2[i] < 0:
#        result.append(np.pi - (_the2[i] % -(np.pi)))

#i=0 
#for i in _the2:
#    if i > 999:
#        break
#    i = int(i)
#    if _the2[i] < 0:
#        _the2[i] = _the2[i] % (-(np.pi))
#    if _the2[i] > 0:
#        _the2[i] = _the2[i] % (np.pi)
#    i = i+1

i=0
for i in _the2:

    i = int(i)
    _the2[i] % (np.pi)*2
    if _the2[i] > np.pi:
        _the2[i] -= 2 * np.pi
    elif _the2[i] < -np.pi:
        _the2[i] += 2 * np.pi
    i = i+1

print(_the2)

plt.plot(_t, _the2)
plt.show()
