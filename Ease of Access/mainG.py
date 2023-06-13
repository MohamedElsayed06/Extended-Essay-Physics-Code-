import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1, L2')

the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)

the1 = the1(t)
the2 = the2(t)

the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

x1 = L1*smp.sin(the1)
y1 = -L1*smp.cos(the1)
x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)

# Kinetic
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1+T2
# Potential
V1 = m1*g*y1
V2 = m2*g*y2
V = V1 + V2
# Lagrangian
L = T-V

LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

sols = smp.solve([LE1, LE2], (the1_dd, the2_dd),
                 simplify=False, rational=False)

dz1dt_f = smp.lambdify(
    (t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify(
    (t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)


def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
    ]


# ------------------------------------------------------------------------------
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


def _dSdt(A, _t, _g, _m1, _m2, _L1, _L2):
    _the1, _z1, _the2, _z2 = A
    return [
        _d_the1dt_f(_z1),
        _dz1dt_f(_t, _g, _m1, _m2, _L1, _L2, _the1, _the2, _z1, _z2),
        _d_the2dt_f(_z2),
        _dz2dt_f(_t, _g, _m1, _m2, _L1, _L2, _the1, _the2, _z1, _z2),
    ]


# ----------------------------------------------------------------------------

_t = np.linspace(0, 40, 1001)
_g = 9.81
_m1 = 1
_m2 = 1
_L1 = 1
_L2 = 1

t = np.linspace(0, 40, 1001)
g = 9.81
m1 = 1
m2 = 1
L1 = 1
L2 = 1

# initial condition    the1, omega1, the2, omega2
ans = odeint(dSdt, y0=[0, 0, 0, 0], t=t, args=(g, m1, m2, L1, L2))
_ans = odeint(_dSdt, y0=[3.2, 0, 3, 0],
              t=_t, args=(_g, _m1, _m2, _L1, _L2))

the1 = ans.T[0]
the2 = ans.T[2]

_the1 = _ans.T[0]
_the2 = _ans.T[2]

if the1[0] > np.pi:
    i = 0
    for j in the1:
        if i > 999:
            break

        the1[i] = 2*np.pi - j
        i = i+1

if _the1[0] > np.pi:
    i = 0
    for j in _the1:
        if i > 999:
            break
        _the1[i] = 2*np.pi - j
        i = i+1


plt.plot(_t, _the1)
plt.plot(t, the1)

plt.show()
