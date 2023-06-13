import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


def odes(x, t):
    g = 9.81
    m1 = 1
    m2 = 2
    l1 = 1
    l2 = 1

    theta1 = x[0]
    omega1 = x[1]
    theta2 = x[2]
    omega2 = x[3]

    deltatheta = theta1 - theta2

    theta1_d = omega1

    omega1_d = ((m2*l1*(omega1 ** 2)*np.sin(2*deltatheta)) + (2*m2*l2*(omega2 ** 2) * np.sin(deltatheta)) + (2 *
                g*m2*np.cos(theta2)*np.sin(deltatheta)) + (2*g*m1*np.sin(theta1))) / -2*l1 * (m1+(m2*(np.sin(deltatheta)**2)))

    theta2_d = omega2

    omega2_d = ((m2*l2*(omega2 ** 2)*np.sin(2*deltatheta)) +
                ((2*m1*l1*(omega1**2)*np.sin(deltatheta))) +
                ((2*m2*l1*(omega1**2)*np.sin(deltatheta))) +
                (2*g*(m1 + m2)*np.cos(theta1)*np.sin(deltatheta))) / (2*l2*(m1 + (m2*(np.sin(deltatheta) ** 2))))

    return [theta1_d, omega1_d, theta2_d, omega2_d]


x0 = [3, 0, 0, 0]

t = np.linspace(0, 40, 1001)

x = odeint(odes, x0, t)

theta1 = x[:, 0]
omega1 = x[:, 1]
theta2 = x[:, 2]
omega2 = x[:, 3]

plt.plot(t, theta1)
plt.show()
