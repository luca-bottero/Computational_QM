import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Kuznetsov-Ma Breather
s = 1.
b = 2.

assert b >= s, "b >= s must be true"

# Define the Kuznetsov-Ma Breather function
def kuznetsov_ma_breather(x, t):
    xi = 2*b*(b**0.5 - s**0.5)

    numerator = 2*(b**2 - s**2)*np.cos(xi*t) + 1j*xi*np.sin(xi*t)
    denominator = b*np.cosh(x*xi/b) - s*np.cos(xi*t)
    psi = (s - numerator/denominator) * np.exp(1j*s**2*t) 

    return np.real(psi), np.imag(psi), np.abs(psi)**2

# Setup the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)
t_max = 20
t_0 = -10
frames = 200

def animate(t):
    ax.clear()
    t += t_0
    real_part, imag_part, modulus = kuznetsov_ma_breather(x, t)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-5.5, 20)

    ax.set_title(f'Kuznetsov-Ma Breather\nTime [a.u.]= {t:.2f}', fontsize=20)
    ax.set_xlabel('x [a.u.]', fontsize=18)
    ax.set_ylabel('$\psi(x,t)$', fontsize=18)
    ax.legend()
    ax.grid(True)
    return ax

# Create the animation
anim = animation.FuncAnimation(fig, animate, frames=np.linspace(0, t_max, frames), interval=25, repeat=True)

# Ensure the animations directory exists
if not os.path.exists('animations'):
    os.makedirs('animations')

# Save the animation
anim.save('animations/kuznetsov_ma_breather.gif', writer='imagemagick', fps=15)

plt.show()
