import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Akhmediev Breather
#a = 0.5*(1 - omega**2)
a = 0.25
omega = (1-2*a)**0.5
b = (8*a*(1 - 2*a))**0.5

# Define the Akhmediev Breather function
def akhmediev_breather(x, t):

    numerator = (1 - 4*a)*np.cosh(b*x) + (2*a)**0.5*np.cos(omega*t) + 1j*b*np.sinh(b*x)
    denominator = (2*a)**0.5*np.cos(omega*t) - np.cosh(b*x)
    psi = numerator/denominator * np.exp(1j*x)

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
    real_part, imag_part, modulus = akhmediev_breather(x, t)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-2.5, 6)

    ax.set_title(f'Akhmediev Breather\nTime [a.u.]= {t:.2f}', fontsize=20)
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
anim.save('animations/akhmediev_breather.gif', writer='imagemagick', fps=15)

plt.show()
