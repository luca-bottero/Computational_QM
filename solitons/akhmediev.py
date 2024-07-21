import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Akhmediev Breather
a = 0.25
omega = (1-2*a)**0.5
b = (8*a*(1 - 2*a))**0.5

# Define the Akhmediev Breather function
def akhmediev_breather(x, t):
    numerator = ((1 - 4 * a) * np.cosh(b * x) + np.sqrt(2 * a) * np.cos(omega * t) + 1j * b * np.sinh(b * x))
    denominator = (np.sqrt(2 * a) * np.cos(omega * t) - np.cosh(b * x))
    phi = numerator / denominator * np.exp(1j * x)
    return np.real(phi), np.imag(phi), np.abs(phi)**2

# Setup the figure and axis for 2D animation
fig2d, ax2d = plt.subplots()
x = np.linspace(-5, 5, 1000)
t_max = 20
t_0 = -10
frames = 200

def animate(t):
    ax2d.clear()
    t += t_0
    real_part, imag_part, modulus = akhmediev_breather(x, t)
    
    ax2d.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax2d.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax2d.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax2d.set_xlim(-5, 5)
    ax2d.set_ylim(-2.5, 6)

    ax2d.set_title(f'Akhmediev Breather\nTime [a.u.]= {t:.2f}', fontsize=20)
    ax2d.set_xlabel('x [a.u.]', fontsize=18)
    ax2d.set_ylabel('$\psi(x,t)$', fontsize=18)
    ax2d.legend()
    ax2d.grid(True)
    return ax2d

# Create the 2D animation
anim2d = animation.FuncAnimation(fig2d, animate, frames=np.linspace(0, t_max, frames), interval=25, repeat=True)

# Ensure the animations directory exists
if not os.path.exists('animations'):
    os.makedirs('animations')

# Save the 2D animation
anim2d.save('animations/akhmediev_breather.gif', writer='imagemagick', fps=15)