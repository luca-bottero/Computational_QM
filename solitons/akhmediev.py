import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

# Parameters for the Akhmediev Breather
a = 0.25
omega = (1-2*a)**0.5
b = (8*a*(1 - 2*a))**0.5

# Define the Akhmediev Breather function
def akhmediev_breather(x, t):
    numerator = (1 - 4*a)*np.cosh(b*x) + (2*a)**0.5*np.cos(omega*t) + 1j*b*np.sinh(b*x)
    denominator = (2*a)**0.5*np.cos(omega*t) - np.cosh(b*x)
    psi = numerator/denominator * np.exp(1j*x)
    return np.real(psi), np.imag(psi), np.abs(psi)**2

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

# Setup the figure and axis for 3D animation
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

def animate_3d(t):
    ax3d.clear()
    re, img, _ = akhmediev_breather(x, t)
    wavefunction = re + 1j*img    
    magnitude = np.abs(wavefunction)
    phase = np.angle(wavefunction)
    
    ax3d.plot(x, magnitude * np.cos(phase), magnitude * np.sin(phase), label='Wavefunction')
    ax3d.plot(x, np.zeros_like(x), np.zeros_like(x), label='X-axis', color='black')

    ax3d.set_ylim(-3, 3)
    ax3d.set_zlim(-3, 3)

    ax3d.set_title(f'Wavefunction 3D\nTime [a.u.]= {t:.2f}', fontsize=20)
    ax3d.set_xlabel('x [a.u.]', fontsize=18)
    ax3d.set_ylabel('Real part', fontsize=18)
    ax3d.set_zlabel('Imaginary part', fontsize=18)
    ax3d.legend()
    return ax3d

# Create the 3D animation
anim3d = animation.FuncAnimation(fig3d, animate_3d, frames=np.linspace(0, t_max, frames), interval=25, repeat=True)

# Save the 3D animation
anim3d.save('animations/akhmediev_breather_3d.gif', writer='imagemagick', fps=15)

plt.show()
