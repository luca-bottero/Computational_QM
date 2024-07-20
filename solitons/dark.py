import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Dark Soliton
n_0 = 1.
v = 0.75
v_1 = 3.
xi = 1.5

# Define the Dark Soliton function
def dark_soliton(x, t):
    lor = (1-(v/v_1)**2)**0.5
    psi = n_0**0.5*(1j*v/v_1 + lor * np.tanh((x-v*t)*lor/(2**0.5*xi)))

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
    real_part, imag_part, modulus = dark_soliton(x, t)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)

    ax.set_title(f'Dark Soliton\nTime [a.u.]= {t:.2f}', fontsize=20)
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
anim.save('animations/dark_soliton.gif', writer='imagemagick', fps=15)

plt.show()

# Setup the figure and axis for 3D animation
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

def animate_3d(t):
    ax3d.clear()
    re, img, _ = dark_soliton(x, t)
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
anim3d.save('animations/dark_soliton_3d.gif', writer='imagemagick', fps=15)

plt.show()