import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Dark Soliton
A = 1.0            # Background amplitude
beta = np.pi / 8  # Phase angle
v = 1.0            # Velocity
x0 = -5.0           # Initial position

# Define the Dark Soliton function
def dark_soliton(x, t, A, beta, v, x0):
    # Compute the real and imaginary parts
    argument = (A * np.sin(beta) / np.sqrt(2)) * (x - v * t - x0)
    real_part = A * (np.cos(beta) * np.tanh(argument) + 1j * np.sin(beta)) * np.exp(1j * (0.5 * v * x - (0.25 * v**2 - 0.5 * A**2) * t))
    imag_part = np.imag(real_part)
    real_part = np.real(real_part)
    modulus = np.abs(real_part + 1j * imag_part)
    return real_part, imag_part, modulus**2

# Setup the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)
t_max = 10
frames = 200

def animate(t):
    ax.clear()
    real_part, imag_part, modulus = dark_soliton(x, t, A, beta, v, x0)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 2)

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
