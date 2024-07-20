import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Define the Peregrine Soliton function
def peregrine_soliton(x, t):
    numerator = 1 - 4 * (1 + 2j * t)
    denominator = 1 + 4 * x**2 + 4 * t**2
    psi = (numerator / denominator) * np.exp(1j * t)
    real_part = np.real(psi)
    imag_part = np.imag(psi)
    modulus = np.abs(psi)
    return real_part, imag_part, modulus

# Setup the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)
t_max = 10
t_0 = -5.
frames = 200

def animate(t):
    ax.clear()
    t += t_0
    real_part, imag_part, modulus = peregrine_soliton(x, t)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-3.5, 3.5)
    
    ax.set_title(f'Peregrine Soliton\nTime [a.u.]= {t:.2f}', fontsize=20)
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
anim.save('animations/peregrine_soliton.gif', writer='imagemagick', fps=10)

plt.show()
