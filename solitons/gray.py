import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the gray Soliton
n_0 = 1.
v = 0.
v_1 = 3.
xi = 1.5

# Define the gray Soliton function
def gray_soliton(x, t):
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
    real_part, imag_part, modulus = gray_soliton(x, t)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)

    ax.set_title(f'gray Soliton\nTime [a.u.]= {t:.2f}', fontsize=20)
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
anim.save('animations/gray_soliton.gif', writer='imagemagick', fps=15)

plt.show()
