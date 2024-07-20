import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Multi-Soliton Solution (Two-Soliton Example)
eta = 1.0  # Example value for eta, you can change this as needed

# Define the Multi-Soliton Solution function
def multi_soliton(x, t):
    numerator = 4 * eta * np.exp(2j * eta**2 * t) * (np.exp(eta * x) + 1j * np.exp(-eta * x))
    denominator = np.exp(2 * eta * x) + np.exp(-2 * eta * x)
    psi = numerator / denominator
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
    real_part, imag_part, modulus = multi_soliton(x, t)
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-5.5, 5.5)

    ax.set_title(f'Multi-Soliton Solution\nTime [a.u.]= {t:.2f}', fontsize=20)
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
anim.save('animations/multi_soliton_solution.gif', writer='imagemagick', fps=15)

plt.show()
