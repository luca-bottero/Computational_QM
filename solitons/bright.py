import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters for the Bright Soliton
A = 1.0          # Amplitude
w_number = 1.0          # Wavenumber
x0 = -5.0         # Initial position
theta0 = 0.0         # Initial phase

# Define the Bright Soliton function
def bright_soliton(x, t, A, w_number, x0, theta0):
    # Compute the real and imaginary parts
    real_part = A * np.cosh(A * (x - 2 * w_number * t - x0))**(-1) * np.cos(w_number * x - (w_number**2 - A**2) * t + theta0)
    imag_part = A * np.cosh(A * (x - 2 * w_number * t - x0))**(-1) * np.sin(w_number * x - (w_number**2 - A**2) * t + theta0)
    return real_part, imag_part

# Setup the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)
t_max = 10
frames = 200

def animate(t):
    ax.clear()
    real_part, imag_part = bright_soliton(x, t, A, w_number, x0, theta0)
    modulus = real_part**2 + imag_part**2
    
    ax.plot(x, real_part, label='Re{$\psi(x,t)$}', color='blue')
    ax.plot(x, imag_part, label='Im{$\psi(x,t)$}', color='red')
    ax.plot(x, modulus, label='$|\psi(x,t)|^2$', color='green')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)

    ax.set_title(f'Bright Soliton\nTime [a.u.]= {t:.2f}', fontsize=20)
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
anim.save('animations/bright_soliton.gif', writer='imagemagick', fps=15)

#plt.show()

# Setup the figure and axis for 3D animation
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

def animate_3d(t):
    ax3d.clear()
    real_part, imag_part = bright_soliton(x, t, A, w_number, x0, theta0)
    modulus = np.sqrt(real_part**2 + imag_part**2)
    phase = np.arctan2(imag_part, real_part)
    
    ax3d.plot(x, modulus * np.cos(phase), modulus * np.sin(phase), label='Wavefunction')
    ax3d.plot(x, np.zeros_like(x), np.zeros_like(x), label='X-axis', color='black')
    
    ax3d.set_xlim(-10, 10)
    ax3d.set_ylim(-1.5, 1.5)
    ax3d.set_zlim(-1.5, 1.5)
    
    ax3d.set_title(f'3D Wavefunction\nTime [a.u.]= {t:.2f}', fontsize=20)
    ax3d.set_xlabel('x [a.u.]', fontsize=18)
    ax3d.set_ylabel('Real Part', fontsize=18)
    ax3d.set_zlabel('Imaginary Part', fontsize=18)
    ax3d.legend()
    return ax3d

# Create the 3D animation
anim3d = animation.FuncAnimation(fig3d, animate_3d, frames=np.linspace(0, t_max, frames), interval=25, repeat=True)

# Save the 3D animation
anim3d.save('animations/bright_soliton_3d.gif', writer='imagemagick', fps=15)

#plt.show()