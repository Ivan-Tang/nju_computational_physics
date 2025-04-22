import numpy as np
import matplotlib.pyplot as plt

N = 400                  
x = np.linspace(0, 1, N) 
dx = x[1] - x[0]
gamma = 2.0              
CFL = 0.4
t_end = 0.2

def initial_conditions():
    rho = np.where(x < 0.5, 1.0, 0.125)
    vx  = np.where(x < 0.5, 0.0, 0.0)
    vy  = np.where(x < 0.5, 0.0, 0.0)
    vz  = np.where(x < 0.5, 0.0, 0.0)
    By  = np.where(x < 0.5, 1.0, -1.0)
    Bz  = np.where(x < 0.5, 0.0, 0.0)
    Bx  = 0.75
    p   = np.where(x < 0.5, 1.0, 0.1)

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    E = p / (gamma - 1) + 0.5 * rho * v2 + 0.5 * B2

    U = np.array([rho, rho*vx, rho*vy, rho*vz, By, Bz, E])  # shape: (7, N)
    return U, Bx

def compute_flux(U, Bx):
    rho, mx, my, mz, By, Bz, E = U
    vx = mx / rho
    vy = my / rho
    vz = mz / rho

    p_gas = (gamma - 1) * (E - 0.5 * rho * (vx**2 + vy**2 + vz**2) - 0.5 * (Bx**2 + By**2 + Bz**2))
    ptot = p_gas + 0.5 * (Bx**2 + By**2 + Bz**2)


    flux = np.array([
        mx,
        mx * vx + ptot - Bx**2,
        my * vx - Bx * By,
        mz * vx - Bx * Bz,
        By * vx - vy * Bx,
        Bz * vx - vz * Bx,
        (E + ptot) * vx - Bx * (vx * Bx + vy * By + vz * Bz)
    ])
    return flux

def hll_flux(UL, UR, Bx):

    flux_L = compute_flux(UL, Bx)
    flux_R = compute_flux(UR, Bx)

    rhoL, mxL, *_ = UL
    rhoR, mxR, *_ = UR
    vxL = mxL / rhoL
    vxR = mxR / rhoR
    aL = np.sqrt(gamma * (compute_pressure(UL, Bx)) / rhoL)
    aR = np.sqrt(gamma * (compute_pressure(UR, Bx)) / rhoR)

    SL = np.minimum(vxL - aL, vxR - aR)
    SR = np.maximum(vxL + aL, vxR + aR)

    # HLL flux
    flux = np.where(SL > 0, flux_L, np.where(SR < 0, flux_R, (SR * flux_L - SL * flux_R + SL * SR * (UR - UL)) / (SR - SL)))
    return flux

def compute_pressure(U, Bx):
    rho, mx, my, mz, By, Bz, E = U
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    magnetic = 0.5 * (Bx**2 + By**2 + Bz**2)
    p = (gamma - 1) * (E - kinetic - magnetic)
    return np.clip(p, 1e-10, np.inf)

def timestep(U, Bx):
    rho, mx, *_ = U
    vx = mx / rho
    a = np.sqrt(gamma * compute_pressure(U, Bx) / rho)
    max_speed = np.max(np.abs(vx) + a)
    dt = CFL * dx / max_speed
    return dt

def evolve(U, Bx, dt):
    UL = U[:, :-1]
    UR = U[:, 1:]
    flux = hll_flux(UL, UR, Bx)

    U_new = U.copy()
    U_new[:, 1:-1] -= dt / dx * (flux[:, 1:] - flux[:, :-1])
    return U_new

def main():
    U, Bx = initial_conditions()
    t = 0.0
    while t < t_end:
        dt = timestep(U, Bx)
        if t + dt > t_end:
            dt = t_end - t
        U = evolve(U, Bx, dt)
        t += dt
        print(f"t = {t:.4f}", end='\r')


    rho = U[0]
    vx = U[1] / U[0]
    By = U[4]


    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(x, rho)
    plt.title('Density')
    plt.savefig('density.jpg') 

    plt.subplot(3, 1, 2)
    plt.plot(x, vx)
    plt.title('Velocity vx')
    plt.savefig('velocity.jpg') 

    plt.subplot(3, 1, 3)
    plt.plot(x, By)
    plt.title('Magnetic Field By')
    plt.savefig('magnetic_field.jpg')

    plt.tight_layout()
    plt.close() 
    print("Done!")
if __name__ == '__main__':
    main()
