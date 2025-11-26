import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k, pi
import pandas as pd

# -----------------------------------------------------------------------------
# Planck spectral radiance (per wavelength)
# B_λ(T) = 2hc² / λ⁵ · 1 / (exp(hc / (λkT)) − 1)
# Returns spectral radiance in W / (m³ sr)
# -----------------------------------------------------------------------------
def planck_lambda(lmbd, T):
    """
    Compute Planck spectral radiance as a function of wavelength.

    Parameters:
        lmbd : wavelength [m]
        T    : temperature [K]

    Returns:
        Spectral radiance [W / (m^3 sr)]
    """
    return (2 * h * c**2 / lmbd**5) / (np.exp((h * c) / (lmbd * k * T)) - 1)

# -----------------------------------------------------------------------------
# Plot: Planck radiance curves for different temperatures
# -----------------------------------------------------------------------------
wavelength_um = np.linspace(0.1, 14, 1000)
wavelength_m = wavelength_um * 1e-6

plot_temperatures = [300, 650, 1000]
labels = ["300 K", "650 K", "1000 K"]

plt.figure(figsize=(8, 5))
for T, label in zip(plot_temperatures, labels):
    radiance = planck_lambda(wavelength_m, T)
    plt.plot(wavelength_um, radiance, label=label)

plt.xlabel("Wavelength (µm)")
plt.ylabel(r"Spectral Radiance [W·m⁻³·sr⁻¹]")
plt.title("Planck Spectral Radiance vs. Wavelength")
plt.yscale('log')
plt.ylim(1e6, 1e10)
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# MODTRAN CSV loader
# -----------------------------------------------------------------------------
def get_modtran_data(file_path):
    """
    Load MODTRAN output from a CSV file containing atmospheric transmission.

    Returns:
        wavelength : wavelengths [m]
        tau        : atmospheric transmission
        df         : processed dataframe
    """
    df = pd.read_csv(file_path, header=4)
    df.set_index('Freq', inplace=True)

    # remove first and last row (MODTRAN formatting artifacts)
    df = df.iloc[1:-1]

    # convert frequency index → wavelength [m] using λ = c / ν, but here file uses cm⁻¹
    df.index = 1e-2 / df.index.astype(float)
    df.index.name = 'Wavelength [m]'
    df.columns = df.columns.str.strip()

    df['combin'] = pd.to_numeric(df['combin'], errors='coerce')
    tau = df['combin'].values
    wavelength = df.index.values

    # sort in ascending wavelength
    s = np.argsort(wavelength)
    return wavelength[s], tau[s], df

# -----------------------------------------------------------------------------
# Plot MODTRAN transmittance
# -----------------------------------------------------------------------------
wavelength, transmittance, _ = get_modtran_data('./AngstromLaw2.csv')

plt.figure(figsize=(10, 6))
plt.plot(wavelength, transmittance, label='Atmospheric Transmission')
plt.xlabel('Wavelength (m)')
plt.ylabel('Transmission')
plt.title('MODTRAN Atmospheric Transmission')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Sensor & scene parameters
# -----------------------------------------------------------------------------
D = 0.06           # Aperture diameter [m]
d = 600e3          # Orbit altitude [m]
theta_ap = 0.5 * D / d   # Aperture half-angle [rad]

tau_op = 0.5       # Optical transmission
alpha_det = 0.8    # Detector efficiency

T_fire = 1000      # Fire temperature [K]
T_ground = 300     # Ground temperature [K]

gsd = 200          # Ground sampling distance [m]
area = gsd**2      # Pixel surface area [m²]

# Restrict MODTRAN band (8–14 µm)
condition = (wavelength >= 8e-6) & (wavelength <= 14e-6)
band_edges = wavelength[condition]
tau_atm = transmittance[condition]

netd_values = [0.04, 0.05, 0.06]     # NETD values

Fno = 2

# Temperature grid for simulations
temperatures = np.linspace(200, 500, 200)
delta_fire = np.linspace(0, 100, 200)

# -----------------------------------------------------------------------------
# Integrate Planck radiance across a band
# -----------------------------------------------------------------------------
def integrate_band(T, l0, l1, n_samples=200):
    wavelengths = np.linspace(l0, l1, n_samples)
    spectral = planck_lambda(wavelengths, T)
    return np.trapezoid(spectral, wavelengths)

# -----------------------------------------------------------------------------
# Generic band power computation
# -----------------------------------------------------------------------------
def generic_band_power(band_edges, tau_band, area, spectral_func, spectral_args,
                       factor, theta_ap, single_transmission, n_samples=400):
    """
    Compute band‑integrated pixel power.

    Returns:
        powers_per_band : array of band powers [W]
        total_power     : sum of all bands [W]
    """
    powers = []
    for i in range(len(band_edges)-1):
        l0, l1 = band_edges[i], band_edges[i+1]
        wavelengths = np.linspace(l0, l1, n_samples)
        spectral = spectral_func(wavelengths, *spectral_args)
        I_band = np.trapezoid(spectral, wavelengths)

        if single_transmission:
            I_band_atm = I_band * tau_band[i]
        else:
            I_band_atm = I_band * tau_band[i]**2

        solid_angle = pi * (np.sin(theta_ap))**2
        P_band = factor * area * I_band_atm * solid_angle
        powers.append(P_band)

    powers = np.array(powers)
    return powers, powers.sum()

# -----------------------------------------------------------------------------
# Band power for ground emission
# -----------------------------------------------------------------------------
ground_powers, _ = generic_band_power(
    band_edges, tau_atm, area,
    spectral_func=planck_lambda, spectral_args=(T_ground,),
    factor=tau_op * alpha_det, theta_ap=theta_ap,
    single_transmission=True
)

band_centers_um = 0.5 * (band_edges[:-1] + band_edges[1:]) * 1e6

plt.figure(figsize=(8, 5))
plt.plot(band_centers_um, ground_powers, marker='o', label="Ground Band Power")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Band Power [W]")
plt.title("Band‑Integrated Ground Power (8–14 µm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Pixel power vs. ground temperature
# -----------------------------------------------------------------------------
total_powers = []
for T in temperatures:
    _, total_power = generic_band_power(
        band_edges, tau_atm, area,
        spectral_func=planck_lambda, spectral_args=(T,),
        factor=tau_op * alpha_det, theta_ap=theta_ap,
        single_transmission=True
    )
    total_powers.append(total_power)

plt.figure(figsize=(10, 6))
plt.plot(temperatures, total_powers, label='Total LWIR Power (8–14 µm)')
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pixel Power [W]')
plt.title('Pixel Power vs. Temperature (No Fire)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Derivative of Planck radiance wrt temperature
# -----------------------------------------------------------------------------
def dplanck_dT(lmbd, T):
    x = (h * c) / (lmbd * k * T)
    expx = np.exp(x)
    dx_dT = -(h * c) / (lmbd * k * T**2)
    pref = 2 * h * c**2 / lmbd**5
    return pref * (-(expx * dx_dT) / (expx - 1)**2)

# -----------------------------------------------------------------------------
# dP/dT and NEP conversion
# -----------------------------------------------------------------------------
def dPdet_dT(T0, lambdas, tau_atm_lambda, area, F_number, tau_opt, theta_ap):
    factor = pi / (4 * F_number**2) * tau_opt
    powers, total = generic_band_power(
        lambdas, tau_atm_lambda, area,
        spectral_func=dplanck_dT, spectral_args=(T0,),
        factor=factor, theta_ap=theta_ap,
        single_transmission=True
    )
    return powers, total


def netd_to_nep(NETD, T0, lambdas, tau_atm_lambda, area, F_number, tau_opt, theta_ap):
    _, dP_dT = dPdet_dT(T0, lambdas, tau_atm_lambda, area, F_number, tau_opt, theta_ap)
    NEP = NETD * dP_dT
    return NEP, dP_dT

# -----------------------------------------------------------------------------
# Compute NEP for each NETD
# -----------------------------------------------------------------------------
nep_results = {}
for netd in netd_values:
    nep = []
    for T in temperatures:
        nep_single, _ = netd_to_nep(netd, T, band_edges, tau_atm, area, Fno, tau_op, theta_ap)
        nep.append(nep_single)
    nep_results[netd] = nep

# -----------------------------------------------------------------------------
# Plot NEP curves
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for netd, nep in nep_results.items():
    plt.plot(temperatures, nep, label=f'NETD = {netd:.3f} K')

plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('NEP [W]')
plt.title('Noise Equivalent Power vs. Temperature')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Mixed pixel model (ground + fire)
# -----------------------------------------------------------------------------
def mixed_pixel_power_ground_fire(T_ground, T_fire, delta_fire, band_edges, tau_band,
                                  area, tau_opt, alpha_det, theta_ap):
    """
    Compute pixel power for a mixture of ground and fire emission.

    delta_fire : fractional fire coverage [0..1]
    Returns: (band powers, total power)
    """
    factor = tau_opt * alpha_det

    P_ground_band, _ = generic_band_power(
        band_edges, tau_band, area, planck_lambda, (T_ground,),
        factor=factor, theta_ap=theta_ap, single_transmission=True)

    P_fire_band, _ = generic_band_power(
        band_edges, tau_band, area, planck_lambda, (T_fire,),
        factor=factor, theta_ap=theta_ap, single_transmission=True)

    P_mix_band = (1 - delta_fire) * P_ground_band + delta_fire * P_fire_band
    return P_mix_band, P_mix_band.sum()

# -----------------------------------------------------------------------------
# Mixed pixel power vs. fire coverage
# -----------------------------------------------------------------------------
mix_powers = []
for delta_cur in delta_fire:
    _, Pmix = mixed_pixel_power_ground_fire(T_ground, T_fire, delta_cur,
                                           band_edges, tau_atm, area,
                                           tau_op, alpha_det, theta_ap)
    mix_powers.append(Pmix)

nep_fire, _ = netd_to_nep(netd_values[2], T_fire, band_edges, tau_atm, area, Fno, tau_op, theta_ap)
nep_fire_array = [nep_fire] * len(delta_fire)

plt.figure(figsize=(10,6))
plt.plot(delta_fire, mix_powers, label="Mixed Pixel Power")
plt.plot(delta_fire, nep_fire_array, label="NEP Threshold (Fire Temperature)")
plt.yscale("log")
plt.xlabel('Fire Fraction [%]')
plt.ylabel('Power [W]')
plt.title('Mixed Pixel Power vs. Fire Coverage Fraction')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
