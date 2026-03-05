# propiedades.py
# Funciones @njit para calcular propiedades termofísicas.
# Estas funciones eliminan la repetición de fórmulas en Spray().

import numba
import numpy as np

njit = numba.njit


@njit
def entalpia_aire(T, T_Ref):
    """
    Entalpía específica del aire seco [kJ/kg].
    Polinomio integrado de Cp del aire entre T_Ref y T.
    """
    T_K = T + 273.15
    T_Ref_K = T_Ref + 273.15

    h1 = (969.542 / 1000) * (T - T_Ref)
    h2 = ((6.801e-2) / 1000) * (T_K**2 / 2 - T_Ref_K**2 / 2)
    h3 = ((16.569e-5) / 1000) * (T_K**3 / 3 - T_Ref_K**3 / 3)
    h4 = ((-67.828e-9) / 1000) * (T_K**4 / 4 - T_Ref_K**4 / 4)

    return h1 + h2 + h3 + h4


@njit
def entalpia_vapor(T, T_Ref):
    """
    Entalpía específica del vapor de agua [kJ/kg].
    Polinomio integrado de Cp del vapor entre T_Ref y T.
    """
    h1 = 1.883 * (T - T_Ref)
    h2 = (-1.674e-4) * (T**2 / 2 - T_Ref**2 / 2)
    h3 = (8.4390e-7) * (T**3 / 3 - T_Ref**3 / 3)
    h4 = (-2.6970e-10) * (T**4 / 4 - T_Ref**4 / 4)

    return h1 + h2 + h3 + h4


@njit
def densidad_agua(T):
    """
    Densidad del agua líquida [kg/m³] en función de la temperatura T [°C].
    """
    return (1.0020825 - 1.14e-4 * T - 3.325e-6 * T**2) * 1000


@njit
def densidad_aire(T):
    """
    Densidad del aire [kg/m³] usando gas ideal a 1 atm.
    T en °C.
    """
    return 1.293 * 273.15 / (273.15 + T)


@njit
def viscosidad_aire(T):
    """
    Viscosidad dinámica del aire [Pa·s].
    T en °C.
    """
    return 1.72e-5 + 4.568e-8 * T


@njit
def conductividad_aire(T):
    """
    Conductividad térmica del aire [kW/(m·K)].
    T en °C.
    """
    return 1.731 * (0.014 + 4.296e-5 * T) / 1000


@njit
def cp_aire(T):
    """
    Capacidad calorífica del aire [kJ/(kg·K)].
    T en °C.
    """
    T_K = T + 273.15
    return (969.542 + 6.801e-2 * T_K + 16.569e-5 * T_K**2 - 67.828e-9 * T_K**3) / 1000


@njit
def cp_vapor(T):
    """
    Capacidad calorífica del vapor de agua [kJ/(kg·K)].
    T en °C.
    """
    return 1.883 - 1.674e-4 * T + 8.4390e-7 * T**2 - 2.6970e-10 * T**3


@njit
def presion_saturacion(T):
    """
    Presión de vapor saturado del agua [Pa] (ecuación de Antoine).
    T en °C.
    """
    return (101325.0 / 760) * 10**(7.95581 - 1668.210 / (T + 228))


@njit
def calor_latente(T):
    """
    Calor latente de vaporización del agua [kJ/kg].
    T en °C.
    """
    return 3.15e3 - (T + 273.15) * 2.38


@njit
def difusividad_vapor(T_aire, T_gota, P):
    """
    Difusividad del vapor de agua en aire [m²/s].
    T_aire, T_gota en °C. P en Pa.
    """
    T_media = (T_aire + T_gota) / 2 + 273.15
    return 2.302e-5 * 0.98e5 / P * (T_media / 256)**1.81


@njit
def densidad_maltodextrina(T):
    """
    Densidad de la maltodextrina [g/cm³].
    T en °C.
    """
    return 1.635 - 0.0026 * T + 2e-5 * T**2


@njit
def densidad_gota(T, x_R):
    """
    Densidad inicial de la gota [kg/m³].
    T en °C, x_R fracción de soluto.
    """
    rho_M = densidad_maltodextrina(T)
    rho_W = densidad_agua(T) / 1000.0  # convertir a g/cm³
    return 100000.0 / (100 * (1 - x_R) * (1 / rho_M - 1 / rho_W) + 100.0 / rho_W)


@njit
def dcv_vapor_dT(T):
    """
    Derivada de la capacidad calorífica a volumen constante del vapor
    respecto a la temperatura.
    T en °C.
    """
    return 2.558e-12 * T**3 + (-4.681e-9) * T**2 + 2.615e-6 * T + 0.0001957


@njit
def dcv_aire_dT(T):
    """
    Derivada de la capacidad calorífica a volumen constante del aire
    respecto a la temperatura.
    T en °C.
    """
    return 9.785e-6 * T**0.5526 - 1.634e-5
