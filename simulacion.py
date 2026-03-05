# simulacion.py
# Función principal Spray() refactorizada para la simulación
# de una torre de secado por aspersión.
# Basado en el trabajo de Santiago González Gallego y Lina Steffania.

import numpy as np
import math as mt
import numba
from numba.typed import List
import numba.types as nt

from propiedades import (
    entalpia_aire, entalpia_vapor, densidad_agua, densidad_aire,
    viscosidad_aire, conductividad_aire, cp_aire, cp_vapor,
    presion_saturacion, calor_latente, difusividad_vapor,
    densidad_maltodextrina, densidad_gota, dcv_vapor_dT, dcv_aire_dT,
)
from transferencia import (
    coeficiente_arrastre, diametro_gota, velocidad_gas,
    flux_evaporacion, interpolacion_perfil, diametro_torre,
)

njit = numba.njit


@njit
def Spray(T_1, h_eu_1, Part_1, Mov_VarTemp_1, T_EpTemp_1, Y_p_1, M_wDL_1):
    """
    Función de simulación para un proceso de secado por aspersión.
    Calcula la trayectoria, temperatura y humedad de las gotas a lo largo del tiempo.

    Parámetros:
        T_1: Número de tandas (pasos de tiempo macroscópicos)
        h_eu_1: Paso de tiempo de Euler [s]
        Part_1: Máximo número de iteraciones por tanda de gotas
        Mov_VarTemp_1: Perfil de posición axial del aire (inicial)
        T_EpTemp_1: Perfil de temperatura del aire (inicial)
        Y_p_1: Perfil de humedad absoluta del aire (inicial)
        M_wDL_1: Masa de agua en la fase gaseosa (inicial)
    """

    # ================================================================
    # Parámetros de simulación
    # ================================================================

    T = T_1
    h_eu = h_eu_1
    Part = Part_1
    TotalTime = T * h_eu

    # ================================================================
    # Almacenamiento de resultados globales
    # ================================================================

    T_ECT  = List.empty_list(nt.float64)
    T_Eout = List.empty_list(nt.float64)
    Y_Eout = List.empty_list(nt.float64)
    Z_Mout = List.empty_list(nt.float64)
    P_inMV = List.empty_list(nt.float64)
    m_LMV  = List.empty_list(nt.float64)
    R_time = List.empty_list(nt.float64)

    # Perfiles del estado del aire (se actualizan entre tandas)
    Mov_VarTemp = Mov_VarTemp_1
    T_EpTemp    = T_EpTemp_1
    Y_p         = Y_p_1
    M_wDL       = M_wDL_1

    f_param = 1.0  # Parámetro de ajuste del balance de masa

    # ================================================================
    # Sistema de calentamiento y ducto
    # ================================================================

    T_outh = np.zeros(T)
    P_h = np.zeros(T)
    time_sim = 0
    m_outh = np.zeros(T)
    T_wall = np.zeros(T)
    m_ngas = np.zeros(T)
    T_duct = np.zeros(T)
    m_duct = np.zeros(T)
    Delay_j = np.zeros(2)

    # Inicialización de snapshots (Bef0-Bef6)
    dummy_array = np.zeros(1)
    Bef0 = (0, dummy_array, dummy_array, dummy_array, dummy_array,
            dummy_array, dummy_array, dummy_array, dummy_array, 0.0)
    Bef1 = Bef0
    Bef2 = Bef0
    Bef3 = Bef0
    Bef4 = Bef0
    Bef5 = Bef0
    Bef6 = Bef0

    # Condiciones iniciales del sistema
    T_outh[0] = 25
    P_h[0] = 101325
    T_wall[0] = 200
    g = 9.81

    mv_flux = 0.0
    f_balanc = 0.0

    # ================================================================
    # Parámetros experimentales de la boquilla
    # ================================================================

    q_n = 2.09
    D_n = 70.5e-6

    GammaF = mt.gamma(1 - 1 / q_n)
    SMD_exp = D_n * GammaF**(-1)
    MMD_SMD = (0.693**(1 / q_n)) * GammaF

    # Geometría de la boquilla
    d0  = 0.711e-3
    Ds  = 2.4 * d0
    A_p = 0.50 * d0 * Ds
    l0  = 0.1 * d0
    Ls  = 0.5 * Ds

    # Propiedades del líquido
    x_R = 0.46
    T_R0 = 52.5
    mu_L = 1.8e-3
    P_in = 9e5
    P_out = 101325.0
    P_L = P_in - P_out

    rho_p0 = densidad_gota(T_R0, x_R)

    # ================================================================
    # Sistema de calentamiento
    # ================================================================

    T_Ref = 0
    T_amb = 25

    Cp_ah = cp_aire(T_amb)
    R_h = 8.314 / 29
    Cvair = Cp_ah - R_h

    H_airinh = entalpia_aire(T_amb, T_Ref)

    m_inh = 1710
    V_h = 4
    h_comb = 48.5e3
    P_duct = 101325
    P_oper_r = 1.5
    C = m_inh / (P_duct * ((P_oper_r * (P_oper_r - 1))**0.5))

    T_amb_K = T_amb + 273.15
    dCvdT = (6.801e-2 + 2 * 16.569e-5 * T_amb_K - 3 * 67.828e-9 * T_amb_K**2) / 1000

    # Ducto
    Dist_duct = 9
    A_duct = 0.45 * 0.45
    Vel_duct = 0.0

    T_EC = 170.3
    T_E0 = T_EC

    # ================================================================
    # Hidrodinámica de la boquilla
    # ================================================================

    K   = A_p / (Ds * d0)
    K_v = (0.00367 * K**0.29) * (P_L * rho_p0 / mu_L)**0.2
    Vel = K_v * ((2 * P_L / rho_p0)**0.5)

    C1 = d0 * rho_p0 * Vel / mu_L
    C2 = l0 / d0
    C3 = Ls / Ds
    C4 = A_p / (Ds * d0)
    C5 = Ds / d0

    C_D = 0.45 * C1**(-0.02) * C2**(-0.03) * C3**0.05 * C4**0.52 * C5**0.23
    A_0 = mt.pi * (d0 / 2)**2

    Param_m_L = 0.802
    m_L = Param_m_L * C_D * A_0 * rho_p0 * Vel / K_v

    FNum = m_L / (rho_p0**0.5 * P_L**0.5)

    # Ángulo del cono
    theta_2 = (16.156 / 2) * K**(-0.39) * d0**1.13 * mu_L**(-0.9) * P_L**0.39
    theta = theta_2 - 360

    if theta * 180 / mt.pi > 90:
        theta = 80 * mt.pi / 180

    # ================================================================
    # Propiedades del Spray
    # ================================================================

    D_0   = 154.39
    S_0   = 0.5
    F_p0  = 0.021921
    T_p0r = 326.35

    a_smd = 400
    b_smd = -10.1
    c_smd = -600.0
    F_pin = 89.2 / 3600.0
    U_r = 0.5
    T_p0  = T_R0 + 273.15
    S_in  = x_R
    MMD_SMD = 1.1
    SMD = (D_0 + a_smd * (F_pin - F_p0) + b_smd * (T_p0 - T_p0r) + c_smd * (S_in - S_0)) * 1e-6 / (MMD_SMD * 1.45)
    m_L = F_pin
    U_tan = 1.0
    U_x = 7

    # ================================================================
    # Arrays para el bucle interno de la gota
    # ================================================================

    V_ax  = np.zeros(Part)
    V_rad = np.zeros(Part)
    V_tan = np.zeros(Part)
    R_eq  = np.zeros(Part)
    h_eq  = np.zeros(Part)
    Mov_R = np.zeros(Part)
    Mov_Z = np.zeros(Part)
    M_R   = np.zeros(Part)
    M_Rw  = np.zeros(Part)
    M_E   = np.zeros(Part)
    M_Ew  = np.zeros(Part)
    M_Ewp = np.zeros(Part)
    M_wDL = np.zeros(Part)
    x_w   = np.zeros(Part)
    X_w   = np.zeros(Part)
    y_w   = np.zeros(Part)
    Y_w   = np.zeros(Part)
    Y_pt  = np.zeros(Part)
    Y_wDL = np.zeros(Part)
    T_R   = np.zeros(Part)
    T_E   = 75 * np.ones(Part)
    T_Ep  = np.zeros(Part)
    parte3 = np.zeros(Part)
    d_drop = np.zeros(Part)

    # Condiciones iniciales de la gota
    m_air = 1710.0 / 3600.0
    V_rad[0] = U_r
    V_ax[0] = U_x
    V_tan[0] = U_tan
    T_R[0] = T_R0
    d_drop[0] = SMD

    x_w0 = 1.0 - x_R
    Y0 = 0.0021
    y_w0 = Y0 / (1.0 - Y0)

    M_R[0]  = m_L * h_eu
    M_Rw[0] = x_w0 * m_L * h_eu
    M_E[0]  = m_air * h_eu
    M_Ew[0] = y_w0 * m_air * h_eu

    rho_air0 = densidad_aire(T_E0)
    D_eq = 2.0
    Vel_g = m_air / (rho_air0 * np.pi * (D_eq / 2.0)**2)

    m = rho_p0 * np.pi * (1.0 / 6.0) * SMD**3
    Gotas = (m_L / m) * h_eu

    if (M_E[0] - M_Ew[0]) != 0:
        Y_w[0] = M_Ew[0] / (M_E[0] - M_Ew[0])
    else:
        Y_w[0] = 0.0

    # Constantes derivadas
    m_da  = (1.0 - y_w0) * m_air
    m_ss  = m_L * (1.0 - x_w0)
    m_ssi = m_ss * h_eu
    m_ssg = m * (1.0 - x_w0)  # Masa de maltodextrina en una gota

    MW = 18.0
    MA = 28.96
    R = 8314.0
    P = 101325.0
    sigma = 5.67e-11
    epsil = 0.96
    Cp_M = 1.5
    Cp_wl = 4.18
    X_cr = 0.54
    Diff = 2.5 * 5.9e-9
    U_h = 16.75 * 0.4 / 3600.0

    # ================================================================
    # Bucle principal de simulación (tandas)
    # ================================================================

    for j in range(0, T):

        if j > 0:
            P_h[j] = P_h[j - 1]
            T_outh[j] = T_outh[j - 1]
            m_ngas[j] = 6

        # Entalpía del aire de salida del calentador
        H_airout = entalpia_aire(T_outh[j], T_Ref)

        Q = 0.85 * m_ngas[j] * h_comb
        m_outh[j] = C * ((P_h[j] * (P_h[j] - P_duct))**0.5)

        Massb_1 = ((T_outh[j] + 273.15) * R_h * 1000 / V_h) * (m_inh - m_outh[j])
        Energb_1 = (Q + m_inh * H_airinh - m_outh[j] * H_airout) / (dCvdT * P_h[j] * V_h / (1000 * R_h))
        dPdt = (Massb_1 + P_h[j] * Energb_1 / (T_outh[j] + 273.15)) / (1 + Cvair / (dCvdT * (T_outh[j] + 273.15)))
        dTdt = Energb_1 - (Cvair * dPdt / (P_h[j] * dCvdT))

        if j < T - 1:
            P_h[j + 1] = P_h[j] + h_eu * dPdt / 3600
            T_outh[j + 1] = T_outh[j] + h_eu * dTdt / 3600
            time_sim = time_sim + h_eu

            if time_sim < TotalTime:

                # ---- Modelo del ducto ----
                if m_outh[j] > 1e-9:
                    Vel_duct = m_outh[j] / (3600 * 1.293 * (273.15 / (273.15 + T_outh[j])) * A_duct)

                if j <= 2 or Vel_duct < 1e-9:
                    Delay = 0
                    Delayp = 0
                else:
                    Delay = Dist_duct / (h_eu * Vel_duct)
                    Vel_ductp = m_outh[j - 1] / (3600 * 1.293 * (273.15 / (273.15 + T_outh[j - 1])) * A_duct)
                    if Vel_ductp > 1e-9:
                        Delayp = Dist_duct / (h_eu * Vel_ductp)
                    else:
                        Delayp = 0.0

                Delay_j[0] = mt.ceil(Delayp)
                Delay_j[1] = mt.ceil(Delay)

                Pos_j1 = j - 1 + Delay_j[0]
                Pos_j2 = j + Delay_j[1]

                if Pos_j2 < T:
                    if j == 1:
                        Pos_j1 = 0
                    for h in range(int(Pos_j1), int(Pos_j2)):
                        T_duct[h] = T_outh[j - 1]
                    T_duct[int(Pos_j2)] = T_outh[j]

                # ============================================================
                # Bucle interno: trayectoria de un lote de gotas
                # ============================================================

                for i in range(Part - 1):

                    if np.isnan(Mov_Z[i]):
                        break

                    # ---- Interpolación del perfil de aire ----
                    if j == 0:
                        T_Ep[i] = 75.0
                        Y_wDL[i] = Y0

                        Vol_p0 = np.pi * ((2 / 2)**2) * h_eu * V_ax[i]
                        m_airp0 = P * Vol_p0 * MA / (8.314 * 1000 * (T_E0 + 273.15))
                        M_wDL[i] = Y0 * m_airp0

                        k = i
                        T_Ep[k] = T_E0
                        Y_wDL[k] = Y0
                    else:
                        Recorrido = Mov_Z[i]
                        len_prev = len(Mov_VarTemp)

                        # Valores por defecto
                        Y_wDL[i] = Y_p[min(i, len_prev - 1)]
                        M_wDL[i] = M_Ewp[min(i, len_prev - 1)]
                        T_Ep[i]  = T_EpTemp[min(i, len_prev - 1)]
                        k = 0

                        interp_Y, interp_M, interp_T, interp_k = interpolacion_perfil(
                            Recorrido, Mov_VarTemp, Y_p, M_Ewp, T_EpTemp,
                            T_E0, Y0, P, h_eu, V_ax[0]
                        )
                        Y_wDL[i] = interp_Y
                        M_wDL[i] = interp_M
                        T_Ep[i]  = interp_T
                        k = interp_k

                    # ---- Fracciones de humedad ----
                    X_w[i] = M_Rw[i] / m_ssi
                    if (m_ssi + M_Rw[i]) > 0:
                        x_w[i] = M_Rw[i] / (m_ssi + M_Rw[i])
                    else:
                        x_w[i] = 0.0

                    Y_w[i] = M_Ew[i] / (m_da * h_eu)
                    Delta_L = h_eu * V_ax[i]

                    D_eqp = diametro_torre(Mov_Z[i])
                    Vol = np.pi * (D_eq / 2)**2 * Delta_L
                    y_w[i] = Y_wDL[i] / (1 + Y_wDL[i])

                    # ---- Propiedades termofísicas ----
                    rho_W_calc = densidad_agua(T_R[i])
                    rho_W = max(500.0, rho_W_calc)

                    lamb_v = calor_latente(T_R[i])
                    mu_air  = viscosidad_aire(T_Ep[i])
                    rho_Air = densidad_aire(T_Ep[i])
                    D_air   = difusividad_vapor(T_Ep[i], T_R[i], P)
                    k_d     = conductividad_aire(T_Ep[i])
                    Cp_a    = cp_aire(T_Ep[i])
                    Cp_vap  = cp_vapor(T_Ep[i])
                    P_sat   = presion_saturacion(T_R[i])

                    y_wmol  = Y_wDL[i] * (MA / MW) / (1 + Y_wDL[i] * (MA / MW))
                    P_v     = P * y_wmol

                    # ---- Entalpías del aire actual ----
                    H_a  = entalpia_aire(T_Ep[i], T_Ref)
                    H_vw = entalpia_vapor(T_Ep[i], T_Ref)

                    # ---- Diámetro de la gota ----
                    X0 = x_w0 / (1.0 - x_w0)

                    d_drop_i, f = diametro_gota(X_w[i], X_cr, SMD, m_ssg, X0,
                                                rho_W, P_sat, P, P_v, T_R[i])
                    d_drop[i] = d_drop_i

                    # ---- Velocidad del gas ----
                    Vel_g = 0 * m_air / (rho_Air * np.pi * ((2 / 2)**2))
                    V_gz, V_gt, V_gr = velocidad_gas(Mov_R[i], Mov_Z[i], D_eq)

                    # ---- Números adimensionales ----
                    V_rel = ((V_ax[i] - V_gz)**2 + (0.0 - V_gt)**2 + (V_rad[i] - V_gr)**2)**0.5
                    Re = rho_Air * d_drop[i] * abs(V_rel) / max(mu_air, 1e-10)

                    Sc = mu_air / (rho_Air * D_air)
                    if lamb_v == 0:
                        lamb_v = 1e-9
                    B = Cp_vap * (T_Ep[i] - T_R[i]) / lamb_v
                    Sh = (2 + 0.6 * Re**0.5 * Sc**(1 / 3)) / ((1 + B)**0.7)
                    Pr = mu_air * Cp_a / k_d

                    # ---- Coeficientes de transferencia ----
                    if d_drop[i] > 1e-12:
                        kmass = 0.4 * Sh * D_air / d_drop[i]
                    else:
                        kmass = 0.0

                    T_Ep_K = T_Ep[i] + 273.15
                    T_R_K = T_R[i] + 273.15

                    h_rad = sigma * epsil * (T_Ep_K + T_R_K) * (T_Ep_K**2 + T_R_K**2) / 1000
                    h_conv = (k_d / d_drop[i]) * (2 + 0.6 * Re**0.5 * Pr**1 / 3) / ((1 + B)**0.7)
                    h_v = h_rad + h_conv

                    # ---- Evaporación ----
                    m_evap, mv_flux_i, parte3_val = flux_evaporacion(
                        P, MW, R, T_R[i], T_Ep[i], d_drop[i], f, Diff, kmass,
                        P_sat, P_v, Gotas, h_v, X_w[i], lamb_v
                    )
                    mv_flux = mv_flux_i
                    parte3[i] = parte3_val

                    # ---- Fuerzas y dinámica ----
                    m_gota = m_ssg + M_Rw[i] / Gotas

                    C_drag = coeficiente_arrastre(Re, X_w[i], X_cr)

                    F_dt = (np.pi / 8) * rho_Air * d_drop[i]**2 * C_drag * np.abs(V_gt - V_tan[i]) * (V_gt - V_tan[i])
                    F_dr = (np.pi / 8) * rho_Air * d_drop[i]**2 * C_drag * np.abs(V_gr - V_rad[i]) * (V_gr - V_rad[i])
                    alfa = (np.pi / 12) * rho_Air * d_drop[i]**3

                    F_dz = (np.pi / 8) * rho_Air * d_drop[i]**2 * C_drag * np.abs(V_gz - V_ax[i]) * (V_gz - V_ax[i])
                    F_b  = -(np.pi / 8) * rho_Air * d_drop[i]**3 * g
                    Sum_Fax = F_dz + F_b

                    # ---- Integración de Euler (velocidades y posiciones) ----
                    dV_tan = F_dt / (m_gota + alfa)
                    V_rad[i + 1] = V_tan[i] + h_eu * dV_tan

                    dV_rad = F_dr / (m_gota + alfa)
                    V_rad[i + 1] = V_rad[i] + h_eu * dV_rad

                    dV_ax = (Sum_Fax / m_gota + g) / (1 + alfa / m_gota)
                    V_ax[i + 1] = V_ax[i] + h_eu * dV_ax

                    Mov_R[i + 1] = Mov_R[i] + h_eu * V_rad[i]
                    Mov_Z[i + 1] = Mov_Z[i] + h_eu * V_ax[i]

                    # ---- Balances de masa y energía ----
                    D_eqp = diametro_torre(Mov_Z[i])

                    k_safe = min(k, len(T_EpTemp) - 1)
                    rho_Airk = densidad_aire(T_EpTemp[k_safe])

                    f_air_i = rho_Air * Vel_g * np.pi * (D_eq / 2)**2
                    f_air_k = rho_Airk * Vel_g * np.pi * (D_eq / 2)**2

                    # Entalpías del paso previo
                    H_app  = entalpia_aire(T_EpTemp[k_safe], T_Ref)
                    H_vwpp = entalpia_vapor(T_EpTemp[k_safe], T_Ref)

                    # Entalpía del agua que se evapora
                    H_wvevap = entalpia_vapor(T_Ep[i], T_Ref) - entalpia_vapor(T_R[i], T_Ref)
                    # Recalcular como en el original: integral desde T_R a T_Ep
                    C_Rv1 = 1.883 * (T_Ep[i] - T_R[i])
                    C_Rv2 = (-1.674e-4) * (T_Ep[i]**2 / 2 - T_R[i]**2 / 2)
                    C_Rv3 = (8.4390e-7) * (T_Ep[i]**3 / 3 - T_R[i]**3 / 3)
                    C_Rv4 = (-2.6970e-10) * (T_Ep[i]**4 / 4 - T_R[i]**4 / 4)
                    H_wvevap = C_Rv1 + C_Rv2 + C_Rv3 + C_Rv4

                    lamb_v0 = 3.15e3 - (T_Ref + 273.15) * 2.38

                    H_p = H_app + (H_vwpp + lamb_v0) * Y_p[k_safe]
                    H   = H_a  + (H_vw + lamb_v0) * Y_wDL[i]

                    # Balance de masa
                    dM_Rwdt = -m_evap
                    dM_Ewdt = f_air_k * Y_p[k_safe] - f_air_i * Y_wDL[i] + m_evap

                    M_Rw[i + 1] = max(0.0, M_Rw[i] + h_eu * dM_Rwdt)

                    RemainW = m_evap

                    m_airp = P * Vol * MA / (8.314 * 1000 * (T_Ep[i] + 273.15))
                    M_Ew[i] = Y_wDL[i] * m_airp + h_eu * dM_Ewdt

                    # Balance de energía
                    denTR = M_Rw[i] * Cp_wl + m_ssi * Cp_M

                    Cv_air = Cp_a - 8.314 / MA
                    Cv_wv  = Cp_vap - 8.314 / MW

                    dCv_wv = dcv_vapor_dT(T_Ep[i])
                    dCv_air = dcv_aire_dT(T_Ep[i])

                    denTE = (m_airp * (T_Ep[i] + 273.15) * dCv_air + Y_wDL[i] * m_airp * (T_Ep[i] + 273.15) * dCv_wv) + (m_airp * Cv_air + Y_wDL[i] * m_airp * Cv_wv)

                    NumTR = -m_evap * (Cp_wl * (T_R[i] - T_Ref) + lamb_v) + Gotas * np.pi * d_drop[i]**2 * h_v * (T_Ep[i] - T_R[i]) - Cp_M * (T_R[i] + 273.15) * dM_Rwdt

                    T_Edif = Cv_wv * (T_Ep[i] + 273.15) * dM_Ewdt

                    NumTE = (f_air_k * H_p - f_air_i * H + m_evap * H_wvevap
                             - Gotas * np.pi * d_drop[i]**2 * h_v * (T_Ep[i] - T_R[i])
                             - np.pi * D_eqp * Delta_L * U_h * (T_Ep[i] - 25) - T_Edif)

                    if abs(denTR) > 1e-9:
                        dT_Rdt = NumTR / denTR
                    else:
                        dT_Rdt = 0.0

                    if abs(denTE) > 1e-9:
                        dT_Edt = NumTE / denTE
                    else:
                        dT_Edt = 0.0

                    # Solución de Euler
                    T_R[i + 1] = T_R[i] + h_eu * dT_Rdt
                    T_E[i] = T_Ep[i] + h_eu * dT_Edt
                    Y_pt[i] = M_Ew[i] / (P * Vol * MA / (8.314 * 1000 * (T_E[i] + 273.15)))

                    # ---- Condición de salida (gota alcanza pared o fondo) ----
                    if Mov_Z[i] > 3.5 or Mov_R[i] >= D_eqp / 2.0:

                        if j == 0 or j == (T - 1):
                            print()
                            print("Se alcanzó la altura del equipo")
                            print("La última iteración fue la", i)
                            print("El diferencial de tiempo para las iteraciones es de:", h_eu, "s")

                        T_Eg = T_Ep[0:i]
                        T_EpTemp = T_E[0:i]

                        # Ajuste del balance de masa
                        if j % 250 == 0:
                            f_balanc = ((Y0 / (1 + Y0)) * m_air + m_L * (1 - x_R)
                                        - (1 - Y0) * m_air * Y_wDL[i]
                                        - (x_R * m_L) * X_w[i]) * 3600

                            if abs(f_balanc) > 1:
                                if 0.6 < f_param < 1.4:
                                    if f_balanc < 0:
                                        f_param = f_param + 0.001
                                    else:
                                        f_param = f_param - 0.001

                        Vel_gF = m_air / (rho_air0 * np.pi * ((2 / 2)**2))
                        Mov_VarTemp = Mov_Z[0:i] + f_param * Vel_gF * h_eu
                        M_Ewp = M_Ew[0:i]
                        Y_graf = Y_wDL[0:i]
                        Vel_graf = V_ax[0:i]
                        Y_p = Y_pt[0:i]

                        # Almacenar resultados periódicos
                        if j % 1000 == 0:
                            T_ECT.append(T_EC)
                            T_Eout.append(T_Eg[-1])
                            Y_Eout.append(Y_graf[-1])
                            Z_Mout.append(X_w[i])
                            R_time.append(h_eu * i)
                            P_inMV.append(P_in)
                            m_LMV.append(m_L)

                        # Snapshots en iteraciones específicas
                        snapshot = (i, d_drop[0:i], Mov_Z[0:i], T_Eg, Y_graf,
                                    T_R[0:i], Mov_R, M_Ew, X_w[0:i], mv_flux)

                        if j == 1000:
                            Bef0 = snapshot
                        if j == 2000:
                            Bef1 = snapshot
                        if j == 3000:
                            Bef2 = snapshot
                        if j == 4000:
                            Bef3 = snapshot
                        if j == 15000:
                            Bef4 = snapshot
                        if j == 20000:
                            Bef5 = snapshot
                        if j == 30000:
                            Bef6 = snapshot

                        break

            else:
                break

    return (parte3[0:i], d_drop[0:i], i, Mov_Z[0:i], T_Eg, Y_graf, T_R[0:i], V_gz, Mov_R, M_Ew,
            Vel_graf, X_w[0:i], Y_p, mv_flux, Vel_g, Vel_gF,
            Bef0, Bef1, Bef2, Bef3, Bef4, Bef5, Bef6, T_Eout, Y_Eout, T_ECT, Z_Mout,
            P_inMV, m_LMV, RemainW, f_param, f_balanc, R_time)
