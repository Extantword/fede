# transferencia.py
# Funciones @njit para cálculos de transferencia de masa, calor y mecánica de fluidos.

import numba
import numpy as np

njit = numba.njit


@njit
def coeficiente_arrastre(Re, X_w, X_cr):
    """
    Calcula el coeficiente de arrastre C_drag según el número de Reynolds
    y el régimen de secado de la gota.

    Retorna C_drag.
    """
    if Re < 1e-9:
        return 0.0

    if X_w > X_cr:
        # Gota en etapa de evaporación (correlación simplificada)
        if Re > 80:
            return 0.271 * Re**0.217
        else:
            return 27 * Re**(-0.84)

    # Gota con costra formada (correlación por rangos de Re)
    if Re <= 0.1:
        a_1, a_2, a_3 = 0.0, 24.0, 0.0
    elif Re <= 1:
        a_1, a_2, a_3 = 3.69, 22.73, 0.0903
    elif Re <= 10:
        a_1, a_2, a_3 = 1.222, 29.167, -3.889
    elif Re <= 100:
        a_1, a_2, a_3 = 0.6167, 46.5, -116.667
    elif Re <= 1000:
        a_1, a_2, a_3 = 0.3644, 98.33, -2778.0
    elif Re <= 5000:
        a_1, a_2, a_3 = 0.357, 148.62, -4.75e4
    elif Re <= 10000:
        a_1, a_2, a_3 = 0.46, -490.546, 57.87e4
    else:
        a_1, a_2, a_3 = 0.5191, -1662.5, 5.4167e6

    re_safe = max(Re, 1e-6)
    return a_1 + a_2 / re_safe + a_3 / re_safe**2


@njit
def diametro_gota(X_w, X_cr, SMD, m_ssg, X0, rho_W, P_sat, P, P_v, T_R):
    """
    Calcula el diámetro de la gota y el factor de corrección f según
    el régimen de secado.

    Retorna (d_drop, f).
    """
    if X_w > X_cr:
        # Etapa de evaporación: el diámetro se encoge
        f = 0.0
        vol_term = SMD**3 - 6 * m_ssg * (X0 - X_w) / (np.pi * rho_W)
        d = max(0.0, vol_term) ** (1.0 / 3.0)
        d = max(d, 1e-12)
        return d, f

    elif P_sat < P:
        # Costra formada, presión de saturación menor que atmosférica
        aw = P_v / P_sat  # Actividad de agua

        # Parámetros GAB para humedad de equilibrio
        Weq = 0.05 * np.exp(-99.27 / (T_R + 273.15))
        Keq = 0.65 * np.exp(144.57 / (T_R + 273.15))
        Ceq = 0.04 * np.exp(1257.14 / (T_R + 273.15))

        ConcReq = Ceq * Keq * Weq * aw / ((1 - Keq * aw) * (1 - Keq * aw + Ceq * Keq * aw))

        f = ((X_w - ConcReq) / (X_cr - ConcReq))**(-1.0 / 3.0) - 1

        vol_term = SMD**3 - 6 * m_ssg * (X0 - X_cr) / (np.pi * rho_W)
        d = max(0.0, vol_term) ** (1.0 / 3.0)
        d = max(d, 1e-12)
        return d, f

    else:
        # Caso especial: P_sat >= P
        f = 0.0
        rho_W100 = (1.0020825 - 1.14e-4 * 100 - 3.325e-6 * 100**2) * 1000

        vol_term = SMD**3 - 6 * m_ssg * (X0 - X_cr) / (np.pi * rho_W100)
        d = max(0.0, vol_term) ** (1.0 / 3.0)
        d = max(d, 1e-12)
        return d, f


@njit
def velocidad_gas(Mov_R, Mov_Z, D_eq):
    """
    Calcula las componentes de velocidad del gas en la torre
    según la posición de la gota.

    Retorna (V_gz, V_gt, V_gr).
    """
    Coef_velz = 7.376 * 0.45

    term_pos = (D_eq - Mov_R) / D_eq
    if term_pos < 0:
        term_pos = 0.0

    V_gz = Coef_velz * term_pos**2.5
    V_gt = 0.7 * term_pos**0.5

    if 0 <= Mov_R <= 0.15:
        if 0 <= Mov_Z <= 0.1:
            V_gr = -5.19
        else:
            V_gr = 0.0
    else:
        V_gr = 0.0

    return V_gz, V_gt, V_gr


@njit
def flux_evaporacion(P, MW, R, T_R, T_Ep, d_drop, f, Diff, kmass,
                     P_sat, P_v, Gotas, h_v, X_w, lamb_v):
    """
    Calcula la masa de agua evaporada m_evap [kg/s] según el régimen.

    Retorna (m_evap, mv_flux, parte3_val).
    """
    if P_sat < P:
        # Cálculo del flux de vapor
        parte1 = P * MW / (R * ((T_R + T_Ep) / 2 + 273.15))

        if d_drop > 1e-12 and kmass > 1e-12:
            denom_flux = d_drop * (f + 2 * Diff / (kmass * d_drop))
            if denom_flux != 0:
                parte2 = 2 * Diff / denom_flux
            else:
                parte2 = 0.0
        else:
            parte2 = 0.0

        if (P - P_sat) > 0 and (P - P_v) > 0:
            parte3_val = np.log((P - P_v) / (P - P_sat))
        else:
            parte3_val = 0.0

        mv_flux = parte1 * parte2 * parte3_val
        m_evap = mv_flux * Gotas * np.pi * d_drop**2

    elif X_w > 0:
        m_evap = Gotas * np.pi * d_drop**2 * h_v * (T_Ep - T_R) / lamb_v
        mv_flux = 0.0
        parte3_val = 0.0
    else:
        m_evap = 0.0
        mv_flux = 0.0
        parte3_val = 0.0

    m_evap = max(0.0, m_evap)
    return m_evap, mv_flux, parte3_val


@njit
def interpolacion_perfil(Recorrido, Mov_VarTemp, Y_p, M_Ewp, T_EpTemp,
                         T_E0, Y0, P, h_eu, V_ax_0):
    """
    Interpola el perfil de estado del aire (humedad, masa de agua, temperatura)
    en la posición actual de la gota usando los datos de la tanda anterior.

    Retorna (Y_wDL, M_wDL, T_Ep, k).
    """
    MA = 28.96
    len_prev = len(Mov_VarTemp)

    # Valores por defecto
    Y_wDL = Y0
    M_wDL = 0.0
    T_Ep = T_E0
    k = 0

    if Recorrido == 0:
        Vol_p0 = np.pi * ((2 / 2)**2) * h_eu * V_ax_0
        m_airp0 = P * Vol_p0 * MA / (8.314 * 1000 * (T_E0 + 273.15))
        M_wDL = Y0 * m_airp0
        return Y_wDL, M_wDL, T_Ep, k

    if len_prev < 3:
        # Vector previo demasiado corto
        Y_wDL = Y_p[-1]
        M_wDL = M_Ewp[-1]
        T_Ep = T_EpTemp[-1]
        k = len_prev - 1
        return Y_wDL, M_wDL, T_Ep, k

    if 0 <= Recorrido <= Mov_VarTemp[0]:
        # Interpolación inicial
        denom = Mov_VarTemp[1]
        if abs(denom) > 1e-9:
            Y_wDL = Recorrido * (Y_p[0] - Y_p[0]) / denom + Y_p[0]
            M_wDL = Recorrido * (M_Ewp[0] - M_Ewp[0]) / denom + M_Ewp[0]
            T_Ep = Recorrido * (T_EpTemp[0] - T_EpTemp[0]) / denom + T_EpTemp[0]
        else:
            Y_wDL = Y_p[0]
            M_wDL = M_Ewp[0]
            T_Ep = T_EpTemp[0]
        k = 0

    elif Mov_VarTemp[0] <= Recorrido <= Mov_VarTemp[-1]:
        # Búsqueda con searchsorted e interpolación lineal
        idx = np.searchsorted(Mov_VarTemp, Recorrido)
        if idx >= len_prev:
            idx = len_prev - 1
        idx_prev = idx - 1

        dist_diff = Mov_VarTemp[idx] - Mov_VarTemp[idx_prev]

        if abs(dist_diff) > 1e-9:
            ratio = (Recorrido - Mov_VarTemp[idx_prev]) / dist_diff
            Y_wDL = ratio * (Y_p[idx] - Y_p[idx_prev]) + Y_p[idx_prev]
            M_wDL = ratio * (M_Ewp[idx] - M_Ewp[idx_prev]) + M_Ewp[idx_prev]
            T_Ep = ratio * (T_EpTemp[idx] - T_EpTemp[idx_prev]) + T_EpTemp[idx_prev]
        else:
            Y_wDL = Y_p[idx_prev]
            M_wDL = M_Ewp[idx_prev]
            T_Ep = T_EpTemp[idx_prev]

        k = idx

    else:
        # Extrapolación final
        dist_diff = Mov_VarTemp[-1] - Mov_VarTemp[-2]
        if abs(dist_diff) > 1e-9:
            ratio = (Recorrido - Mov_VarTemp[-2]) / dist_diff
            Y_wDL = ratio * (Y_p[-1] - Y_p[-2]) + Y_p[-2]
            M_wDL = ratio * (M_Ewp[-1] - M_Ewp[-2]) + M_Ewp[-2]
            T_Ep = ratio * (T_EpTemp[-1] - T_EpTemp[-2]) + T_EpTemp[-2]
        else:
            Y_wDL = Y_p[-1]
            M_wDL = M_Ewp[-1]
            T_Ep = T_EpTemp[-1]

        k = len_prev

    return Y_wDL, M_wDL, T_Ep, k


@njit
def diametro_torre(Mov_Z):
    """
    Calcula el diámetro de la torre a la altura axial dada.
    La torre tiene sección cilíndrica hasta 2.3 m y luego cónica.

    Retorna D_eqp.
    """
    if Mov_Z < 2.3:
        return 2.0
    else:
        return np.tan(20 * np.pi / 180) * (2 * np.tan(70 * np.pi / 180) - (2.3 - Mov_Z))
