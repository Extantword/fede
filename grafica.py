# grafica.py
# Script de ejecución y generación de gráficas de la simulación.
# Reemplazo limpio de "Código gráfica.py".

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import time

from simulacion import Spray
from constantes import (
    IDX_PARTE3, IDX_D_DROP, IDX_FINAL_I, IDX_MOV_Z, IDX_T_EG, IDX_Y_GRAF,
    IDX_T_R, IDX_V_GZ, IDX_MOV_R, IDX_M_EW, IDX_VEL_GRAF, IDX_X_W, IDX_Y_P,
    IDX_BEF0, IDX_BEF1, IDX_BEF2, IDX_BEF3, IDX_BEF4, IDX_BEF5, IDX_BEF6,
    IDX_T_EOUT, IDX_Y_EOUT, IDX_T_ECT, IDX_Z_MOUT, IDX_P_INMV, IDX_M_LMV,
)
from datos_experimentales import (
    trayectoria_z, trayectoria_r,
    temp_axial_x, temp_axial_y,
    humedad_axial_x, humedad_axial_y,
    X_Exp2, Y_Exp2, X_Exp3, Y_Exp3,
    Time_exp, T_exp,
)
from utilidades_graficas import (
    configurar_estilo, crear_figura, guardar_figura,
    grafica_vs_distancia, grafica_vs_tiempo,
)


if __name__ == "__main__":
    configurar_estilo()
    start = time.process_time()

    # ================================================================
    # Ejecutar simulación
    # ================================================================

    T = 50000
    h_eu = 0.0005

    Mov_VarTemp_1 = np.zeros(T)
    T_EpTemp_1    = np.zeros(T)
    Y_p_1         = np.zeros(T)
    M_wDL_1       = np.zeros(T)

    print("Comenzando la simulación...")
    Vector = Spray(T, h_eu, 80000, Mov_VarTemp_1, T_EpTemp_1, Y_p_1, M_wDL_1)

    # ================================================================
    # Extraer resultados
    # ================================================================

    i = Vector[IDX_FINAL_I]
    Mov_Z = Vector[IDX_MOV_Z]
    T_E = Vector[IDX_T_EG]
    Y_w = Vector[IDX_Y_GRAF]
    T_R = Vector[IDX_T_R]
    Mov_R = Vector[IDX_MOV_R][0:i]

    elapsed = time.process_time() - start
    print()
    print(f"{elapsed:.2f} seconds")

    # ================================================================
    # Figure 2: Trayectoria de la gota (radial vs axial)
    # ================================================================

    fig, ax = crear_figura(figsize=(5, 7))
    ax.plot(Mov_R, Mov_Z, linewidth=0.5, color='k')
    ax.scatter(trayectoria_r, trayectoria_z, color='k', s=5)
    ax.set(xlabel="Radial distance from nozzle [m]",
           ylabel="Axial distance from nozzle [m]")
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    plt.xticks(np.arange(0, 0.7, 0.05))
    guardar_figura(fig, "Figure2.pdf")

    # ================================================================
    # Figure 1: Temperatura vs distancia axial
    # ================================================================

    fig, ax = crear_figura()
    ax.plot(Mov_Z, T_R[0:i], linewidth=0.5, color='b')
    ax.plot(Mov_Z, T_E[0:i], linewidth=0.5, color='r')
    ax.scatter(temp_axial_x, temp_axial_y, color='k', s=5)
    ax.set(xlabel="Axial distance from nozzle [m]",
           ylabel="Refined and Extract phase Temperature [°C]")
    guardar_figura(fig, "Figure1.pdf")

    # ================================================================
    # Figure 3: Humedad del aire vs distancia axial
    # ================================================================

    grafica_vs_distancia(
        Mov_Z, Y_w,
        xlabel="Axial distance from nozzle [m]",
        ylabel="Extract humidity [kg/kg]",
        nombre_pdf="Figure3.pdf",
        x_exp=humedad_axial_x, y_exp=humedad_axial_y,
    )

    # ================================================================
    # Figure 4: Contenido de humedad del refinado vs distancia axial
    # ================================================================

    X_graf = Vector[IDX_X_W]
    grafica_vs_distancia(
        Mov_Z, X_graf,
        xlabel="Axial distance from nozzle [m]",
        ylabel="Refined moisture content [kg of water/kg of maltodextrin]",
        nombre_pdf="Figure4.pdf",
    )

    # ================================================================
    # Figure 5: Diámetro de la gota vs distancia axial
    # ================================================================

    D_graf = Vector[IDX_D_DROP]
    grafica_vs_distancia(
        Mov_Z, D_graf,
        xlabel="Axial distance from nozzle [m]",
        ylabel="Droplet diameter [m]",
        nombre_pdf="Figure5.pdf",
    )

    # ================================================================
    # Figure 6: Temperatura de salida vs tiempo (con datos experimentales)
    # ================================================================

    T_Eout = Vector[IDX_T_EOUT]

    rango_limite = min(100, len(T_Eout))
    for u in range(rango_limite):
        T_Eout[u] = 75.0

    EjX = np.linspace(0, 25, len(T_Eout))

    grafica_vs_tiempo(
        EjX, T_Eout,
        xlabel="Time [s]",
        ylabel="Outlet temperature [°C]",
        nombre_pdf="Figure6.pdf",
        xtick_step=500,
        ytick_range=(70, 90, 1),
        hlines=[85],
        vlines=[80],
        x_exp=X_Exp2, y_exp=Y_Exp2,
        x_exp2=X_Exp3, y_exp2=Y_Exp3,
    )

    # ================================================================
    # Figure 7: Humedad del aire de salida vs tiempo
    # ================================================================

    Y_Eout = Vector[IDX_Y_EOUT]
    EjX = np.linspace(0, 25, len(T_Eout))

    grafica_vs_tiempo(
        EjX, Y_Eout,
        xlabel="Time [s]",
        ylabel="Humidity of outlet air [kg/kg]",
        nombre_pdf="Figure7.pdf",
        xtick_step=25,
        vlines=[80],
    )

    # ================================================================
    # Figure 8: Temperatura de entrada vs tiempo (con datos experimentales)
    # ================================================================

    Y_Eout_8 = Vector[IDX_T_ECT]
    EjX = np.linspace(0, 25, len(T_Eout))

    grafica_vs_tiempo(
        EjX, Y_Eout_8,
        xlabel="Time [s]",
        ylabel="Inlet temperature [°C]",
        nombre_pdf="Figure8.pdf",
        xtick_step=100,
        vlines=[80],
        x_exp=Time_exp, y_exp=T_exp,
    )

    # ================================================================
    # Figure 9: Contenido de humedad vs tiempo
    # ================================================================

    Z_Rout = Vector[IDX_Z_MOUT]
    EjX = np.linspace(0, 25, len(T_Eout))

    grafica_vs_tiempo(
        EjX, Z_Rout,
        xlabel="Time [s]",
        ylabel="Moisture content [kg/kg]",
        nombre_pdf="Figure9.pdf",
        xtick_step=25,
        vlines=[80],
    )

    # ================================================================
    # Figure 10: Presión de entrada del líquido vs tiempo
    # ================================================================

    P_inMV = Vector[IDX_P_INMV]
    EjX = np.linspace(0, 25, len(T_Eout))

    grafica_vs_tiempo(
        EjX, P_inMV,
        xlabel="Time [s]",
        ylabel="Inlet pressure of the liquid [Pa]",
        nombre_pdf="Figure10.pdf",
        xtick_step=25,
        vlines=[80],
    )

    # ================================================================
    # Figure 11: Flujo del líquido vs tiempo
    # ================================================================

    m_LMV = Vector[IDX_M_LMV]
    EjX = np.linspace(0, 25, len(T_Eout))

    grafica_vs_tiempo(
        EjX, m_LMV,
        xlabel="Time [s]",
        ylabel="Liquid flow [kg/h]",
        nombre_pdf="Figure11.pdf",
        xtick_step=25,
        vlines=[80],
    )

    # ================================================================
    # Verificación del balance de masa
    # ================================================================

    m_ss = (82.2 / 3600) * (1 - 0.4)

    T_E0 = 170
    rho_air0 = 1.293 * 273.15 / (273.15 + T_E0)
    m_air = 1750
    Vol_air = m_air / rho_air0

    m_da = 0.4846172128430196

    Izq = Y_w[0] * m_da + m_ss * X_graf[0]
    Der = Y_w[-1] * m_da + m_ss * X_graf[-1]

    f1 = ((0.003 / 1.003) * 1750 + 82.2 * 0.6
          - (1 - 0.003) * 1750 * Y_w[-1]
          - 0.6 * 82.2 * X_graf[-1])

    f_transf1 = 82.2 * 0.6 - 0.6 * 82.2 * X_graf[-1]
    f_transf2 = (0.003 / 1.003) * 1750 - (1 - 0.003) * 1750 * Y_w[-1]

    f2 = ((0.003 / 1.003) * 1710 + 88.2 * 0.5
          - (1 - 0.003) * 1710 * Y_w[-1]
          - 0.5 * 88.2 * X_graf[-1])
