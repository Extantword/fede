# Basado en el trabajo de Santiago González Gallego y Lina Steffania

import numpy as np
import math as mt
import pandas as pd
import matplotlib.pyplot as plt
import time
import numba
from numba.typed import List
import numba.types as nt

# Decorador de Numba para acelerar la ejecución
njit = numba.njit

start = time.process_time()

plt.rcParams["font.family"]="serif"

@njit
def Spray(T_1, h_eu_1, Part_1, Mov_VarTemp_1, T_EpTemp_1, Y_p_1, M_wDL_1):
    """
    Función de simulación para un proceso de secado por aspersión.
    Calcula la trayectoria, temperatura y humedad de las gotas a lo largo del tiempo.
    """
    # Inicialización de variables de almacenamiento

    # Inicio de solución

    T = T_1        # Número de tandas
    h_eu = h_eu_1  # Diferencial del tiempo solución de euler
    Part = Part_1  # Máximo número de iteraciones por tanda de gotas


    # Data storage

    TotalTime = T*h_eu  # Total time of simulation

    T_ECT  = List.empty_list(nt.float64)  # Lista de temperatura del aire entrando
    T_Eout = List.empty_list(nt.float64)  # Lista de temperatura del aire saliendo
    Y_Eout = List.empty_list(nt.float64)  # Lista de la humedad del aire saliendo
    Z_Mout = List.empty_list(nt.float64)  # Lista de la humedad de los sólidos saliendo
    P_inMV = List.empty_list(nt.float64)  # Presión de entrada del aire
    m_LMV  = List.empty_list(nt.float64)  # Flujo del líquido
    R_time = List.empty_list(nt.float64)  # Tiempo de Residencia de los lotes de gotas


    # Variables que describen el perfil de estado inicial del aire

    Mov_VarTemp = Mov_VarTemp_1 # Posición axial (distancia vertical)
    T_EpTemp    = T_EpTemp_1    # Perfil de temperatura del aire
    Y_p         = Y_p_1         # Perfil de humedad absoluta del aire
    M_wDL       = M_wDL_1       # Masa de agua en la fase gaseosa

    f_param = 1.0 # Parámetro de ajuste


    # Variables para el sistema de calentamiento y ducto

    T_outh = np.zeros(T)     # Temperatura de salida del calentador
    P_h = np.zeros(T)        # Presión en el calentador
    time_sim = 0   # Tiempo de simulación (renombrado para evitar conflicto con módulo time)
    m_outh = np.zeros(T)     # Flujo másico de salida del calentador
    T_wall = np.zeros(T)     # Temperatura en la pared
    m_ngas = np.zeros(T)     # Flujo de gas natural
    T_duct = np.zeros(T)     # Temperatura en el ducto entre el calentador y la torre
    m_duct = np.zeros(T)     # Flujo másico en el ducto
    Delay_j= np.zeros(2)

    # Inicialización de variables de retorno
    dummy_array = np.zeros(1)
    Bef0 = (0, dummy_array, dummy_array, dummy_array, dummy_array, dummy_array, dummy_array, dummy_array, dummy_array, 0.0)
    Bef1 = Bef0
    Bef2 = Bef0
    Bef3 = Bef0
    Bef4 = Bef0
    Bef5 = Bef0
    Bef6 = Bef0

    # Condiciones iniciales del sistema
    T_outh[0] = 25  # Inicialmente, el aire tiene esta temperatura en el calentador
    P_h[0] = 101325 # Inicialmente, el aire tiene esta presión en el calentador
    T_wall[0] = 200 # Temperatura en la pared
    g = 9.81        # Aceleración de la gravedad

    # Variable mv_flux necesita inicialización previa al loop por seguridad
    mv_flux = 0.0
    f_balanc = 0.0 # Inicialización segura por si el loop rompe antes de calcularlo

    # Parametros obtenidos experimentalmente.
    q_n = 2.09
    D_n = 70.5*10**(-6)

    GammaF = mt.gamma(1 - 1/q_n)

    SMD_exp = D_n*(GammaF**(-1))      # Diámetro promedio de una gota (experimental)

    MMD_SMD = (0.693**(1/q_n))*GammaF

    # Parametros de diseño:

    d0  = 0.711*10**(-3)     # Diametro del orificio
    Ds  = 2.4*d0             # Diametro de la cabina mezcladora
    A_p = 0.50*d0*Ds         # Area total de los puertos de entrada

    l0 = 0.1*d0        # Longitud del orificio
    Ls = 0.5*Ds        # Longitud de la porción paralela a la cabina mezcladora


    # Parámetros de la boquilla y líquido (pueden cambiar con j)

    x_R = 0.46          # Fracción de soluto
    T_R0 = 52.5         # Temperatura inicial del refinado
    mu_L = 1.8*10**(-3) # Viscosidad de la leche
    P_in = 9*10**(5)    # Presión de entrada del líquido
    P_out = 101325.     # Presión de salida
    P_L = P_in - P_out  # Caída de presión en la boquilla

    rho_M0 = 1.635 - 0.0026 * T_R0 + 2*10**(-5) * T_R0**2                                                    # Densidad de la maltodextrina
    rho_W0 = (1.0020825 - 1.14*10**(-4) * T_R0 - 3.325*10**(-6) * T_R0**2) * 1000                            # Densidad del agua
    rho_p0 = 100000.0 / (100 * (1 - x_R) * (1 / rho_M0 - 1 / (rho_W0 / 1000.0)) + 100.0 / (rho_W0 / 1000.0)) # Densidad inicial de la gota


    # Sistema de calentamiento previo a ingresar a la torre de secado

    T_Ref = 0
    T_amb = 25

    Cp_ah   = (969.542 + 6.801*(T_amb+273.15)*10**(-2) + 16.569*((T_amb+273.15)**2)*10**(-5) - 67.828*((T_amb+273.15)**3)*10**(-9))/1000    # Capacidad calorífica del aire a la temperatura ambiente
    R_h = 8.314/29       # Constante específica del gas para el aire
    Cvair=Cp_ah-R_h      # Capacidad calorífica del aire a volumen constante

    #  Términos individuales utilizados para calcular la entalpía del aire de entrada a la temperatura ambiente

    C_EntAh1 = (969.542/1000)*(T_amb - T_Ref)
    C_EntAh2 = ((6.801*10**(-2))/1000)*( ((T_amb+273.15)**2)/2 - ((T_Ref+273.15)**2)/2 )
    C_EntAh3 = ((16.569*10**(-5))/1000)*( ((T_amb+273.15)**3)/3 - ((T_Ref+273.15)**3)/3 )
    C_EntAh4 = ((-67.828*10**(-9))/1000)*( ((T_amb+273.15)**4)/4 - ((T_Ref+273.15)**4)/4 )

    H_airinh = C_EntAh1 + C_EntAh2 + C_EntAh3 + C_EntAh4  # Entalpía total del aire de entrada al calentador

    m_inh = 1710                                       # Flujo de aire de entrada
    V_h = 4                                            # Volumen del calentador
    h_comb = 48.5*1000                                 # Calor de combustión (o poder calorífico)
    P_duct = 101325                                    # Presión en el ducto
    P_oper_r = 1.5                                     # Relación de presión operativa
    C = m_inh/(P_duct*((P_oper_r*(P_oper_r-1))**0.5))  # Coeficiente de flujo calculado

    dCvdT = (6.801*10**(-2) + 2*16.569*(T_amb+273.15)*10**(-5) - 3*67.828*((T_amb+273.15)**2)*10**(-9))/1000  # Derivada de la capacidad calorífica a volumen constante con respecto a la temperatura

    # Ducto (modela el transporte de aire caliente desde el calentador hasta la entrada de la torre de secado)

    Dist_duct = 9            # Longitud del ducto
    A_duct = 0.45*0.45       # Área de la sección transversal del ducto
    Vel_duct = 0.0

    T_EC = 170.3     # Temperatura de entrada del aire de secado
    T_E0 = T_EC          # Valor inicial de referencia para la temperatura de entrada


    # Velocidad (modela la hidrodinámica interna de la boquilla atomizadora para determinar la velocidad de salida del líquido)

    K   = A_p/(Ds*d0)                               # Parámetro geométrico de la boquilla
    K_v = (0.00367*K**0.29)*(P_L*rho_p0/mu_L)**0.2  # Coeficiente de velocidad
    Vel = K_v*((2*P_L/rho_p0)**0.5)                 # Velocidad del líquido


    # Coeficientes de descarga y flujo (refina el cálculo del flujo másico real que atraviesa la boquilla)

    C1 = d0*rho_p0*Vel/mu_L
    C2 = l0/d0
    C3 = Ls/Ds
    C4 = A_p/(Ds*d0)
    C5 = Ds/d0

    C_D = 0.45*(C1**(-0.02))*(C2**(-0.03))*(C3**0.05)*(C4**0.52)*(C5**0.23) # Coeficiente de descarga de la boquilla

    A_0 = mt.pi*(d0/2)**2   # Área del orificio de la boquilla

    Param_m_L = 0.802       # Pparámetro de ajuste

    m_L = Param_m_L*C_D*A_0*rho_p0*Vel/K_v     # Flujo másico del líquido


    # Película gruesa y X (modela la estructura del flujo dentro de un atomizador de presión-torbellino)

    FNum = m_L/((rho_p0**0.5)*(P_L**0.5))                # Parámetro adimensional relacionado con la atomización

    # Ángulo del cono

    theta_2 = (16.156/2)*(K**(-0.39))*(d0**(1.13))*(mu_L**(-0.9))*(P_L**0.39)  # Ángulo relacionado con el cono de aspersión
    theta = theta_2 - 360                                                      # Ángulo del cono de aspersión ajustado

    if theta*180/mt.pi > 90:
        theta_g = 80               # Ángulo geométrico límite (fijado en 80 grados si el cálculo excede 90)
        theta = theta_g*mt.pi/180


    # Propiedades del Spray

    D_0   = 154.39        # Diámetro
    S_0   = 0.5           # Fracción de sólidos
    F_p0  = 0.021921      # Flujo de alimentación inicial
    T_p0r = 326.35        # Temperatura en Kelvin
    ParamSMD6 = 1.1938    # Parámetro añadido en el trabajo


    # Coeficientes de control (modela el ajuste dinámico del tamaño de gota en función de las condiciones operativas cambiantes)

    a_smd = 400
    b_smd = -10.1
    c_smd = -600.0
    F_pin = 89.2 / 3600.0 # Flujo de alimentación
    U_r = 0.5             # Velocidad radial
    T_p0  = T_R0 + 273.15 # Temperatura inicial en Kelvin
    S_in  = x_R           # Fracción de sóluto inicial en una gota
    MMD_SMD = 1.1         # Valor aproximado si no se calcula
    SMD = (D_0 + a_smd * (F_pin - F_p0) + b_smd * (T_p0 - T_p0r) + c_smd * (S_in - S_0)) * 10**(-6) / (MMD_SMD * 1.45) # Diámetro medio de la gota
    m_L = F_pin           # Flujo másico del líquido
    U_tan = 1.0           # Velocidad tangencial
    U_x = 7               # Velocidad axial


    # Bucle interno para la simulación de una gota

    V_ax = np.zeros(Part)  # V axial de la gota en el tiempo
    V_rad = np.zeros(Part) # V radial de la gota en el tiempo
    V_tan = np.zeros(Part) # V tangencial de la gota en el tiempo

    R_eq = np.zeros(Part)  # Posición de referencia radial respecto a spray
    h_eq = np.zeros(Part)  # Posición de referencia axial respecto a spray

    Mov_R = np.zeros(Part) # Posición radial de la gota en el tiempo
    Mov_Z = np.zeros(Part) # Posición axial de la gota en el tiempo

    M_R  = np.zeros(Part)  # Masa total
    M_Rw = np.zeros(Part)  # Masa de la gota en el tiempo
    M_E  = np.zeros(Part)  # Masa de extracto en el tiempo
    M_Ew = np.zeros(Part)  # Masa de agua en el aire
    M_Ewp = np.zeros(Part)
    M_wDL = np.zeros(Part) # Masa de agua en la capa límite

    x_w  = np.zeros(Part)
    X_w  = np.zeros(Part)  # Humedad de la gota en el tiempo
    y_w  = np.zeros(Part)
    Y_w  = np.zeros(Part)

    Y_pt  = np.zeros(Part) # Perfil de la humedad del aire en el tiempo
    Y_wDL = np.zeros(Part) # Humedad del aire en el tiempo

    T_R = np.zeros(Part)   # Temperatura de la gota en el tiempo
    T_E  = 75*np.ones(Part)# Temperatura del entorno alrededor de la gota
    T_Ep = np.zeros(Part)  # Temperatura del aire en el tiempo

    parte3 = np.zeros(Part)
    d_drop = np.zeros(Part) # Diámetro de la gota en el tiempo

    # Condiciones para una gota

    m_air = 1710.0 / 3600.0                               # Flujo másico del aire de secado en kg/s
    Mov_R[0] = 0    # Posición inicial radial
    Mov_Z[0] = 0    # Posición inicial axial

    V_rad[0] = U_r   # Velocidad radial inicial de la gota
    V_ax[0] = U_x    # Velocidad axial inicial de la gota
    V_tan[0] = U_tan # Velocidad tangencial inicial de la gota

    R_eq[0] = 0  # Posición de referencia radial inicial respecto a spray
    h_eq[0] = 0  # Posición de referencia axial inicial respecto a spray

    T_R[0] = T_R0   # Temperatura inicial de la gota
    d_drop[0] = SMD # Diametro inicial de la gota

    x_w0 = 1.0 - x_R       # Fracción de masa inicial de agua en una gota
    Y0 = 0.0021            # Humedad inicial del aire (en base seca)
    y_w0 = Y0 / (1.0 - Y0) # Humedad inicial del aire (en base líquida)

    M_R[0] = m_L*h_eu             # Masa total en la primera iteración
    M_Rw[0] = x_w0 * m_L * h_eu   # Masa del agua en la primera iteración
    M_E[0]  = (m_air)*h_eu        # Masa del extracto en la primera iteración
    M_Ew[0] = y_w0*(m_air)*h_eu   # Masa inicial de agua en el aire al comienzo de la simulación de una gota


    rho_air0 = 1.293 * 273.15 / (273.15 + T_E0)           # Densidad del aire a la temeratura de entrada
    D_eq = 2.0                                            # Diámetro de la torre en m
    Vel_g = m_air / (rho_air0 * np.pi * (D_eq / 2.0)**2)  # Velocidad promedio inicial del aire

    m = rho_p0 * np.pi * (1.0 / 6.0) * SMD**3             # Masa de inicial una gota
    Gotas = (m_L / m) * h_eu                              # Cantidad de gotas h_eu segundos

    if (M_E[0]-M_Ew[0]) != 0:
        Y_w[0]  = M_Ew[0]/(M_E[0]-M_Ew[0]) # Humedad absoluta inicial del aire
    else:
        Y_w[0] = 0.0

    # Constantes

    m_da = (1.0 - y_w0) * m_air # Masa de aire seco
    m_ss = m_L * (1.0 - x_w0)   # Masa total de maltodextrina
    m_ssi = m_ss * h_eu         # Masa de maltodextrina en un instante de tiempo
    m_ssg = m_da * h_eu         # Masa de aire seco en un instante de tiempo
    m_ssg = m * (1.0 - x_w0)    # Masa de maltodextrina en una gota

    MW = 18.0                   # Peso molecular del agua
    MA = 28.96                  # Peso molecular del aire
    R = 8314.0                  # Constante universal de los gases
    P = 101325.0                # Presión atmosférica
    sigma = 5.67*10**(-11)      # Constante de Stefan-Boltzmann
    epsil = 0.96                # Emisividad de la partícula
    Cp_M = 1.5                  # Capacidad calorífica de la maltodextrina
    Cp_wl = 4.18                # Capacidad calorífica del agua
    X_cr = 0.54                 # Humedad crítica en base seca (el punto donde se forma la costra)
    Diff = 2.5 * 5.9*10**(-9)   # Difusividad efectiva del vapor a través de la costra
    U_h = 16.75 * 0.4 / 3600.0  # Coeficiente global de transferencia de calor


    # Bucle principal de la simulación a lo largo del tiempo (tandas)

    for j in range(0, T):

        if j > 0:
            P_h[j] = P_h[j-1]
            T_outh[j] = T_outh[j-1]
            m_ngas[j] = 6

        # Componentes para calcular la entalpía del aire de salida, basados en la temperatura de salida actual (T_outh[j])
        C_EntAho1 = (969.542/1000)*(T_outh[j] - T_Ref)
        C_EntAho2 = ((6.801*10**(-2))/1000)*( ((T_outh[j]+273.15)**2)/2 - ((T_Ref+273.15)**2)/2 )
        C_EntAho3 = ((16.569*10**(-5))/1000)*( ((T_outh[j]+273.15)**3)/3 - ((T_Ref+273.15)**3)/3 )
        C_EntAho4 = ((-67.828*10**(-9))/1000)*( ((T_outh[j]+273.15)**4)/4 - ((T_Ref+273.15)**4)/4 )

        H_airout = C_EntAho1 + C_EntAho2 + C_EntAho3 + C_EntAho4   # Entalpía total del aire que sale del calentador

        Q = 0.85*m_ngas[j]*h_comb     # Tasa de calor suministrada al sistema

        m_outh[j] = C*((P_h[j]*(P_h[j]-P_duct))**0.5)      # Flujo másico de salida del calentador [kg/h]

        Massb_1 = ((T_outh[j]+273.15)*R_h*1000/V_h)*(m_inh-m_outh[j])                                 # Término relacionado con el balance de masa en el calentador

        Energb_1 = (Q + m_inh*H_airinh - m_outh[j]*H_airout)/(dCvdT*P_h[j]*V_h/(1000*R_h))            # Término relacionado con el balance de energía en el calentador

        dPdt = (Massb_1 + P_h[j]*Energb_1/(T_outh[j]+273.15))/(1 + Cvair/(dCvdT*(T_outh[j]+273.15)))  # Derivada de la presión con respecto al tiempo

        dTdt = Energb_1 - (Cvair*dPdt/(P_h[j]*dCvdT))                                                 # Derivada de la temperatura con respecto al tiempo

        if j < T-1: # Evitar índice fuera de rango
            P_h[j+1] = P_h[j]+h_eu*dPdt/3600                                                              # Presión en el calentador
            T_outh[j+1]=T_outh[j]+h_eu*dTdt/3600                                                          # Temperatura de salida del calentador
            time_sim = time_sim + h_eu                                                            # Tiempo

            if time_sim < TotalTime:

                # Parámetros del ducto

                if m_outh[j] > 10**(-9):
                    Vel_duct = m_outh[j]/(3600*1.293*((273.15)/(273.15+T_outh[j]))*A_duct)

                if j <= 2 or Vel_duct < 10**(-9):
                    Delay = 0                       # Tiempo de retardo calculado para el paso actual
                    Delayp = 0                      # Tiempo de retardo calculado para el paso anterior
                else:
                    Delay = (Dist_duct)/(h_eu*Vel_duct)
                    # Para Vel_ductp necesitamos m_outh[j-1]. Como P_h cambia, m_outh cambia.
                    # Recalculamos la velocidad previa aprox
                    Vel_ductp = m_outh[j-1]/(3600*1.293*((273.15)/(273.15+T_outh[j-1]))*A_duct)
                    if Vel_ductp > 10**(-9):
                        Delayp = (Dist_duct)/(h_eu*Vel_ductp)
                    else:
                        Delayp = 0.0

                Delay_j[0] = mt.ceil(Delayp)
                Delay_j[1] = mt.ceil(Delay)

                # Rango de posiciones en el vector de temperatura del ducto
                Pos_j1 = j - 1 + Delay_j[0]
                Pos_j2 = j + Delay_j[1]

                if Pos_j2 < T :

                    if j == 1:
                        Pos_j1=0

                    for h in range(int(Pos_j1),int(Pos_j2)):

                    # Temperatura del aire en diferentes posiciones "virtuales" dentro del ducto
                        T_duct[h] = T_outh[j-1]

                    T_duct[int(Pos_j2)] = T_outh[j]

                # Trayectoria de un único lote de gotas (j) a medida que desciende por la torre,
                # dividida en Part (1700) secciones. Cada iteración i representa un pequeño paso
                # (h_eu) en el tiempo para ese lote de gotas.


                for i in range(Part - 1):

                    if np.isnan(Mov_Z[i]):
                        break

                    # Primero se halla la iteración previa que corresponde al recorrido

                    if j == 0:

                        T_Ep[i] = 75.0
                        Y_wDL[i] = Y0

                        Vol_p0 = np.pi*((2/2)**2)*h_eu*V_ax[i]
                        m_airp0 = P*Vol_p0*MA/(8.314*1000*(T_E0+273.15))

                        M_wDL[i] = Y0*m_airp0

                        k = i

                        T_Ep[k] = T_E0
                        Y_wDL[k] = Y0

                    else:

                        # Variables temporales, se vuelven a definir al final del code
                        Recorrido = Mov_Z[i]

                        # Verificar longitud de vectores previos. Si j=0 rompió temprano, Mov_VarTemp es corto.
                        len_prev = len(Mov_VarTemp)

                        # Valores por defecto
                        Y_wDL[i] = Y_p[min(i, len_prev-1)]
                        M_wDL[i] = M_Ewp[min(i, len_prev-1)]
                        T_Ep[i]  = T_EpTemp[min(i, len_prev-1)]
                        k = 0

                        if Recorrido == 0:
                            T_Ep[i] = T_E0
                            Y_wDL[i] = Y0
                            Vol_p0 = np.pi*((2/2)**2)*h_eu*V_ax[0]
                            m_airp0 = P*Vol_p0*MA/(8.314*1000*(T_E0+273.15))
                            M_wDL[i] = Y0*m_airp0
                            k = 0

                        # Si el vector previo es demasiado corto (ej. longitud 2), usamos extrapolación segura
                        elif len_prev < 3:
                            # Usar el último valor conocido para evitar divisiones por índices extraños
                            Y_wDL[i] = Y_p[-1]
                            M_wDL[i] = M_Ewp[-1]
                            T_Ep[i]  = T_EpTemp[-1]
                            k = len_prev - 1

                        # Lógica normal si tenemos suficientes datos históricos
                        else:
                            if 0 <= Recorrido <= Mov_VarTemp[0]:
                                # Interpolación inicial
                                denom = Mov_VarTemp[1] # Asumimos que Mov_VarTemp[0] es ~0
                                if abs(denom) > 1e-9:
                                    Y_wDL[i] = (Recorrido)*(Y_p[k]-Y_p[0])/(denom) + Y_p[0]
                                    M_wDL[i] = (Recorrido)*(M_Ewp[k]-M_Ewp[0])/(denom) + M_Ewp[0]
                                    T_Ep[i] = (Recorrido)*(T_EpTemp[k]-T_EpTemp[0])/(denom) + T_EpTemp[0]
                                k = 0

                            elif Mov_VarTemp[0] <= Recorrido <= Mov_VarTemp[-1]:
                                # Búsqueda segura del índice usando np.searchsorted
                                # Encuentra índice donde Mov_VarTemp[idx] >= Recorrido
                                idx = np.searchsorted(Mov_VarTemp, Recorrido)
                                # idx será al menos 1 porque Recorrido > Mov_VarTemp[0]
                                if idx >= len_prev: idx = len_prev - 1

                                idx_prev = idx - 1

                                dist_diff = Mov_VarTemp[idx] - Mov_VarTemp[idx_prev]

                                if abs(dist_diff) > 10**(-9):
                                    ratio = (Recorrido - Mov_VarTemp[idx_prev]) / dist_diff
                                    Y_wDL[i] = ratio * (Y_p[idx] - Y_p[idx_prev]) + Y_p[idx_prev]
                                    M_wDL[i] = ratio * (M_Ewp[idx] - M_Ewp[idx_prev]) + M_Ewp[idx_prev]
                                    T_Ep[i]  = ratio * (T_EpTemp[idx] - T_EpTemp[idx_prev]) + T_EpTemp[idx_prev]
                                else:
                                    Y_wDL[i] = Y_p[idx_prev]
                                    M_wDL[i] = M_Ewp[idx_prev]
                                    T_Ep[i]  = T_EpTemp[idx_prev]

                                k = idx

                            else:
                                # Extrapolación final
                                dist_diff = Mov_VarTemp[-1] - Mov_VarTemp[-2]
                                if abs(dist_diff) > 10**(-9):
                                    ratio = (Recorrido - Mov_VarTemp[-2]) / dist_diff
                                    Y_wDL[i] = ratio * (Y_p[-1] - Y_p[-2]) + Y_p[-2]
                                    M_wDL[i] = ratio * (M_Ewp[-1] - M_Ewp[-2]) + M_Ewp[-2]
                                    T_Ep[i]  = ratio * (T_EpTemp[-1] - T_EpTemp[-2]) + T_EpTemp[-2]
                                else:
                                    Y_wDL[i] = Y_p[-1]
                                    M_wDL[i] = M_Ewp[-1]
                                    T_Ep[i]  = T_EpTemp[-1]

                                k = len_prev


                    X_w[i] = M_Rw[i]/m_ssi  # Es posible porque m_ssi:[kgSólido(SolventeR)/iter]
                    if (m_ssi + M_Rw[i]) > 0:
                        x_w[i] = M_Rw[i]/(m_ssi + M_Rw[i])
                    else:
                        x_w[i] = 0.0

                    Y_w[i] = M_Ew[i]/(m_da*h_eu) # m_da:[kgAireSeco/s]*[s/iter]

                    Delta_L = h_eu*V_ax[i]  # Distancia vertical que recorre la gota durante el paso de tiempo específico

                    if Mov_Z[i] < 2.3:
                        D_eqp = 2
                    else:
                        D_eqp = (np.tan(20*np.pi/180))*(2*np.tan(70*np.pi/180)-(2.3-Mov_Z[i]))

                    Vol = np.pi*((D_eq/2)**2)*Delta_L     # Volumen de control diferencial en la torre donde ocurre la interacción gota-aire en un paso de tiempo

                    y_w[i] = Y_wDL[i]/(1+Y_wDL[i])


                    # Propiedades del líquido y del aire

                    rho_W_calc = (1.0020825 - 1.14*10**(-4)*T_R[i] - 3.325*10**(-6)*T_R[i]**2)*1000
                    rho_W = max(500.0, rho_W_calc) # Bloqueo físico

                    lamb_v = (3.15*10**(3) - (T_R[i]+273.15)*2.38)               # Evaporación calórica específica

                    mu_air  = 1.72*10**(-5) + 4.568*(10**(-8))*T_Ep[i]                                        # Viscosidad dinámica del aire
                    rho_Air = 1.293*(273.15)/(273.15+T_Ep[i])                                                 # Densidad del aire
                    D_air   = (2.302*(10**(-5))*0.98*(10**5)/P)*( ((T_Ep[i]+T_R[i])/2 + 273.15 )/256 )**1.81  # Difusividad del vapor de agua en el aire

                    k_d    = 1.731*( 0.014 + 4.296*(T_Ep[i])*(10**(-5)))/1000                                                                                    # Conductividad del aire
                    Cp_a   = (969.542 + 6.801*(T_Ep[i]+273.15)*10**(-2) + 16.569*((T_Ep[i]+273.15)**2)*10**(-5) - 67.828*((T_Ep[i]+273.15)**3)*10**(-9))/1000    # Capacidad calorífica del aire

                    Cp_vap = (1.883 - 1.674*(T_Ep[i])*(10**(-4))+ 8.4390*((T_Ep[i])**2)*(10**(-7)) - 2.6970*((T_Ep[i])**3)*10**(-10))  # Capacidad térmica del vapor a presión constante
                    P_sat = (101325/760)*10**(7.95581-(1668.210/(T_R[i]+228)))                                                         # Presión de vapor saturado del Agua

                    y_wmol  = Y_wDL[i]*(MA/MW)/(1+Y_wDL[i]*(MA/MW))   # Cantidad de moles de maltodextrin en una porción

                    P_v   = P*y_wmol                                  # Presión parcial del agua


                    # Integraciones y entalpías

                    # Términos de entalpía para el aire y el vapor respectivamente
                    C_EntA1 = (969.542/1000)*(T_Ep[i] - T_Ref)
                    C_EntA2 = ((6.801*1e-2)/1000)*( ((T_Ep[i]+273.15)**2)/2 - ((T_Ref+273.15)**2)/2 )
                    C_EntA3 = ((16.569*1e-5)/1000)*( ((T_Ep[i]+273.15)**3)/3 - ((T_Ref+273.15)**3)/3 )
                    C_EntA4 = ((-67.828*1e-9)/1000)*( ((T_Ep[i]+273.15)**4)/4 - ((T_Ref+273.15)**4)/4 )

                    C_EntV1 = (1.883)*(T_Ep[i] - T_Ref)
                    C_EntV2 = ((-1.674*1e-4))*( ((T_Ep[i])**2)/2 - ((T_Ref)**2)/2 )
                    C_EntV3 = ((8.4390*1e-7))*( ((T_Ep[i])**3)/3 - ((T_Ref)**3)/3 )
                    C_EntV4 = ((-2.6970*1e-10))*( ((T_Ep[i])**4)/4 - ((T_Ref)**4)/4 )

                    H_a  = C_EntA1 + C_EntA2 + C_EntA3 + C_EntA4  # Entalpía específica del aire seco a la temperatura actual
                    H_vw = C_EntV1 + C_EntV2 + C_EntV3 + C_EntV4  # Entalpía específica del vapor de agua a la temperatura actual

                    # Números adimensionales

                    # Tamaño de gota (requerido para Re)

                    X0 = x_w0 / (1.0 - x_w0)                                                # Relación entre la fracción de agua y la de maltodextrina en una gota

                    # Si X_w > X_cr: La gota está en la etapa de tasa de evaporación y su diámetro se encoge

                    if X_w[i] > X_cr:

                        f = 0
                        vol_term = SMD**3 - 6*m_ssg * (X0 - X_w[i]) / (np.pi * rho_W)

                        d_drop[i] = (max(0.0, vol_term))**(1.0/3.0)
                        d_drop[i] = max(d_drop[i], 10**(-12))


                    # Si X_w <= X_cr: Se ha formado la costra y el diámetro de la partícula permanece constante (ecuas 31 y 32)

                    elif P_sat < P:

                        aw = P_v/P_sat                                                   # Actividad de agua

                        # Parámetros dependientes de la temperatura para calcular el contenido de humedad de equilibrio
                        Weq = 0.05*np.exp(-99.27/(T_R[i]+273.15))
                        Keq = 0.65*np.exp(144.57/(T_R[i]+273.15))
                        Ceq = 0.04*np.exp(1257.14/(T_R[i]+273.15))


                        ConcReq = Ceq*Keq*Weq*aw/((1-Keq*aw)*(1- Keq*aw + Ceq*Keq*aw))   # Contenido de humedad de equilibrio requerido

                        f = ((X_w[i]-ConcReq)/(X_cr-ConcReq))**(-1/3) -1                 # Factor de corrección para la velocidad de secado cuando el contenido de humedad cae por debajo del crítico

                        vol_term = SMD**3 - 6*m_ssg*(X0 - X_cr)/(np.pi*rho_W)
                        d_drop[i] = (max(0.0, vol_term))**(1.0/3.0)
                        d_drop[i] = max(d_drop[i], 10**(-12))

                    else:

                        rho_W100 = (1.0020825 - 1.14*10**(-4)*100 - 3.325*10**(-6)*100**2)*1000 # Densidad del agua líquida evaluada a 100 °C

                        vol_term = SMD**3 - 6*m_ssg*(X0 - X_cr)/(np.pi*rho_W100)

                        d_drop[i] = (max(0.0, vol_term))**(1.0/3.0)
                        d_drop[i] = max(d_drop[i], 10**(-12))

                    Vel_g = 0*m_air/(rho_Air*np.pi*((2/2)**2))          # Velocidad del gas (aire) en la torre

                    Coef_velz = 7.376 * 0.45                            # Coeficiente utilizado para perfilar la velocidad axial del gas en función de la posición radial

                    term_pos = (D_eq - Mov_R[i]) / D_eq

                    if term_pos < 0:

                        term_pos = 0.0

                    V_gz = Coef_velz * (term_pos)**2.5  # Velocidad axial
                    V_gt = 0.7 * (term_pos)**0.5        # Velocidad tangencial

                    if 0<=Mov_R[i]<=0.15:
                        if 0<=Mov_Z[i]<=0.1:
                            V_gr = -5.19                                # Velocidad radial
                        else:
                            V_gr = 0                                    # Velocidad radial
                    else:
                        V_gr = 0                                        # Velocidad radial

                    V_rel = ((V_ax[i] - V_gz)**2 + (0.0 - V_gt)**2 + (V_rad[i] - V_gr)**2)**0.5 # Velocidad relativa de las gotas y la fase gaseosa
                    Re = rho_Air * d_drop[i] * abs(V_rel) / max(mu_air, 10**(-10))       # Numero de Reynolds

                    Sc = mu_air/(rho_Air*D_air)                     # Numero de Schmidt

                    if lamb_v == 0: lamb_v = 1e-9 # Evitar div 0
                    B = Cp_vap*(T_Ep[i]-T_R[i])/lamb_v              # Numero de Spalding
                    Sh = (2 + 0.6*(Re**0.5)*Sc**(1/3))/((1+B)**0.7) # Numero de Sherwood

                    Pr = mu_air*Cp_a/k_d                            # Número de Prandlt


                    # Coeficientes

                    if d_drop[i] > 10**(-12):
                        kmass = 0.4*Sh*D_air/d_drop[i]
                    else:
                        kmass = 0.0

                    # Converción de las temperaturas a Kelvin
                    T_Ep_K = T_Ep[i] + 273.15
                    T_R_K = T_R[i] + 273.15

                    h_rad = sigma * epsil * (T_Ep_K + T_R_K) * (T_Ep_K**2 + T_R_K**2) / 1000  # Coeficiente de calor por radiación
                    h_conv = (k_d/d_drop[i])*(2 + 0.6*(Re**0.5)*(Pr**1/3))/((1+B)**0.7)       # Coeficiente de transferencia de calor convectivo
                    h_v = h_rad + h_conv                                                      # Coeficiente de transferencia de calor general

                    if P_sat < P:

                        # Términos intermedios que componen la ecuación para calcular el flux de vapor de agua desde la gota.

                        parte1 = P*MW/(R*((T_R[i]+T_Ep[i])/2 + 273.15))

                        if d_drop[i] > 10**(-12) and kmass > 10**(-12):

                            denom_flux = (d_drop[i]*(f + 2*Diff/(kmass*d_drop[i])))

                            if denom_flux != 0:

                                parte2 = 2*Diff/denom_flux

                            else:

                                parte2 = 0.0
                        else:
                            parte2 = 0.0

                        if (P - P_sat) > 0 and (P - P_v) > 0:
                            parte3[i] = np.log((P - P_v)/(P - P_sat))
                        else:
                            parte3[i] = 0.0

                        mv_flux = parte1*parte2*parte3[i]  # Flux de vapor de Agua que se transifere

                        m_evap = mv_flux*Gotas*np.pi*(d_drop[i]**2)  # Masa de agua evaporada de todas las gotas en el intervalo de tiempo actual

                    elif X_w[i] > 0:
                            m_evap = Gotas*np.pi*(d_drop[i]**2)*h_v*(T_Ep[i]-T_R[i])/lamb_v

                    else:
                        m_evap = 0.0

                    m_evap = max(0.0, m_evap)

                    # Fuerzas F_dr (Fuerza radial de la gota), F_ar (Fuerza de masa añadida radial)
                    # y Sum_Frad (suma de fuerzas radiales).


                    m_gota = m_ssg + M_Rw[i]/Gotas       # Masa de una gota

                    # Coeficientes de arrastre

                    if Re < 1*10**(-9):

                        # Si Re es cero, no hay arrastre
                        C_drag = 0

                    elif X_w[i] > X_cr:

                    # Si Re no es cero, procedemos con el cálculo original
                        if Re > 80:
                            C_drag = 0.271*Re**0.217
                        else:
                            C_drag = 27*Re**(-0.84)

                    elif Re <= 0.1:

                        a_1 = 0
                        a_2 = 24
                        a_3 = 0

                    elif Re <= 1:

                        a_1 = 3.69
                        a_2 = 22.73
                        a_3 = 0.0903

                    elif Re <= 10:

                        a_1 = 1.222
                        a_2 = 29.167
                        a_3 = -3.889

                    elif Re <= 100:

                        a_1 = 0.6167
                        a_2 = 46.5
                        a_3 = -116.667

                    elif Re <= 1000:

                        a_1 = 0.3644
                        a_2 = 98.33
                        a_3 = -2778

                    elif Re <= 5000:

                        a_1 = 0.357
                        a_2 = 148.62
                        a_3 = -4.75*10**(4)

                    elif Re <= 10000:

                        a_1 = 0.46
                        a_2 = -490.546
                        a_3 = 57.87*10**(4)

                    else:
                        a_1 = 0.5191
                        a_2 = -1662.5
                        a_3 = 5.4167*10**(6)

                    re_safe = max(Re, 10**(-6))
                    C_drag = a_1 + a_2/re_safe + a_3/(re_safe**2) # Coeficiente de arrastre de la gota


                    F_dt = (np.pi/8)*rho_Air*(d_drop[i]**2)*C_drag*(np.abs(V_gt - V_tan[i]))*(V_gt - V_tan[i])
                    F_dr = (np.pi/8)*rho_Air*(d_drop[i]**2)*C_drag*(np.abs(V_gr - V_rad[i]))*(V_gr - V_rad[i])
                    alfa = (np.pi/12)*rho_Air*(d_drop[i]**3)

                    # Fuerzas F_dz (Fuerza axial de la gota), F_az (Fuerza de masa añadida axial)
                    # y Sum_Frad (suma de fuerzas axiales).

                    F_dz = (np.pi/8)*rho_Air*(d_drop[i]**2)*C_drag*(np.abs(V_gz - V_ax[i]))*(V_gz - V_ax[i])
                    F_b  = -(np.pi/8)*rho_Air*(d_drop[i]**3)*g

                    Sum_Fax = F_dz + F_b

                    # Ecuaciones diferenciales


                    # Ecuación diferencial dV_tan/dt y método Euler:

                    dV_tan = F_dt/(m_gota + alfa)
                    V_rad[i+1]  = V_tan[i] + h_eu*dV_tan

                    # Ecuación diferencial dV_rad/dt y método Euler:

                    dV_rad = F_dr/(m_gota + alfa)
                    V_rad[i+1]  = V_rad[i] + h_eu*dV_rad

                    # Ecuación diferencial dV_ax/dt y método Euler:

                    dV_ax = (Sum_Fax/m_gota + g)/(1 + alfa/m_gota)
                    V_ax[i+1] = V_ax[i] + h_eu*dV_ax

                    # Posiciones

                    Mov_R[i+1] = Mov_R[i] + h_eu*V_rad[i]
                    Mov_Z[i+1] = Mov_Z[i] + h_eu*V_ax[i]

                    # Términos y herramientas para las ecuaciones diferenciales de balances

                    # Partes de las ecuaciones masa (corregir los últimos renglones que son de energía)

                    if Mov_Z[i] < 2.3:

                        D_eqp = 2    # Diámetro de la torre a la altura actual de la partícula; que varía si la sección es cónica
                    else:

                        D_eqp = (np.tan(20*np.pi/180))*(2*np.tan(70*np.pi/180)-(2.3-Mov_Z[i]))

                    # Usar índice k seguro
                    k_safe = min(k, len(T_EpTemp)-1)
                    rho_Airk = 1.293*(273.15)/(273.15+T_EpTemp[k_safe]) # Densidad del aire en la iteración previa

                    # Términos de flujo másico de aire en la posición actual (i) y la anterior (k)
                    f_air_i = rho_Air*Vel_g*np.pi*((D_eq/2)**2)
                    f_air_k = rho_Airk*Vel_g*np.pi*((D_eq/2)**2)

                    # Partes de las ecuaciones de energía

                    # Términos de entalpía para el aire y el vapor calculados a la temperatura de la iteración previa
                    C_EntA1p = (969.542/1000)*(T_EpTemp[k_safe] - T_Ref)
                    C_EntA2p = ((6.801*10**(-2))/1000)*( ((T_EpTemp[k_safe]+273.15)**2)/2 - ((T_Ref+273.15)**2)/2 )
                    C_EntA3p = ((16.569*10**(-5))/1000)*( ((T_EpTemp[k_safe]+273.15)**3)/3 - ((T_Ref+273.15)**3)/3 )
                    C_EntA4p = ((-67.828*10**(-9))/1000)*( ((T_EpTemp[k_safe]+273.15)**4)/4 - ((T_Ref+273.15)**4)/4 )

                    C_EntV1p = (1.883)*(T_EpTemp[k_safe] - T_Ref)
                    C_EntV2p = ((-1.674*10**(-4)))*( ((T_EpTemp[k_safe])**2)/2 - ((T_Ref)**2)/2 )
                    C_EntV3p = ((8.4390*10**(-7)))*( ((T_EpTemp[k_safe])**3)/3 - ((T_Ref)**3)/3 )
                    C_EntV4p = ((-2.6970*10**(-10)))*( ((T_EpTemp[k_safe])**4)/4 - ((T_Ref)**4)/4 )

                    H_app  = C_EntA1p + C_EntA2p + C_EntA3p + C_EntA4p  # Entalpía del aire en el paso previo
                    H_vwpp = C_EntV1p + C_EntV2p + C_EntV3p + C_EntV4p  # Entalpía del vapor de agua en el paso previo

                    # Términos de entalpía que representan la energía del agua que se evapora
                    C_Rv1 = (1.883)*(T_Ep[i] - T_R[i])
                    C_Rv2 = ((-1.674*10**(-4)))*( ((T_Ep[i])**2)/2 - ((T_R[i])**2)/2 )
                    C_Rv3 = ((8.4390*10**(-7)))*( ((T_Ep[i])**3)/3 - ((T_R[i])**3)/3 )
                    C_Rv4 = ((-2.6970*10**(-10)))*( ((T_Ep[i])**4)/4 - ((T_R[i])**4)/4 )

                    H_wvevap = C_Rv1 + C_Rv2 + C_Rv3 + C_Rv4  # Entalpía del agua que se evapora

                    # En este caso se usan las tandas previas con iteración próxima
                    # Es decir, pp (previa, próxima)

                    lamb_v0 = (3.15*10**(3) - (T_Ref + 273.15)*2.38)  # Calor latente de vaporización del agua evaluado a 0 °C

                    H_p = H_app + (H_vwpp + lamb_v0)*Y_p[k_safe]  # Entalpía total de la mezcla de gas en la iteración previa
                    H   = H_a  + (H_vw + lamb_v0)*Y_wDL[i]   # Entalpía total de la mezcla de gas en la iteración actual

                    # Ecuaciones diferenciales del balance de masa

                    dM_Rwdt = -m_evap                                           # Ecuación 5
                    dM_Ewdt    = (f_air_k*Y_p[k_safe] - f_air_i*Y_wDL[i] + m_evap )

                    M_Rw[i+1] = max(0.0, M_Rw[i] + h_eu*( dM_Rwdt ))

                    RemainW = 0
                    RemainW = RemainW + m_evap    # Variable acumuladora que rastrea el agua total evaporada o remanente en el proceso

                    m_airp = P*Vol*MA/(8.314*1000*(T_Ep[i]+273.15)) # Moles de aire en el volumen de control actual
                    M_Ew[i] = Y_wDL[i]*m_airp + h_eu*( dM_Ewdt )


                    # Ecuaciones diferenciales del balance de energía

                    denTR = M_Rw[i]*Cp_wl + (m_ssi)*Cp_M

                    # Capacidad calorífica a volumen constante del aire y del vapor de agua, respectivamente
                    Cv_air = Cp_a - 8.314/MA
                    Cv_wv  = Cp_vap - 8.314/MW

                    dCv_wvdT = (2.558*10**(-12))*(T_Ep[i]**3) + (-4.681*10**(-9))*(T_Ep[i]**2) + (2.615*10**(-6))*T_Ep[i] + 0.0001957
                    dCv_airdT = (9.785*10**(-6))*(T_Ep[i]**(0.5526)) -1.634*10**(-5)


                    denTE = ((m_airp*(T_Ep[i]+273.15)*dCv_airdT + Y_wDL[i]*m_airp*(T_Ep[i]+273.15)*dCv_wvdT )+(m_airp*Cv_air + Y_wDL[i]*m_airp*Cv_wv)) # Inercia térmica del gas

                    NumTR = -m_evap*(Cp_wl*(T_R[i]-T_Ref) + lamb_v) + Gotas*np.pi*(d_drop[i]**2)*h_v*(T_Ep[i]-T_R[i]) - Cp_M*(T_R[i]+273.15)*dM_Rwdt   # Fase refinada

                    T_Edif = Cv_wv*(T_Ep[i]+273.15)*dM_Ewdt # Cambio de energía asociado al diferencial de masa de agua

                    NumTE = f_air_k*H_p - f_air_i*H + m_evap*H_wvevap - Gotas*np.pi*(d_drop[i]**2)*h_v*(T_Ep[i]-T_R[i]) - (np.pi*D_eqp*Delta_L)*U_h*(T_Ep[i]-25) - T_Edif

                    if abs(denTR) > 10**(-9):
                        dT_Rdt = NumTR/denTR         # Ecuación 8
                    else:
                        dT_Rdt = 0.0

                    if abs(denTE) > 10**(-9):
                        dT_Edt = NumTE/denTE
                    else:
                        dT_Edt = 0.0

                    # Solución con el método de Euler:

                    T_R[i+1]  = T_R[i] + h_eu*( dT_Rdt )

                    T_E[i] = T_Ep[i] + h_eu*( dT_Edt )

                    Y_pt[i] = M_Ew[i]/(P*Vol*MA/(8.314*1000*(T_E[i]+273.15)))

                    if Mov_Z[i] > 3.5 or Mov_R[i] >= D_eqp/2.0:

                        if j == 0:

                            print()
                            print("Se alcanzó la altura del equipo")
                            print("La última iteración fue la",i)
                            print("El diferencial de tiempo para las iteraciones es de:",h_eu,"s")


                        if j == (T-1):

                            print()
                            print("Se alcanzó la altura del equipo")
                            print("La última iteración fue la",i)
                            print("El diferencial de tiempo para las iteraciones es de:",h_eu,"s")


                        T_Eg = T_Ep[0:i]

                        T_EpTemp = T_E[0:i]


                        if j % 250 == 0:

                            f_balanc = ( (Y0/(1+Y0))*m_air + m_L*(1-x_R) - (1-Y0)*m_air*Y_wDL[i] - (x_R*m_L)*X_w[i] )*3600 # Variable utilizada para verificar el cierre del balance de masa global del sistema

                            if abs(f_balanc) > 1 :

                                if 0.6 < f_param < 1.4:

                                    if f_balanc < 0:

                                        f_param = f_param + 0.001

                                    else:

                                        f_param = f_param - 0.001

                        Vel_gF = m_air/(rho_air0*np.pi*((2/2)**2))     # Velocidad del gas

                        Mov_VarTemp = Mov_Z[0:i] + f_param*Vel_gF*h_eu # Almacena la posición o movimiento temporal de las parcelas de gas para interpolación de estados previos

                        M_Ewp = M_Ew[0:i]     # Almacena la masa de agua en el aire de los pasos previos

                        Y_graf = Y_wDL[0:i]   # Almacena el perfil de humedad del aire con fines de graficación

                        Vel_graf = V_ax[0:i]  # Almacena el perfil de velocidad axial de la gota para graficación

                        Y_p = Y_pt[0:i]


                        if j % 1000 == 0:

                            T_ECT.append(T_EC)
                            T_Eout.append(T_Eg[-1])
                            Y_Eout.append(Y_graf[-1])
                            Z_Mout.append(X_w[i])

                            R_time.append(h_eu*i)

                            P_inMV.append(P_in)
                            m_LMV.append(m_L)


                        # Los Befn almacenan datos del estado de la simulación (diámetro, posición, temperaturas, humedades) en momentos específicos del tiempo (iteraciones j = 1000, 2000, etc.)

                        if j == 1000: #17000:

                            Bef0 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        if j == 2000: #18000:

                            Bef1 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        if j == 3000: #19000:

                            Bef2 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        if j == 4000: #20000:

                            Bef3 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        if j == 15000: #21000:

                            Bef4 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        if j == 20000: #22000:

                            Bef5 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        if j == 30000: #28000:

                            Bef6 = (i,d_drop[0:i],Mov_Z[0:i],T_Eg,Y_graf,T_R[0:i],Mov_R,M_Ew,X_w[0:i], mv_flux)

                        break

            else:
                break



    return (parte3[0:i], d_drop[0:i], i, Mov_Z[0:i], T_Eg, Y_graf, T_R[0:i], V_gz, Mov_R, M_Ew,  # llega al 9
            Vel_graf, X_w[0:i], Y_p, mv_flux, Vel_g, Vel_gF,
            Bef0,Bef1,Bef2,Bef3,Bef4,Bef5,Bef6,T_Eout,Y_Eout,T_ECT,Z_Mout,
            P_inMV,m_LMV, RemainW, f_param, f_balanc, R_time)

# Bloque de ejecución y análisis de resultados

if __name__ == "__main__":
    # Parámetros iniciales

    T = 50000
    h_eu = 0.0005

    # Vectores iniciales para la primera iteración
    Mov_VarTemp_1 = np.zeros(T)
    T_EpTemp_1    = np.zeros(T)
    Y_p_1         = np.zeros(T)
    M_wDL_1       = np.zeros(T)

    print("Comenzando la simulación...")

    # Ejecutamos la función
    Vector = Spray(T, h_eu, 80000, Mov_VarTemp_1, T_EpTemp_1, Y_p_1, M_wDL_1)

    i = Vector[2]
    parte3 = Vector[1]

    Mov_Z = Vector[3]
    T_E = Vector[4]
    Y_w = Vector[5]
    T_R = Vector[6]

    Vel_g = Vector[7]

    Mov_R = Vector[8][0:Vector[2]]

    Mov_ZR = Mov_Z
    Mov_ZE = Mov_Z

    Bef = Vector[16]
    Bef1 = Vector[17]
    Bef2 = Vector[18]
    Bef3 = Vector[19]
    Bef4 = Vector[20]
    Bef5 = Vector[21]
    Bef6 = Vector[22]

    #%%

    #TheLast = Vector[-1][0:Vector[2]]
    #%%

    # print()
    # print("Se alcanzó la altura del equipo")
    # print("La última iteración fue la",i)
    # print("El diferencial de tiempo para las iteraciones es de:",h_eu,"s")
    # print("El tiempo de residencia es de:",h_eu*i,"s")

    elapsed = (time.process_time() - start)

    print()
    print(elapsed, "seconds")

    #%%

    x_expz = [0,0.2,0.2,0.6,1.0,1.40,2.6]
    y_expr = [0,0,0.29,0.29,0.29,0.29,0.0]

    fig, ax = plt.subplots(1, figsize=(5,7))

    ax.plot(Mov_R, Mov_Z,linewidth=0.5,color='k') # Datos de los ejes y color
    ax.scatter(y_expr,x_expz, color='k',s=5)

    ax.set(xlabel="Radial distance from nozzle [m]",
            ylabel="Axial distance from nozzle [m]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    #ax.set_xlim([0, (mt.ceil(Mov_R[-1]))/2 ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    #plt.xticks(np.arange(0, mt.ceil(Mov_R[-1])/2, 0.05))
    plt.xticks(np.arange(0, 0.7, 0.05))
    #plt.axis('scaled')

    fig.savefig("Figure2.pdf", dpi=600)
    #plt.show()



    #%%

    # x_exp = [0,0.2,0.6,1.00,1.40,2.3]
    # y_exp = [195,143.675, 113.233, 121.858, 113.417,113]

    x_exp = [0,0.2,0.6,1.0,1.40,2.3]
    y_exp = [195,175, 113.233, 121.858, 113.417,113]

    fig, ax = plt.subplots(1, figsize=(10,5))





    # ax.plot(Bef[2],Bef[5],linewidth=0.5,color='b')
    # ax.plot(Bef[2],Bef[3],linewidth=0.5,color='r')

    # ax.plot(Bef1[2],Bef1[5],linewidth=0.5,color='b')
    # ax.plot(Bef1[2],Bef1[3],linewidth=0.5,color='r')

    # ax.plot(Bef2[2],Bef2[5],linewidth=0.5,color='b')
    # ax.plot(Bef2[2],Bef2[3],linewidth=0.5,color='r')

    # ax.plot(Bef3[2],Bef3[5],linewidth=0.5,color='b')
    # ax.plot(Bef3[2],Bef3[3],linewidth=0.5,color='r')

    # ax.plot(Bef4[2],Bef4[5],linewidth=0.5,color='b')
    # ax.plot(Bef4[2],Bef4[3],linewidth=0.5,color='r')

    # ax.plot(Bef5[2],Bef5[5],linewidth=0.5,color='b')
    # ax.plot(Bef5[2],Bef5[3],linewidth=0.5,color='r')

    # ax.plot(Bef5[2],Bef5[5],linewidth=0.5,color='b')
    # ax.plot(Bef5[2],Bef5[3],linewidth=0.5,color='r')





    ax.plot(Mov_ZR,T_R[0:i],linewidth=0.5,color='b') # Datos de los ejes y color
    #ax.plot(Mov_Epgraf,T_Egraf,linewidth=0.5,color='r') # Datos de los ejes y color
    #ax.plot(Mov_Epgraf,T_E[0:i],linewidth=0.5,color='k') # Datos de los ejes y color
    ax.plot(Mov_ZE,T_E[0:i],linewidth=0.5,color='r') # Datos de los ejes y color
    # ax.plot(Mov_Zs[-2],T_Rs[-2],linewidth=0.5,color='r') # Datos de los ejes y c
    # ax.plot(Mov_Zs[-2],T_Es[-2],linewidth=0.5,color='r') # Datos de los ejes y color






    ax.scatter(x_exp,y_exp, color='k',s=5)

    ax.set(xlabel="Axial distance from nozzle [m]",
            ylabel="Refined and Extract phase Temperature [°C]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    #ax.set_xlim([0, (mt.ceil(Mov_R[-1]))/2 ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    #plt.xticks(np.arange(0, mt.ceil(Mov_R[-1])/2, 0.05))
    #plt.axis('scaled')

    fig.savefig("Figure1.pdf", dpi=600)
    #plt.show()




    # # Falta:
    # #     Añadir la sección de gas nuevo en cada tanda.
    # #       - Cálcular la distancia recorrida por el gas nuevo.       x
    # #       - Hallar la distancia a la que llegaría dicho gas.        x
    # #       - Expresarlo al inicio de los vectores de solución.       x
    # #       - Verificar que todo el vector de distancias se traslade. x
    # #       - Verificar solución para otros casos, es decir, al variar la cantidad de gas...
    # #     Añadir las ecuaciones de energía.                           x
    # #     Cambiar la masa de una gota en las ecuaciones de fuerza.    x
    # #     Hacer las gráficas correctamente.

    #%%

    x_expH = [0,0.6,1.0,1.40]
    y_expH = [0.010, 0.028, 0.034, 0.035]

    Y_graf = Vector[5]

    fig, ax = plt.subplots(1, figsize=(10,5))

    #ax.plot(Mov_ZR,T_R[0:i],linewidth=0.5,color='b') # Datos de los ejes y color

    ax.plot(Mov_ZE,Y_graf,linewidth=0.5,color='r') # Datos de los ejes y color

    ax.scatter(x_expH,y_expH, color='k',s=5)

    #ax.scatter(x_exp,y_exp, color='k',s=5)

    ax.set(xlabel="Axial distance from nozzle [m]",
            ylabel="Extract humidity [kg/kg]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    #ax.set_xlim([0, (mt.ceil(Mov_R[-1]))/2 ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    #plt.xticks(np.arange(0, mt.ceil(Mov_R[-1])/2, 0.05))
    #plt.axis('scaled')

    fig.savefig("Figure3.pdf", dpi=600)
    #plt.show()

    #%%

    X_graf = Vector[11]

    fig, ax = plt.subplots(1, figsize=(10,5))

    #ax.plot(Mov_ZR,T_R[0:i],linewidth=0.5,color='b') # Datos de los ejes y color

    ax.plot(Mov_ZE,X_graf,linewidth=0.5,color='r') # Datos de los ejes y color


    #ax.scatter(x_exp,y_exp, color='k',s=5)

    ax.set(xlabel="Axial distance from nozzle [m]",
            ylabel="Refined moisture content [kg of water/kg of maltodextrin]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    #ax.set_xlim([0, (mt.ceil(Mov_R[-1]))/2 ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    #plt.xticks(np.arange(0, mt.ceil(Mov_R[-1])/2, 0.05))
    #plt.axis('scaled')

    fig.savefig("Figure4.pdf", dpi=600)
    #plt.show()

    #%%

    D_graf = Vector[1]

    fig, ax = plt.subplots(1, figsize=(10,5))

    #ax.plot(Mov_ZR,T_R[0:i],linewidth=0.5,color='b') # Datos de los ejes y color

    ax.plot(Mov_ZE,D_graf,linewidth=0.5,color='r') # Datos de los ejes y color


    #ax.scatter(x_exp,y_exp, color='k',s=5)

    ax.set(xlabel="Axial distance from nozzle [m]",
            ylabel="Droplet diameter [m]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    #ax.set_xlim([0, (mt.ceil(Mov_R[-1]))/2 ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    #plt.xticks(np.arange(0, mt.ceil(Mov_R[-1])/2, 0.05))
    #plt.axis('scaled')

    fig.savefig("Figure5.pdf", dpi=600)
    #plt.show()

    #%%




    T_Eout = Vector[23]

    rango_limite = min(100, len(T_Eout)) 
    
    for u in range(rango_limite):
        T_Eout[u] = 75.0

    EjX = np.linspace(0, 25, len(T_Eout))

    fig, ax = plt.subplots(1, figsize=(10,5))

    ax.plot(EjX,T_Eout,linewidth=0.5,color='b') # Datos de los ejes y color

    ax.set(xlabel="Time [s]",
            ylabel="Outlet temperature [°C]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    ax.set_xlim([0, mt.ceil(EjX[-1]) ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    plt.yticks(np.arange(70, 90, 1))
    plt.xticks(np.arange(0, mt.ceil(EjX[-1]), 500))
    #plt.axis('scaled')

    #l1=plt.axhline(113,linewidth=0.5,color='black',ls='--')
    #l1.set_label('T1')

    l2=plt.axhline(85,linewidth=0.5,color='black',ls='--')
    l2.set_label('T2')


    X_Exp2 = [250.068,
    301.188,
    318.684,
    326.172,
    330.564,
    338.88,
    348.096,
    355.116,
    370.884,
    378.336,
    387.516,
    401.088,
    423.408,
    434.784,
    447.456,
    461.46,
    485.94,
    511.752,
    546.276,
    564.204,
    600.924,
    631.524,
    662.124,
    698.412,
    742.116,
    770.088,
    785.388,
    797.628,
    826.5,
    868.872,
    901.236,
    929.208,
    965.064,
    1000.452,
    1025.796,
    1075.62,
    1111.44,
    1148.196,
    1199.712,
    1233.408,
    1272.72,
    1315.128,
    1379.352,
    1444.512,
    1479,
    1530.588,
    1566.408,
    1610.544,
    1655.148,
    1692.3,
    1725.06]

    X_Exp2 = [400.136,
    451.256,
    468.752,
    476.24,
    480.632,
    488.948,
    498.164,
    505.184,
    520.952,
    528.404,
    537.584,
    551.156,
    573.476,
    584.852,
    597.524,
    611.528,
    636.008,
    661.82,
    696.344,
    714.272,
    750.992,
    781.592,
    812.192,
    848.48,
    892.184,
    920.156,
    935.456,
    947.696,
    976.568,
    1018.94,
    1051.304,
    1079.276,
    1115.132,
    1150.52,
    1175.864,
    1225.688,
    1261.508,
    1298.264,
    1349.78,
    1383.476,
    1422.788,
    1465.196,
    1529.42,
    1594.58,
    1629.068,
    1680.656,
    1716.476,
    1760.612,
    1805.216,
    1842.368,
    1875.128,
    1917.968,
    1954.256,
    1987.448,
    2045.12,
    2078.78,
    2115.068,
    2165.324,
    2205.536,
    2250.968,
    2286.392,
    2357.168,
    2392.556,
    2456.384,
    2510.564,
    2541.164,
    2610.644,
    2657.84,
    2699.384,
    2753.132,
    2801.192,
    2868.944,
    2908.256]

    Y_Exp2 = [81.5272,
    81.5487,
    81.4156,
    81.1238,
    80.8307,
    80.6007,
    80.3453,
    80.0656,
    79.8276,
    79.439,
    79.2574,
    79.0665,
    78.9145,
    78.7774,
    78.647,
    78.499,
    78.3242,
    78.2462,
    78.14,
    77.9867,
    77.8912,
    77.8347,
    77.711,
    77.6814,
    77.6437,
    77.5939,
    77.5106,
    77.4057,
    77.325,
    77.2497,
    77.2053,
    77.1219,
    77.0681,
    77.0546,
    76.9484,
    76.9282,
    76.9954,
    76.8071,
    76.9967,
    76.8124,
    76.8501,
    76.721,
    76.7881,
    76.5609,
    76.6644,
    76.5743,
    76.8203,
    76.6307,
    76.5123,
    76.4854,
    76.671]

    Y_Exp2 = [81.5272,
    81.5487,
    81.4156,
    81.1238,
    80.8307,
    80.6007,
    80.3453,
    80.0656,
    79.8276,
    79.439,
    79.2574,
    79.0665,
    78.9145,
    78.7774,
    78.647,
    78.499,
    78.3242,
    78.2462,
    78.14,
    77.9867,
    77.8912,
    77.8347,
    77.711,
    77.6814,
    77.6437,
    77.5939,
    77.5106,
    77.4057,
    77.325,
    77.2497,
    77.2053,
    77.1219,
    77.0681,
    77.0546,
    76.9484,
    76.9282,
    76.9954,
    76.8071,
    76.9967,
    76.8124,
    76.8501,
    76.721,
    76.7881,
    76.5609,
    76.6644,
    76.5743,
    76.8203,
    76.6307,
    76.5123,
    76.4854,
    76.671,
    76.5015,
    76.3536,
    76.3146,
    76.4261,
    76.3293,
    76.3925,
    76.3817,
    76.301,
    76.4408,
    76.2041,
    76.3211,
    76.4138,
    76.2807,
    76.2363,
    76.1784,
    76.4016,
    76.2133,
    76.0842,
    76.1608,
    76.0479,
    76.1124,
    76.2602]



    X_Exp3 = [1949.456,
    1991.864,
    2018.504,
    2030.276,
    2035.928,
    2047.268,
    2051.156,
    2053.316,
    2060.696,
    2065.916,
    2075.06,
    2078.516,
    2095.076,
    2101.628,
    2103.32,
    2118.152,
    2123.372,
    2140.832,
    2150.876,
    2167.004,
    2180.54,
    2195.408,
    2199.728,
    2216.756,
    2238.176,
    2266.112,
    2291.888,
    2303.696,
    2317.664,
    2325.08,
    2360.468,
    2367.848,
    2383.58,
    2413.712,
    2429.876,
    2439.488,
    2446.904,
    2464.796,
    2492.336,
    2522.036,
    2539.532]

    X_Exp3 = [2449.456,
    2491.864,
    2518.504,
    2530.276,
    2535.928,
    2547.268,
    2551.156,
    2553.316,
    2560.696,
    2565.916,
    2575.06,
    2578.516,
    2595.076,
    2601.628,
    2603.32,
    2618.152,
    2623.372,
    2640.832,
    2650.876,
    2667.004,
    2680.54,
    2695.408,
    2699.728,
    2716.756,
    2738.176,
    2766.112,
    2791.888,
    2803.696,
    2817.664,
    2825.08,
    2860.468,
    2867.848,
    2883.58,
    2913.712,
    2929.876,
    2939.488,
    2946.904,
    2964.796,
    2992.336,
    3022.036,
    3039.532,
    3064.876,
    3076.648,
    3098.5,
    3118.156,
    3151.78,
    3171.004,
    3193.72,
    3213.376,
    3235.228,
    3267.556,
    3285.016,
    3321.736,
    3367.6,
    3407.812,
    3442.3,
    3462.856,
    3488.2,
    3544.144,
    3578.236,
    3600.052,
    3635.44,
    3684.4,
    3739.012,
    3764.752,
    3809.356,
    3855.22,
    3905.044,
    3969.268,
    4006.852,
    4068.916,
    4130.512,
    4179.904,
    4235.416,
    4291.756,
    4345.504,
    4390.936,
    4445.584,
    4494.94]

    Y_Exp3 = [76.0135,
    75.9328,
    76.1009,
    76.3496,
    76.7624,
    77.0824,
    77.3177,
    77.5584,
    77.9457,
    78.2105,
    78.7618,
    79.1639,
    79.3857,
    79.6506,
    79.9827,
    80.21,
    80.5246,
    80.6725,
    80.8164,
    81.0301,
    81.1767,
    81.3596,
    81.5854,
    81.7696,
    81.9149,
    81.9834,
    82.0842,
    82.2214,
    82.3895,
    82.5132,
    82.6153,
    82.77,
    82.8883,
    83.0079,
    83.1383,
    83.2634,
    83.4368,
    83.5444,
    83.5847,
    83.5215,
    83.4233]

    Y_Exp3 = [76.0135,
    75.9328,
    76.1009,
    76.3496,
    76.7624,
    77.0824,
    77.3177,
    77.5584,
    77.9457,
    78.2105,
    78.7618,
    79.1639,
    79.3857,
    79.6506,
    79.9827,
    80.21,
    80.5246,
    80.6725,
    80.8164,
    81.0301,
    81.1767,
    81.3596,
    81.5854,
    81.7696,
    81.9149,
    81.9834,
    82.0842,
    82.2214,
    82.3895,
    82.5132,
    82.6153,
    82.77,
    82.8883,
    83.0079,
    83.1383,
    83.2634,
    83.4368,
    83.5444,
    83.5847,
    83.5215,
    83.4233,
    83.6035,
    83.7796,
    83.9087,
    84.0216,
    84.1157,
    84.2381,
    84.3295,
    84.4169,
    84.4935,
    84.4491,
    84.6212,
    84.6737,
    84.7395,
    84.7139,
    84.8524,
    84.8954,
    84.999,
    84.8793,
    84.8416,
    85.0769,
    85.2745,
    85.1615,
    85.3242,
    85.4519,
    85.3484,
    85.4304,
    85.5715,
    85.4599,
    85.6037,
    85.6736,
    85.6494,
    85.7408,
    85.6789,
    85.7192,
    85.7985,
    85.906,
    86.019,
    85.9638]


    ax.scatter(X_Exp2,Y_Exp2, color='k',s=5)
    ax.scatter(X_Exp3,Y_Exp3, color='k',s=5)

    plt.axvline(x=80,linewidth=0.5,color='red',linestyle='-')

    #plt.axvline(x=200,linewidth=0.5,color='red',linestyle='-')

    fig.savefig("Figure6.pdf", dpi=600)
    #plt.show()


    #%%

    Y_Eout = Vector[24]
    EjX = np.linspace(0, 25, len(T_Eout))

    fig, ax = plt.subplots(1, figsize=(10,5))

    ax.plot(EjX,Y_Eout,linewidth=0.5,color='b') # Datos de los ejes y color

    ax.set(xlabel="Time [s]",
            ylabel="Humidity of outlet air [kg/kg]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    ax.set_xlim([0, mt.ceil(EjX[-1]) ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    plt.xticks(np.arange(0, mt.ceil(EjX[-1]), 25))
    #plt.axis('scaled')

    # l1=plt.axhline(113,linewidth=0.5,color='black',ls='--')
    # l1.set_label('T1')

    # l2=plt.axhline(85,linewidth=0.5,color='black',ls='--')
    # l2.set_label('T2')

    plt.axvline(x=80,linewidth=0.5,color='red',linestyle='-')
    #plt.axvline(x=200,linewidth=0.5,color='red',linestyle='-')

    fig.savefig("Figure7.pdf", dpi=600)
    #plt.show()


    #%%

    Time_exp = [143.800000000000,199.060000000000,245.968000000000,267.244000000000,300.004000000000,313.792000000000,318.364000000000,330.964000000000,337.336000000000,355.732000000000,363.868000000000	,389.284000000000,426.544000000000,476.872000000000,586.996000000000,643.408000000000,689.416000000000]

    T_exp = [170.6506,
    170.3273,
    169.8626,
    169.4455,
    169.128,
    168.4857,
    167.6659,
    166.1259,
    165.0751,
    163.4095,
    162.2563,
    160.8952,
    160.0451,
    160.2126,
    159.5228,
    160.3326,
    160.0007]

    Y_Eout = Vector[25]
    EjX = np.linspace(0, 25, len(T_Eout))

    fig, ax = plt.subplots(1, figsize=(10,5))

    ax.plot(EjX,Y_Eout,linewidth=0.5,color='b') # Datos de los ejes y color

    ax.scatter(Time_exp,T_exp, color='k',s=5)

    ax.set(xlabel="Time [s]",
            ylabel="Inlet temperature [°C]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    ax.set_xlim([0, mt.ceil(EjX[-1]) ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    plt.xticks(np.arange(0, mt.ceil(EjX[-1]), 100))
    #plt.axis('scaled')

    # l1=plt.axhline(113,linewidth=0.5,color='black',ls='--')
    # l1.set_label('T1')

    # l2=plt.axhline(85,linewidth=0.5,color='black',ls='--')
    # l2.set_label('T2')

    plt.axvline(x=80,linewidth=0.5,color='red',linestyle='-')
    #plt.axvline(x=200,linewidth=0.5,color='red',linestyle='-')

    fig.savefig("Figure8.pdf", dpi=600)
    #plt.show()

    #%%

    Z_Rout = Vector[26]
    EjX = np.linspace(0, 25, len(T_Eout))

    fig, ax = plt.subplots(1, figsize=(10,5))

    ax.plot(EjX,Z_Rout,linewidth=0.5,color='b') # Datos de los ejes y color

    ax.set(xlabel="Time [s]",
            ylabel="Moisture content [kg/kg]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    ax.set_xlim([0, mt.ceil(EjX[-1]) ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    plt.xticks(np.arange(0, mt.ceil(EjX[-1]), 25))
    #plt.axis('scaled')

    # l1=plt.axhline(113,linewidth=0.5,color='black',ls='--')
    # l1.set_label('T1')

    # l2=plt.axhline(85,linewidth=0.5,color='black',ls='--')
    # l2.set_label('T2')

    plt.axvline(x=80,linewidth=0.5,color='red',linestyle='-')
    #plt.axvline(x=200,linewidth=0.5,color='red',linestyle='-')

    fig.savefig("Figure9.pdf", dpi=600)
    #plt.show()

    #%%

    P_inMV = Vector[27]
    EjX = np.linspace(0, 25, len(T_Eout))

    fig, ax = plt.subplots(1, figsize=(10,5))

    ax.plot(EjX,P_inMV,linewidth=0.5,color='b') # Datos de los ejes y color

    ax.set(xlabel="Time [s]",
            ylabel="Inlet pressure of the liquid [Pa]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    ax.set_xlim([0, mt.ceil(EjX[-1]) ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    plt.xticks(np.arange(0, mt.ceil(EjX[-1]), 25))
    #plt.axis('scaled')

    # l1=plt.axhline(113,linewidth=0.5,color='black',ls='--')
    # l1.set_label('T1')

    # l2=plt.axhline(85,linewidth=0.5,color='black',ls='--')
    # l2.set_label('T2')

    plt.axvline(x=80,linewidth=0.5,color='red',linestyle='-')
    #plt.axvline(x=200,linewidth=0.5,color='red',linestyle='-')

    fig.savefig("Figure10.pdf", dpi=600)
    #plt.show()

    #%%

    m_LMV = Vector[28]
    EjX = np.linspace(0, 25, len(T_Eout))

    fig, ax = plt.subplots(1, figsize=(10,5))

    ax.plot(EjX,m_LMV,linewidth=0.5,color='b') # Datos de los ejes y color

    ax.set(xlabel="Time [s]",
            ylabel="Liquid flow [kg/h]")
            #title='Recorrido de la partícula')

    ax.grid()
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    #plt.axes().set_aspect('equal')  # Hace que las celdas tengan las mismas proporciones

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5) # This line is now indented


    ax.set_xlim([0, mt.ceil(EjX[-1]) ])
    #ax.set_ylim([0, mt.ceil(Mov_Z[-1])])
    #plt.gca().invert_yaxis()                       # Invierte el eje y

    # Límites de los ejes:

    #plt.yticks(np.arange(0, mt.ceil(Mov_Z[-1]), 0.25))
    plt.xticks(np.arange(0, mt.ceil(EjX[-1]), 25))
    #plt.axis('scaled')

    # l1=plt.axhline(113,linewidth=0.5,color='black',ls='--')
    # l1.set_label('T1')

    # l2=plt.axhline(85,linewidth=0.5,color='black',ls='--')
    # l2.set_label('T2')

    plt.axvline(x=80,linewidth=0.5,color='red',linestyle='-')
    #plt.axvline(x=200,linewidth=0.5,color='red',linestyle='-')

    fig.savefig("Figure11.pdf", dpi=600)
    #plt.show()







    #%% Cálculo de verificación:

    #m_ss = 0.005912783767187386
    #m_da = 0.3144037221752935

    m_ss = (82.2/3600)*(1-0.4)

    T_E0 = 170
    rho_air0 = 1.293*(273.15)/(273.15+T_E0) # [kg/m^3]

    m_air = 1750  # kg/h
    Vol_air = m_air/rho_air0                 # [m^3/s] ... (1750(kg/h)/rho_air0(kg/m^3))/3600

    X_graf = Vector[11]

    m_da = 0.4846172128430196

    Izq = Y_w[0]*m_da + m_ss*X_graf[0]
    Der = Y_w[-1]*m_da + m_ss*X_graf[-1]

    f1 = (0.003/1.003)*1750 + 82.2*(0.6) - (1-0.003)*1750*Y_w[-1] - (0.6*82.2)*X_graf[-1] # [kg/h]

    f_transf1 = 82.2*(0.6) - (0.6*82.2)*X_graf[-1]
    f_transf2 = (0.003/1.003)*1750 - (1-0.003)*1750*Y_w[-1]

    f2 = (0.003/1.003)*1710 + 88.2*(0.5) - (1-0.003)*1710*Y_w[-1] - (0.5*88.2)*X_graf[-1] # [kg/h]
