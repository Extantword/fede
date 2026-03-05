# constantes.py
# Constantes físicas, parámetros de diseño y coeficientes para la simulación
# de la torre de secado por aspersión.

# ============================================================================
# Constantes físicas universales
# ============================================================================

MW = 18.0       # Peso molecular del agua [g/mol]
MA = 28.96      # Peso molecular del aire [g/mol]
R = 8314.0      # Constante universal de los gases [J/(kmol·K)]
R_MOL = 8.314   # Constante universal de los gases [J/(mol·K)]
P_ATM = 101325.0  # Presión atmosférica [Pa]
SIGMA = 5.67e-11  # Constante de Stefan-Boltzmann [kW/(m²·K⁴)]
G = 9.81        # Aceleración de la gravedad [m/s²]
K_OFFSET = 273.15  # Conversión Celsius a Kelvin

# ============================================================================
# Coeficientes polinomiales de propiedades termofísicas
# ============================================================================

# Cp del aire seco: Cp(T) = (a0 + a1*T_K + a2*T_K² + a3*T_K³) / 1000 [kJ/(kg·K)]
# donde T_K = T[°C] + 273.15
CP_AIR_A0 = 969.542
CP_AIR_A1 = 6.801e-2
CP_AIR_A2 = 16.569e-5
CP_AIR_A3 = -67.828e-9

# Cp del vapor de agua: Cp(T) = b0 + b1*T + b2*T² + b3*T³ [kJ/(kg·K)]
# donde T en °C
CP_VAP_B0 = 1.883
CP_VAP_B1 = -1.674e-4
CP_VAP_B2 = 8.4390e-7
CP_VAP_B3 = -2.6970e-10

# Densidad del agua: rho(T) = (c0 + c1*T + c2*T²) * 1000 [kg/m³]
RHO_W_C0 = 1.0020825
RHO_W_C1 = -1.14e-4
RHO_W_C2 = -3.325e-6

# Viscosidad del aire: mu(T) = d0 + d1*T [Pa·s]
MU_AIR_D0 = 1.72e-5
MU_AIR_D1 = 4.568e-8

# Conductividad del aire: k(T) = 1.731 * (e0 + e1*T) / 1000 [kW/(m·K)]
K_AIR_FACTOR = 1.731
K_AIR_E0 = 0.014
K_AIR_E1 = 4.296e-5

# Densidad del aire a STP: rho_0 * T_0 / T [kg/m³]
RHO_AIR_STP = 1.293

# Presión de saturación (Antoine): P_sat = (P_ATM/760) * 10^(A - B/(T+C))
ANTOINE_A = 7.95581
ANTOINE_B = 1668.210
ANTOINE_C = 228.0

# ============================================================================
# Propiedades del material (maltodextrina)
# ============================================================================

CP_MALTODEXTRINA = 1.5   # Capacidad calorífica [kJ/(kg·K)]
CP_AGUA_LIQ = 4.18       # Capacidad calorífica del agua líquida [kJ/(kg·K)]
EMISIVIDAD = 0.96         # Emisividad de la partícula
X_CRITICA = 0.54          # Humedad crítica en base seca (formación de costra)
DIFF_COSTRA = 2.5 * 5.9e-9  # Difusividad efectiva del vapor a través de la costra [m²/s]
U_H = 16.75 * 0.4 / 3600.0  # Coeficiente global de transferencia de calor [kW/(m²·K)]

# ============================================================================
# Parámetros de diseño de la torre
# ============================================================================

TORRE_DIAMETRO = 2.0       # Diámetro de la torre [m]
TORRE_ALTURA_CONO = 2.3    # Altura donde comienza la sección cónica [m]
TORRE_ALTURA_MAX = 3.5     # Altura máxima de la torre [m]

# ============================================================================
# Parámetros de diseño de la boquilla
# ============================================================================

D0_ORIFICIO = 0.711e-3   # Diámetro del orificio [m]
DS_FACTOR = 2.4           # Factor: Ds = DS_FACTOR * d0
AP_FACTOR = 0.50          # Factor: A_p = AP_FACTOR * d0 * Ds
L0_FACTOR = 0.1           # Factor: l0 = L0_FACTOR * d0
LS_FACTOR = 0.5           # Factor: Ls = LS_FACTOR * Ds

# ============================================================================
# Parámetros experimentales del spray
# ============================================================================

Q_N = 2.09       # Parámetro experimental
D_N = 70.5e-6    # Diámetro nominal [m]

# Propiedades del spray de referencia
D_0_REF = 154.39       # Diámetro de referencia [µm]
S_0_REF = 0.5          # Fracción de sólidos de referencia
F_P0_REF = 0.021921    # Flujo de alimentación de referencia [kg/s]
T_P0R_REF = 326.35     # Temperatura de referencia [K]

# Coeficientes de control del SMD
A_SMD = 400
B_SMD = -10.1
C_SMD = -600.0

# ============================================================================
# Parámetros del sistema de calentamiento
# ============================================================================

M_INH = 1710             # Flujo de aire de entrada [kg/h]
V_H = 4                  # Volumen del calentador [m³]
H_COMB = 48.5e3          # Calor de combustión [kJ/kg]
P_DUCT = 101325          # Presión en el ducto [Pa]
P_OPER_R = 1.5           # Relación de presión operativa

# Parámetros del ducto
DIST_DUCT = 9            # Longitud del ducto [m]
A_DUCT = 0.45 * 0.45     # Área de sección transversal del ducto [m²]

# ============================================================================
# Condiciones iniciales por defecto
# ============================================================================

T_AMB = 25       # Temperatura ambiente [°C]
T_REF = 0        # Temperatura de referencia [°C]
T_WALL_INIT = 200  # Temperatura inicial de la pared [°C]
P_INIT = 101325  # Presión inicial [Pa]

# ============================================================================
# Parámetros de la boquilla y líquido
# ============================================================================

X_R_DEFAULT = 0.46          # Fracción de soluto
T_R0_DEFAULT = 52.5         # Temperatura inicial del refinado [°C]
MU_L_DEFAULT = 1.8e-3       # Viscosidad de la leche [Pa·s]
P_IN_DEFAULT = 9e5          # Presión de entrada del líquido [Pa]
PARAM_M_L = 0.802           # Parámetro de ajuste del flujo

# Condiciones de operación del spray
F_PIN_DEFAULT = 89.2 / 3600.0  # Flujo de alimentación [kg/s]
U_R_DEFAULT = 0.5               # Velocidad radial [m/s]
U_TAN_DEFAULT = 1.0             # Velocidad tangencial [m/s]
U_X_DEFAULT = 7.0               # Velocidad axial [m/s]
MMD_SMD_APPROX = 1.1            # Valor aproximado de MMD/SMD
T_EC_DEFAULT = 170.3            # Temperatura de entrada del aire de secado [°C]
Y0_DEFAULT = 0.0021             # Humedad inicial del aire (base seca)
M_AIR_DEFAULT = 1710.0 / 3600.0  # Flujo másico del aire de secado [kg/s]

# ============================================================================
# Coeficientes de velocidad del gas en la torre
# ============================================================================

COEF_VELZ = 7.376 * 0.45   # Coeficiente de velocidad axial del gas
COEF_VELT = 0.7             # Coeficiente de velocidad tangencial del gas
V_GR_NEAR_NOZZLE = -5.19    # Velocidad radial cerca de la boquilla [m/s]

# ============================================================================
# Iteraciones para snapshots (Bef0-Bef6)
# ============================================================================

SNAPSHOT_ITERATIONS = [1000, 2000, 3000, 4000, 15000, 20000, 30000]

# ============================================================================
# Índices del resultado de Spray()
# ============================================================================

IDX_PARTE3 = 0
IDX_D_DROP = 1
IDX_FINAL_I = 2
IDX_MOV_Z = 3
IDX_T_EG = 4
IDX_Y_GRAF = 5
IDX_T_R = 6
IDX_V_GZ = 7
IDX_MOV_R = 8
IDX_M_EW = 9
IDX_VEL_GRAF = 10
IDX_X_W = 11
IDX_Y_P = 12
IDX_MV_FLUX = 13
IDX_VEL_G = 14
IDX_VEL_GF = 15
IDX_BEF0 = 16
IDX_BEF1 = 17
IDX_BEF2 = 18
IDX_BEF3 = 19
IDX_BEF4 = 20
IDX_BEF5 = 21
IDX_BEF6 = 22
IDX_T_EOUT = 23
IDX_Y_EOUT = 24
IDX_T_ECT = 25
IDX_Z_MOUT = 26
IDX_P_INMV = 27
IDX_M_LMV = 28
IDX_REMAINW = 29
IDX_F_PARAM = 30
IDX_F_BALANC = 31
IDX_R_TIME = 32
