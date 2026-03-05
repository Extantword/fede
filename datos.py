# datos.py
# Script de ejecución y análisis de resultados de la simulación.
# Reemplazo limpio de "Código datos.py".

import numpy as np
import time

from simulacion import Spray
from constantes import (
    IDX_FINAL_I, IDX_T_EOUT, IDX_F_BALANC, IDX_R_TIME,
)


def formatear_tiempo(segundos):
    """Formatea un tiempo en segundos a una cadena legible."""
    if segundos < 60:
        return f"{segundos:.2f} segundos"
    elif segundos < 3600:
        minutos = int(segundos // 60)
        segs = int(segundos % 60)
        return f"{minutos} minutos y {segs} segundos"
    else:
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        segs = int(segundos % 60)
        return f"{horas} horas, {minutos} minutos y {segs} segundos"


if __name__ == "__main__":
    start = time.process_time()

    # Parámetros iniciales
    initial_size = 50000
    h_eu = 0.0005

    # Vectores iniciales
    Mov_VarTemp_1 = np.zeros(initial_size)
    T_EpTemp_1    = np.zeros(initial_size)
    Y_p_1         = np.zeros(initial_size)
    M_wDL_1       = np.zeros(initial_size)

    print("Comenzando la simulación...")

    Resultados = Spray(initial_size, h_eu, 80000, Mov_VarTemp_1, T_EpTemp_1, Y_p_1, M_wDL_1)

    print("Simulación completada.")

    elapsed = time.process_time() - start
    print()
    print(f"La simulación tardó {formatear_tiempo(elapsed)}")

    # Extracción de resultados
    final_i = Resultados[IDX_FINAL_I]
    tiempos_residencia = Resultados[IDX_R_TIME]
    temp_salida = Resultados[IDX_T_EOUT]
    f_balanc = Resultados[IDX_F_BALANC]

    print(f"El balance de masa (f_balanc) es: {f_balanc} kg/h")
    print(f"La última iteración de la gota es: {final_i}")

    if len(tiempos_residencia) > 0:
        promedio = np.mean(np.array(tiempos_residencia))
        print(f"Tiempo de residencia promedio: {formatear_tiempo(promedio)}")
    else:
        print("No se registraron tiempos de residencia suficientes para el cálculo promedio.")

    if len(temp_salida) > 0:
        print(f"Temperatura exterior promedio: {np.mean(np.array(temp_salida)):.2f} °C")
