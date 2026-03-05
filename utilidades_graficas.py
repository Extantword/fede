# utilidades_graficas.py
# Funciones helper para la creación de gráficas con matplotlib.
# Elimina la repetición del código de configuración de gráficas.

import matplotlib.pyplot as plt
import numpy as np
import math as mt


def configurar_estilo():
    """Configura el estilo global de las gráficas."""
    plt.rcParams["font.family"] = "serif"


def crear_figura(figsize=(10, 5)):
    """
    Crea una figura con el estilo estándar: grid punteado y spines delgados.

    Retorna (fig, ax).
    """
    fig, ax = plt.subplots(1, figsize=figsize)

    ax.grid(True)
    ax.grid(color='#CCCCCC', linestyle='dotted', linewidth=0.5)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(0.5)

    return fig, ax


def guardar_figura(fig, nombre, dpi=600):
    """Guarda la figura como PDF."""
    fig.savefig(nombre, dpi=dpi)
    plt.close(fig)


def grafica_vs_distancia(x, y, xlabel, ylabel, nombre_pdf,
                         color='r', linewidth=0.5, figsize=(10, 5),
                         x_exp=None, y_exp=None):
    """
    Crea una gráfica de una variable vs distancia axial,
    opcionalmente con datos experimentales.
    """
    fig, ax = crear_figura(figsize)
    ax.plot(x, y, linewidth=linewidth, color=color)

    if x_exp is not None and y_exp is not None:
        ax.scatter(x_exp, y_exp, color='k', s=5)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    guardar_figura(fig, nombre_pdf)


def grafica_vs_tiempo(x, y, xlabel, ylabel, nombre_pdf,
                      color='b', linewidth=0.5, figsize=(10, 5),
                      xtick_step=None, ytick_range=None,
                      hlines=None, vlines=None,
                      x_exp=None, y_exp=None,
                      x_exp2=None, y_exp2=None):
    """
    Crea una gráfica de una variable vs tiempo,
    con opciones para líneas de referencia y datos experimentales.
    """
    fig, ax = crear_figura(figsize)
    ax.plot(x, y, linewidth=linewidth, color=color)

    if x_exp is not None and y_exp is not None:
        ax.scatter(x_exp, y_exp, color='k', s=5)

    if x_exp2 is not None and y_exp2 is not None:
        ax.scatter(x_exp2, y_exp2, color='k', s=5)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim([0, mt.ceil(x[-1])])

    if xtick_step is not None:
        plt.xticks(np.arange(0, mt.ceil(x[-1]), xtick_step))

    if ytick_range is not None:
        plt.yticks(np.arange(ytick_range[0], ytick_range[1], ytick_range[2]))

    if hlines is not None:
        for hl in hlines:
            plt.axhline(hl, linewidth=0.5, color='black', ls='--')

    if vlines is not None:
        for vl in vlines:
            plt.axvline(x=vl, linewidth=0.5, color='red', linestyle='-')

    guardar_figura(fig, nombre_pdf)
