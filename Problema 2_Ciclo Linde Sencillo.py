from CoolProp.CoolProp import PropsSI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLine

def Proceso_isobaro(T_inferior, Tmax, P, n_puntos):

    # Valores de temperatura en ºC y de presión en MPa
    Presion = P*1E6
    ValoresT = np.linspace(Tmax+273.15, T_inferior+273.15, n_puntos, endpoint=True)
    valoresS = np.ones(n_puntos)

    for i in range(n_puntos):
        valoresS[i] = PropsSI('S', 'P', Presion, 'T', ValoresT[i], 'Air')/1000  # en kJ/kg·K

    return ValoresT, valoresS

def Proceso_isobaroVapor(T_inferior, Tmax, P, n_puntos):

    # Valores de temperatura en ºC y de presión en MPa
    Presion = P*1E6
    ValoresT = np.linspace(T_inferior+273.15, Tmax+273.15, n_puntos, endpoint=True)
    valoresS = np.ones(n_puntos)
    valoresS[0] = PropsSI('S', 'P', Presion, 'Q', 1, 'Air')/1000  # en kJ/kg·K

    for i in range(1, n_puntos):
        valoresS[i] = PropsSI('S', 'P', Presion, 'T', ValoresT[i], 'Air')/1000  # en kJ/kg·K

    return ValoresT, valoresS

def Proceso_isentalpico(P_inferior, Pmax, h, n_puntos):

    # Valores de temperatura en ºC y de presión en MPa
    entalpia = h*1E3
    ValoresP = np.linspace(P_inferior*1E6, Pmax*1E6, n_puntos, endpoint=True)
    valoresS = np.ones(n_puntos)
    valoresT = np.ones(n_puntos)

    for i in range(n_puntos):
        valoresS[i] = PropsSI('S', 'H', entalpia, 'P', ValoresP[i], 'Air')/1000  # en kJ/kg·K
        valoresT[i] = PropsSI('T', 'H', entalpia, 'P', ValoresP[i], 'Air') # en K

    return valoresT, valoresS

def Trabajo_isotermo(T, P2, P1):  # Temperatura en ºC y presiones en MPa
    R = 0.287  # kJ/kg·K
    w = R*(T+273.15)*np.log(P2/P1)
    return w


def Linde(T1, P1, P2):  # Temperaturas en ºC y presiones en MPa

    h1 = PropsSI('H', 'T', T1+273.15, 'P', P1*1E6, 'Air') / 1000  # en kJ/kg
    s1 = PropsSI('S', 'T', T1 + 273.15, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    h2 = PropsSI('H', 'T', T1 + 273.15, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg
    s2 = PropsSI('S', 'T', T1 + 273.15, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg·K

    Estado1 = {'Temperatura': T1, 'Presion': P1, 'Entalpia': np.round(h1, 1), 'Entropia': np.round(s1, 4)}
    Estado2 = {'Temperatura': T1, 'Presion': P2, 'Entalpia': np.round(h2, 1), 'Entropia': np.round(s2, 4)}

    Estados = [Estado1, Estado2]

    h5 = PropsSI('H', 'Q', 0, 'P', P1*1E6, 'Air') / 1000  # en kJ/kg
    s5 = PropsSI('S', 'Q', 0, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T5 = PropsSI('T', 'Q', 0, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado5 = {'Temperatura': np.round(T5, 1), 'Presion': P1, 'Entalpia': np.round(h5, 1), 'Entropia': np.round(s5, 4)}

    h6 = PropsSI('H', 'Q', 1, 'P', P1*1E6, 'Air') / 1000  # en kJ/kg
    s6 = PropsSI('S', 'Q', 1, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T6 = PropsSI('T', 'Q', 1, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado6 = {'Temperatura': np.round(T6, 1), 'Presion': P1, 'Entalpia': np.round(h6, 1), 'Entropia': np.round(s6, 4)}


    Q4 = (h2-h5)/(h1-h5)
    s4 = PropsSI('S', 'Q', Q4, 'P', P1 * 1E6, 'Air')/1000  # en kJ/kg·K
    T4 = PropsSI('T', 'Q', Q4 , 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    h4 = PropsSI('H', 'Q', Q4, 'P', P1 * 1E6, 'Air')/1000  # en kJ/kg

    Estado4 = {'Temperatura': np.round(T4, 1), 'Presion': P1, 'Entalpia': np.round(h4, 1), 'Entropia': np.round(s4, 4),
               'Título': np.round(Q4, 3)}

    h3 = h4
    s3 = PropsSI('S', 'H', h3*1E3, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T3 = PropsSI('T', 'H', h3*1E3, 'P', P2 * 1E6, 'Air')-273.15  # en ºC
    Estado3 = {'Temperatura':  np.round(T3, 1), 'Presion': P2, 'Entalpia': np.round(h3, 1), 'Entropia': np.round(s3, 4)}

    Estados.extend([Estado3, Estado4, Estado5, Estado6])
    w = Trabajo_isotermo(T1, P2, P1)

    Q_cambiador = h2 - h3  # kW/kg aire

    return Estados, w, Q_cambiador





# for T1 in []:
T1 = 35  # ºC
P1 = 0.1  # MPa
P2 = 35  # MPa
n_puntos = 50

Estados, w, Q_cambiador = Linde(T1, P1, P2)

T3 = Estados[2]['Temperatura']

from Constructor_Diagrama_Hausen_Aire import figura, ax


Proceso1_T = [T1+273.15, T1+273.15]
Proceso1_S = [Estados[0]['Entropia'], Estados[1]['Entropia']]
lineaProceso1, = ax.plot(Proceso1_S, Proceso1_T, 'r'); lineaProceso1.set_lw(1.5)

Proceso2_T, Proceso2_S = Proceso_isobaro(T1, T3, P2, n_puntos)
lineaProceso2, = ax.plot(Proceso2_S, Proceso2_T, 'r'); lineaProceso2.set_lw(1.5)

Proceso3_T, Proceso3_S = Proceso_isentalpico(P1, P2, Estados[2]['Entalpia'], n_puntos)
lineaProceso3, = ax.plot(Proceso3_S, Proceso3_T, 'r'); lineaProceso3.set_lw(1.5)

Proceso4_T, Proceso4_S = Proceso_isobaroVapor(Estados[3]['Temperatura'], T1, P1, n_puntos)
lineaProceso4, = ax.plot(Proceso4_S, Proceso4_T, 'r'); lineaProceso4.set_lw(1.5)

IsotermaEquilibrio, = ax.plot([Estados[4]['Entropia'], Estados[5]['Entropia']],
                              [Estados[4]['Temperatura']+273.15, Estados[5]['Temperatura']+273.15], 'r')
IsotermaEquilibrio.set_lw(1.5)

figura.show()

metadatos={ 'Author': 'Juan Carlos Domínguez', 'Title': 'Problema 2',
    'Creator': 'Creado con CoolProp en Python'}
figura.savefig('Problema2.pdf', format='pdf', quality=95, metadata=metadatos,
               dpi=300)
# metadatos_png={ 'Author': 'Juan Carlos Domínguez', 'Title': 'Diagrama T-s (Hausen) para el aire',
#     'Creation Time':'01-Dic-2019','Software': 'Creado con CoolProp en Python'}
#
# figura.savefig('Problema2.png',format='png',quality=95,metadata=metadatos_png,
#                dpi=300)

Estado_1 = []; Temperaturas = []; Presiones = []; Entalpias = []; Entropias = []
for i in range(6):
    Estado_1.append(str(i+1))
    Temperaturas.append(Estados[i]['Temperatura'])
    Presiones.append(Estados[i]['Presion'])
    Entalpias.append(Estados[i]['Entalpia'])
    Entropias.append(Estados[i]['Entropia'])

Titulo = ['-', '-', '-', Estados[3]['Título'], '-', '-']

Resultados_NombreColumnas = ['Estados', 'Temperatura (ºC)', 'Presión (MPa)', 'Entalpía (kJ/kg)',
                             'Entropía (kJ/kg·K)', 'Título']
DatosResultados = [Estado_1, Temperaturas, Presiones, Entalpias, Entropias, Titulo]

datos_resultados = dict(zip(Resultados_NombreColumnas, DatosResultados))
Tabla_Resultados = pd.DataFrame(datos_resultados)
Tabla_Resultados = Tabla_Resultados.set_index('Estados')

writer = pd.ExcelWriter('Problema 2 Resultados.xlsx')
Tabla_Resultados.to_excel(writer, sheet_name='Prop  Termodinamicas')
writer.save()
writer.close()






