from CoolProp.CoolProp import PropsSI
import numpy as np
import pandas as pd


def Trabajo_isotermo(T, P2, P1):  # Temperatura en ºC y presiones en MPa
    R = 0.287  # kJ/kg·K
    w = R*(T+273.15)*np.log(P2/P1)
    return np.round(w, 1)


def Kapitza(T1, T3, P1, P2):  # Temperaturas en ºC y presiones en MPa

    h1 = PropsSI('H', 'T', T1+273.15, 'P', P1*1E6, 'Air') / 1000  # en kJ/kg
    s1 = PropsSI('S', 'T', T1 + 273.15, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    h2 = PropsSI('H', 'T', T1 + 273.15, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg
    s2 = PropsSI('S', 'T', T1 + 273.15, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg·K
    h3 = PropsSI('H', 'T', T3 + 273.15, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg
    s3 = PropsSI('S', 'T', T3 + 273.15, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg·K

    Estado1 = {'Temperatura': T1, 'Presion': P1, 'Entalpia': np.round(h1, 1), 'Entropia': np.round(s1, 4)}
    Estado2 = {'Temperatura': T1, 'Presion': P2, 'Entalpia': np.round(h2, 1), 'Entropia': np.round(s2, 4)}
    Estado3 = {'Temperatura': T1, 'Presion': P2, 'Entalpia': np.round(h3, 1), 'Entropia': np.round(s3, 4)}

    Estados = [Estado1, Estado2, Estado3]

    s6 = s3
    h6 = PropsSI('H', 'S', s6 * 1E3, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg
    T6 = PropsSI('T', 'S', s6* 1E3, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado6 = {'Temperatura': np.round(T6, 1) , 'Presion': P1, 'Entalpia': np.round(h6, 1), 'Entropia': np.round(s6, 4)}

    h7 = PropsSI('H', 'Q', 0, 'P', P1*1E6, 'Air') / 1000  # en kJ/kg
    s7 = PropsSI('S', 'Q', 0, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T7 = PropsSI('T', 'Q', 0, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado7 = {'Temperatura': np.round(T7, 1), 'Presion': P1, 'Entalpia': np.round(h7, 1), 'Entropia': np.round(s7, 4)}

    h8 = PropsSI('H', 'Q', 1, 'P', P1*1E6, 'Air') / 1000  # en kJ/kg
    s8 = PropsSI('S', 'Q', 1, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T8 = PropsSI('T', 'Q', 1, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado8 = {'Temperatura': np.round(T8, 1), 'Presion': P1, 'Entalpia': np.round(h8, 1), 'Entropia': np.round(s8, 4)}

    Q5 = 1 - 2*(h2 - h1)/(h7 - h1)
    s5 = PropsSI('S', 'Q', Q5, 'P', P1 * 1E6, 'Air')/1000  # en kJ/kg·K
    T5 = PropsSI('T', 'Q', Q5, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    h5 = PropsSI('H', 'Q', Q5, 'P', P1 * 1E6, 'Air')/1000  # en kJ/kg
    Estado5 = {'Temperatura': np.round(T5, 1), 'Presion': P1, 'Entalpia': np.round(h5, 1), 'Entropia': np.round(s5, 4),
               'Título': np.round(Q5, 3)}

    h4 = h5
    s4 = PropsSI('S', 'H', h4*1E3, 'P', P2 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T4 = PropsSI('T', 'H', h4*1E3, 'P', P2 * 1E6, 'Air')-273.15  # en ºC
    Estado4 = {'Temperatura':  np.round(T4, 1), 'Presion': P2, 'Entalpia': np.round(h4, 1), 'Entropia': np.round(s4, 4)}

    m1 = 1  # kg/s
    m2 = 0.3 * m1  # kg/s
    m3 = 0.7 * m1  # kg/s
    m5 = m2 * Q5
    m4 = m2 * (1 - Q5)
    m6 = m5 + m3
    m_Alimento = m4  # kg/s

    Caudales = [m1, np.round(m2,3), np.round(m3, 3), np.round(m4, 3), np.round(m5, 3), np.round(m6, 3),
                np.round(m_Alimento, 3)]

    # Balance de energía en el mezclador
    h9 = (m3 * h6 + m5 * h8)/m6
    s9 = PropsSI('S', 'H', h9*1E3, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T9 = PropsSI('T', 'H', h9*1E3, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado9 = {'Temperatura':  np.round(T9, 1), 'Presion': P2, 'Entalpia': np.round(h9, 1), 'Entropia': np.round(s9, 4)}

    # Balance de energía en el segundo cambiador
    h10 = (m2 * (h3 - h4))/m6 + h9
    s10 = PropsSI('S', 'H', h10*1E3, 'P', P1 * 1E6, 'Air') / 1000  # en kJ/kg·K
    T10 = PropsSI('T', 'H', h10*1E3, 'P', P1 * 1E6, 'Air')-273.15  # en ºC
    Estado10 = {'Temperatura':  np.round(T10, 1), 'Presion': P2, 'Entalpia': np.round(h10, 1), 'Entropia': np.round(s10, 4)}

    Estados.extend([Estado4, Estado5, Estado6, Estado7, Estado8, Estado9, Estado10])

    W_Compresor = -Trabajo_isotermo(T1, P2, P1)  # kW/kg aire
    W_Turbina = np.round(-(h6 - h3) * m2, 1) # kW/kg aire

    return Estados, W_Compresor, W_Turbina, Caudales

T1 = 25  # ºC
P1 = 0.1013  # MPa
P2 = P1 * 7  # MPa
T3 = 0  # ºC

Estados, W_Compresor, W_Turbina, Caudales = Kapitza(T1, T3, P1, P2)

Estado_1 = []; Temperaturas = []; Presiones = []; Entalpias = []; Entropias = []
for i in range(10):
    Estado_1.append(str(i+1))
    Temperaturas.append(Estados[i]['Temperatura'])
    Presiones.append(Estados[i]['Presion'])
    Entalpias.append(Estados[i]['Entalpia'])
    Entropias.append(Estados[i]['Entropia'])

Titulo = ['-', '-', '-', '-', Estados[4]['Título'], '-',  '-', '-', '-', '-']

Resultados_NombreColumnas = ['Estados', 'Temperatura (ºC)', 'Presión (MPa)', 'Entalpía (kJ/kg)',
                             'Entropía (kJ/kg·K)', 'Título']
DatosResultados = [Estado_1, Temperaturas, Presiones, Entalpias, Entropias, Titulo]

datos_resultados = dict(zip(Resultados_NombreColumnas, DatosResultados))
Tabla_Resultados = pd.DataFrame(datos_resultados)
Tabla_Resultados = Tabla_Resultados.set_index('Estados')

ResultadosCaudales_NombreColumnas = ['m1 (kg/h)', 'm2 (kg/h)', 'm3 (kg/h)', 'm4 (kg/h)', 'm5 (kg/h)', 'm6 (kg/h)', 'm Alimento fresco (kg/h)',
                                     'W compresor (kW/kg en 1)', 'W turbina (kW/kg en 1)']

DatosResultadosCaudales = [Caudales[0], Caudales[1], Caudales[2], Caudales[3], Caudales[4], Caudales[5], Caudales[6],
                           W_Compresor, W_Turbina]

datos_resultadosCaudales = dict(zip(ResultadosCaudales_NombreColumnas, DatosResultadosCaudales))
Tabla_ResultadosCaudales = pd.DataFrame(datos_resultadosCaudales, index=['Caudales'],
                                           columns=ResultadosCaudales_NombreColumnas)

writer = pd.ExcelWriter('Problema 5 Resultados.xlsx')
Tabla_Resultados.to_excel(writer, sheet_name='Prop  Termodinamicas')
Tabla_ResultadosCaudales.to_excel(writer, sheet_name='Caudales')
writer.save()
writer.close()