from CoolProp.CoolProp import PropsSI
import numpy as np
import pandas as pd


def Trabajo_isotermo(T, P2, P1):  # Temperatura en ºC y presiones en MPa
    R = 0.287  # kJ/kg·K
    w = R*(T+273.15)*np.log(P2/P1)
    return np.round(w, 1)


def dual_pressure_Claude(T1, P1, P2, P3, T11, mE):  # Temperaturas en ºC y presiones en MPa

    h1 = PropsSI('H', 'T', T1+273.15, 'P', P1*1E6, 'Nitrogen') / 1000  # en kJ/kg
    s1 = PropsSI('S', 'T', T1 + 273.15, 'P', P1 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    h2 = PropsSI('H', 'T', T1 + 273.15, 'P', P2 * 1E6, 'Nitrogen') / 1000  # en kJ/kg
    s2 = PropsSI('S', 'T', T1 + 273.15, 'P', P2 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    h3 = PropsSI('H', 'T', T3 + 273.15, 'P', P3 * 1E6, 'Nitrogen') / 1000  # en kJ/kg
    s3 = PropsSI('S', 'T', T3 + 273.15, 'P', P3 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    h11 = PropsSI('H', 'T', T11 + 273.15, 'P', P2 * 1E6, 'Nitrogen') / 1000  # en kJ/kg
    s11 = PropsSI('S', 'T', T11 + 273.15, 'P', P2 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    

    Estado1 = {'Temperatura': T1, 'Presion': P1, 'Entalpia': np.round(h1, 1), 'Entropia': np.round(s1, 4)}
    Estado2 = {'Temperatura': T1, 'Presion': P2, 'Entalpia': np.round(h2, 1), 'Entropia': np.round(s2, 4)}
    Estado3 = {'Temperatura': T1, 'Presion': P3, 'Entalpia': np.round(h3, 1), 'Entropia': np.round(s3, 4)}
    Estado11 = {'Temperatura': T11, 'Presion': P2, 'Entalpia': np.round(h11, 1), 'Entropia': np.round(s11, 4)}

    Estados = [Estado1, Estado2, Estado3, Estado11]

    se = s11
    he = PropsSI('H', 'S', se * 1E3, 'P', P1 * 1E6, 'Nitrogen') / 1000  # en kJ/kg
    Te = PropsSI('T', 'S', se* 1E3, 'P', P1 * 1E6, 'Nitrogen')-273.15  # en ºC
    Estadoe = {'Temperatura': np.round(Te, 1) , 'Presion': P1, 'Entalpia': np.round(he, 1), 'Entropia': np.round(s11, 4)}

    hf = PropsSI('H', 'Q', 0, 'P', P1*1E6, 'Nitrogen') / 1000  # en kJ/kg
    sf = PropsSI('S', 'Q', 0, 'P', P1 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    Tf = PropsSI('T', 'Q', 0, 'P', P1 * 1E6, 'Nitrogen')-273.15  # en ºC
    Estadof = {'Temperatura': np.round(Tf, 1), 'Presion': P1, 'Entalpia': np.round(hf, 1), 'Entropia': np.round(sf, 4)}

    hg = PropsSI('H', 'Q', 1, 'P', P1*1E6, 'Nitrogen') / 1000  # en kJ/kg
    sg = PropsSI('S', 'Q', 1, 'P', P1 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    Tg = PropsSI('T', 'Q', 1, 'P', P1 * 1E6, 'Nitrogen')-273.15  # en ºC
    Estadog = {'Temperatura': np.round(Tg, 1), 'Presion': P1, 'Entalpia': np.round(hg, 1), 'Entropia': np.round(sg, 4)}

    # Balance de energía en cambiadores+válvula J-T+ depósito
    Q7 = (h3-h1+(mE*(h2-h3-he-h11)))/(hf-h1)
    s7 = PropsSI('S', 'Q', Q7, 'P', P1 * 1E6, 'Nitrogen')/1000  # en kJ/kg·K
    T7 = PropsSI('T', 'Q', Q7, 'P', P1 * 1E6, 'Nitrogen')-273.15  # en ºC
    h7 = PropsSI('H', 'Q', Q7, 'P', P1 * 1E6, 'Nitrogen')/1000  # en kJ/kg
    Estado7 = {'Temperatura': np.round(T7, 1), 'Presion': P1, 'Entalpia': np.round(h7, 1), 'Entropia': np.round(s7, 4),
               'Título': np.round(Q7, 3)}

    h6 = h7
    s6 = PropsSI('S', 'H', h6*1E3, 'P', P3 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
    T6 = PropsSI('T', 'H', h6*1E3, 'P', P3 * 1E6, 'Nitrogen')-273.15  # en ºC
    Estado6 = {'Temperatura':  np.round(T6, 1), 'Presion': P3, 'Entalpia': np.round(h7, 1), 'Entropia': np.round(s6, 4)}

    m1 = 1  # kg/s
    m2= m1 # kg/s
    m2_1 = mE * m1  # kg/s
    m2_2 = (1-mE) *m1 # kg/s
    m3 = (1-mE) * m1  # kg/s
    m4 = (1-mE) * m1 # kg/s
    m5 = (1-mE) * m1 # kg/s
    m6 = (1-mE) * m1 # kg/s
    m7 = (1-mE) * m1# kg/s
    m8 =(1-mE-Q7) * m1 # kg/s
    m9 = (1-Q7) * m1  # kg/s
    m10 = (1-Q7) * m1  # kg/s
    m11 = mE * m1# kg/s
    mf = (1-mE)*Q7 # kg/s
    mg = (1-mE-mf) * m1  # kg/s
    m_Alimento = m1  # kg/s

    Caudales = [m1, np.round(m2_1,3), np.round(m2_2,3), np.round(m3, 3), np.round(m4, 3), np.round(m5, 3), np.round(m6, 3),
                np.round(m_Alimento, 3)]

#     # Balance de energía en el mezclador
#     h9 = (m3 * h6 + m5 * h8)/m6
#     s9 = PropsSI('S', 'H', h9*1E3, 'P', P1 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
#     T9 = PropsSI('T', 'H', h9*1E3, 'P', P1 * 1E6, 'Nitrogen')-273.15  # en ºC
#     Estado9 = {'Temperatura':  np.round(T9, 1), 'Presion': P2, 'Entalpia': np.round(h9, 1), 'Entropia': np.round(s9, 4)}

#     # Balance de energía en el segundo cambiador
#     h10 = (m2 * (h3 - h4))/m6 + h9
#     s10 = PropsSI('S', 'H', h10*1E3, 'P', P1 * 1E6, 'Nitrogen') / 1000  # en kJ/kg·K
#     T10 = PropsSI('T', 'H', h10*1E3, 'P', P1 * 1E6, 'Nitrogen')-273.15  # en ºC
#     Estado10 = {'Temperatura':  np.round(T10, 1), 'Presion': P2, 'Entalpia': np.round(h10, 1), 'Entropia': np.round(s10, 4)}

#     Estados.extend([Estado4, Estado5, Estado6, Estado7, Estado8, Estado9, Estado10])

#     W_Compresor = -Trabajo_isotermo(T1, P2, P1)  # kW/kg aire
#     W_Turbina = np.round(-(h6 - h3) * m2, 1) # kW/kg aire

#     return Estados, W_Compresor, W_Turbina, Caudales

# T1 = 25  # ºC
# P1 = 0.1013  # MPa
# P2 = P1 * 7  # MPa
# T3 = 0  # ºC

# Estados, W_Compresor, W_Turbina, Caudales = Kapitza(T1, T3, P1, P2)

# figura=plt.figure()
# ax=figura.add_axes([0.12,0.1,0.8,0.8])
# Resultado,=ax.plot([20,25,30, 35,40],Rendimientos,'r');Resultado.set_lw(1.5)
# ax.set_xlabel('Temperatura inicial (ºC)')
# ax.set_ylabel('Título del vapor ()')
# figura.show()


# T3 = Estados[2]['Temperatura']

# from Constructor_Diagrama_Hausen_Nitrógeno import figura, ax


# Proceso1_T = [T1+273.15, T1+273.15]
# Proceso1_S = [Estados[0]['Entropia'], Estados[1]['Entropia']]
# lineaProceso1, = ax.plot(Proceso1_S, Proceso1_T, 'r'); lineaProceso1.set_lw(1.5)

# Proceso2_T, Proceso2_S = Proceso_isobaro(T1, T3, P2, n_puntos)
# lineaProceso2, = ax.plot(Proceso2_S, Proceso2_T, 'r'); lineaProceso2.set_lw(1.5)

# Proceso3_T, Proceso3_S = Proceso_isentalpico(P1, P2, Estados[2]['Entalpia'], n_puntos)
# lineaProceso3, = ax.plot(Proceso3_S, Proceso3_T, 'r'); lineaProceso3.set_lw(1.5)

# Proceso4_T, Proceso4_S = Proceso_isobaroVapor(Estados[3]['Temperatura'], T1, P1, n_puntos)
# lineaProceso4, = ax.plot(Proceso4_S, Proceso4_T, 'r'); lineaProceso4.set_lw(1.5)

# IsotermaEquilibrio, = ax.plot([Estados[4]['Entropia'], Estados[5]['Entropia']],
#                               [Estados[4]['Temperatura']+273.15, Estados[5]['Temperatura']+273.15], 'r')
# IsotermaEquilibrio.set_lw(1.5)

# figura.show()

# metadatos={ 'Author': 'Vicky', 'Title': 'Problema 5',
#     'Creator': 'Creado con CoolProp en Python'}
# figura.savefig('Problema5.pdf', format='pdf', quality=95, metadata=metadatos,
#                 dpi=300)

# Estado_1 = []; Temperaturas = []; Presiones = []; Entalpias = []; Entropias = []
# for i in range(10):
#     Estado_1.append(str(i+1))
#     Temperaturas.append(Estados[i]['Temperatura'])
#     Presiones.append(Estados[i]['Presion'])
#     Entalpias.append(Estados[i]['Entalpia'])
#     Entropias.append(Estados[i]['Entropia'])

# Titulo = ['-', '-', '-', '-', Estados[4]['Título'], '-',  '-', '-', '-', '-']

# Resultados_NombreColumnas = ['Estados', 'Temperatura (ºC)', 'Presión (MPa)', 'Entalpía (kJ/kg)',
#                              'Entropía (kJ/kg·K)', 'Título']
# DatosResultados = [Estado_1, Temperaturas, Presiones, Entalpias, Entropias, Titulo]

# datos_resultados = dict(zip(Resultados_NombreColumnas, DatosResultados))
# Tabla_Resultados = pd.DataFrame(datos_resultados)
# Tabla_Resultados = Tabla_Resultados.set_index('Estados')

# ResultadosCaudales_NombreColumnas = ['m1 (kg/h)', 'm2 (kg/h)', 'm3 (kg/h)', 'm4 (kg/h)', 'm5 (kg/h)', 'm6 (kg/h)', 'm Alimento fresco (kg/h)',
#                                      'W compresor (kW/kg en 1)', 'W turbina (kW/kg en 1)']

# DatosResultadosCaudales = [Caudales[0], Caudales[1], Caudales[2], Caudales[3], Caudales[4], Caudales[5], Caudales[6],
#                            W_Compresor, W_Turbina]

# datos_resultadosCaudales = dict(zip(ResultadosCaudales_NombreColumnas, DatosResultadosCaudales))
# Tabla_ResultadosCaudales = pd.DataFrame(datos_resultadosCaudales, index=['Caudales'],
#                                            columns=ResultadosCaudales_NombreColumnas)

# writer = pd.ExcelWriter('Problema Claude dual Resultados.xlsx')
# Tabla_Resultados.to_excel(writer, sheet_name='Prop  Termodinamicas')
# Tabla_ResultadosCaudales.to_excel(writer, sheet_name='Caudales')
# writer.save()
# writer.close()