from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLine

def LineaLS(Tmin):
    Tc=PropsSI('Tcrit','Air') # En K
    Temperaturas=np.linspace(Tmin, Tc, 200, endpoint=True)
    Entropias=np.ones(200)
    for i in range(200):
        Entropias[i]=PropsSI('S', 'T', Temperaturas[i], 'Q', 0, 'Air')/1000 # en kJ/kg·K
    return Temperaturas,Entropias

def LineaVS(Tmin):
    Tc=PropsSI('Tcrit','Air') # En K
    Temperaturas=np.linspace(Tmin, Tc, 200, endpoint=True)
    Entropias=np.ones(200)
    for i in range(200):
        Entropias[i]=PropsSI('S', 'T', Temperaturas[i], 'Q', 1, 'Air')/1000 # en kJ/kg·K
    Temperaturas=np.append(Temperaturas,Tc)
    Entropias=np.append(Entropias,PropsSI('S', 'T', Tc, 'Q', 0, 'Air')/1000) # en kJ/kg·K
    return Temperaturas,Entropias

def isentropica_Sobrecalentado(ValorH,Pmin, Pmax,n_puntos):
    assert type(Pmin) == int or type(Pmin) == float
    assert type(Pmax) == int or type(Pmax) == float
    assert type(ValorH) == int or type(ValorH) == float
    assert type(n_puntos) == int

    H=1E3*ValorH # Conversión de la entalpía de kJ/kg  J/kg para usar CoolProp
    ValoresP = np.linspace(Pmax,Pmin,  n_puntos, endpoint=True)*1E5 # Cambio de bar a Pa
    valoresS=np.ones(n_puntos)
    valoresT = np.ones(n_puntos)

    for i in range(n_puntos):
        valoresS[i]= PropsSI('S', 'P', ValoresP[i], 'H', H, 'Air')/1000 # en kJ/kg·K
        valoresT[i] = PropsSI('T', 'P',  ValoresP[i], 'H', H, 'Air')   # en kJ/kg·K
    return valoresS,valoresT

def isobara_Sobrecalentado( Valor,Tmin, Tmax,n_puntos):

    assert type(Tmin) == int or type(Tmin) == float
    assert type(Tmax) == int or type(Tmax) == float
    assert type(Valor) == int or type(Valor) == float
    assert type(n_puntos) == int

    P=1E5*Valor # Conversión de la presión en bar a Pa
    try:
        T_inferior = PropsSI('T', 'P', P, 'Q', 1, 'Air')+5 # en K
        Cierre = True
    except:
        T_inferior=Tmin
        Cierre=False

    ValoresT = np.linspace(Tmax,T_inferior,  n_puntos, endpoint=True)
    valoresS=np.ones(n_puntos)
    for i in range(n_puntos):
        valoresS[i]= PropsSI('S', 'P', P, 'T', ValoresT[i], 'Air')/1000 # en kJ/kg·K
    if Cierre:
        ValoresT = np.append(ValoresT, PropsSI('T', 'P', P, 'Q', 1, 'Air'))
        valoresS = np.append(valoresS, PropsSI('S', 'P', P, 'Q', 1, 'Air') / 1000)  # en kJ/kg·K
    return valoresS,ValoresT


def isobara_Equilibrio( Valor,n_puntos):

    assert type(Valor) == int or type(Valor) == float
    assert type(n_puntos) == int

    P=1E5*Valor # Conversión de la presión en bar a Pa

    try:
        T= PropsSI('T', 'P', P, 'Q', 1, 'Air') # en K
    except:
        return np.zeros([n_puntos]),np.zeros([n_puntos])
    valoresS = np.ones(n_puntos)
    valoresT = np.ones(n_puntos)
    Titulos=np.linspace(0,1,  n_puntos, endpoint=True)
    for i in range(n_puntos):
        valoresS[i] = PropsSI('S', 'P', P, 'Q', Titulos[i], 'Air')/1000 # en kJ/kg·K
        valoresT[i] = PropsSI('T', 'P', P, 'Q', Titulos[i], 'Air')  # en K
    return valoresS, valoresT


def Isentalpicas_Sobrecalentado(Isentalpicas_Valores ,Pmin, Pmax,n_puntos):
    IsentalpicasT={}
    IsentalpicasS={}
    for i in Isentalpicas_Valores:
        [valoresS, ValoresT] = isentropica_Sobrecalentado(float(i), Pmin, Pmax,n_puntos)
        IsentalpicasT[str(i)]=ValoresT
        IsentalpicasS[str(i)]=valoresS
    return IsentalpicasT,IsentalpicasS

def Isobaras_Sobrecalentado(Isobaras_Valores,Tmin, Tmax,n_puntos):
    IsobarasT={}
    IsobarasS={}
    for i in Isobaras_Valores:
        [valoresS, ValoresT] = isobara_Sobrecalentado(float(i), Tmin, Tmax,n_puntos)
        IsobarasT[str(i)]=ValoresT
        IsobarasS[str(i)]=valoresS
    return IsobarasT,IsobarasS

def Isobaras_Equilibrio(Isobaras_Valores,n_puntos):
    IsobarasT={}
    IsobarasS={}
    for i in Isobaras_Valores:
        [valoresS, ValoresT] = isobara_Equilibrio(float(i), n_puntos)
        IsobarasT[str(i)]=ValoresT
        IsobarasS[str(i)]=valoresS
    return IsobarasT,IsobarasS

def Plot_Isobaras(ax,valoresS,ValoresT,Isobaras_Valores,Etiquetas):
    if  valoresS.any():
        Isobara_1, = ax.plot(valoresS, ValoresT, 'k', label=str(round(Isobaras_Valores, 1)))
        Isobara_1.set_lw(0.75)
        Isobara_1.set_ls('--')
        if Etiquetas:
            lines = plt.gca().get_lines()
            l1 = lines[-1]
            index_medio=(valoresS.size)/2
            x=valoresS[int(index_medio)]
            if str(round(Isobaras_Valores,1)):
                labelLine(l1, x, ha='left', va='bottom', align=True, fontsize=4)
            else:
                labelLine(l1, x,label=r'$P=${}'.format(l1.get_label())+' bar', ha='left', va='bottom', align=True, fontsize=4)

def Plot_Isentalpicas(ax,valoresS,ValoresT,Isentalpicas_Valores,Etiquetas):
    if  valoresS.any():
        Isentalpica_1, = ax.plot(valoresS, ValoresT, 'k', color='black', label=str(int(round(Isentalpicas_Valores,0))))
        Isentalpica_1.set_lw(0.75)
        Isentalpica_1.set_ls('--')
        if Etiquetas:
            lines = plt.gca().get_lines()
            l1 = lines[-1]
            x=(valoresS[0])
            if str(int(round(Isentalpicas_Valores,0))):
                labelLine(l1, x, ha='right', va='bottom', align=False, fontsize=6)
            #else:
                #labelLine(l1, x,label=r'$h=${}'.format(l1.get_label())+' kJ/kg', ha='left', va='bottom', align=True, fontsize=4)

[TemperaturasLS,EntropiasLS]=LineaLS(70)
[TemperaturasVS,EntropiasVS]=LineaVS(70)

Isobaras_Valores = np.logspace(np.log10(1), np.log10(500),18, endpoint=True)
[IsobarasT,IsobarasS]=Isobaras_Sobrecalentado(Isobaras_Valores,75,320,20)
[IsobarasT_equilibrio,IsobarasS_equilibrio]=Isobaras_Equilibrio(Isobaras_Valores,20)

Isentalpicas_sobrecalentado_Valores =np.arange(40, 460, 20)
[IsentalpicasT,IsentalpicasS]=Isentalpicas_Sobrecalentado(Isentalpicas_sobrecalentado_Valores,1, 550,40)

# Dibujo del digrama T-S Domo de saturación

figura=plt.figure()
ax=figura.add_axes([0.12,0.1,0.8,0.8])
lineaLS,=ax.plot(EntropiasLS,TemperaturasLS,'k');lineaLS.set_lw(1.5)
lineaVS,=ax.plot(EntropiasVS,TemperaturasVS,'k'); lineaVS.set_lw(1.5)
ax.set_xlabel('Entropía (kJ/kg·K)')
ax.set_ylabel('Temperatura (K)')
for i in Isobaras_Valores:
    Plot_Isobaras(ax, IsobarasS[str(i)], IsobarasT[str(i)],i,True)
    Plot_Isobaras(ax, IsobarasS_equilibrio[str(i)], IsobarasT_equilibrio[str(i)], i,False)

for i in Isentalpicas_sobrecalentado_Valores:
    Plot_Isentalpicas(ax, IsentalpicasS[str(i)], IsentalpicasT[str(i)],i,True)

# Insertar texto explicatorio
textoExplicatorio = 'Entalpías (kJ(kg): -. \nPresiones (bar): - - \nEstado Ref.: Líquido saturado'+'\n\t\t    '+\
                    r'$h= 0\ kJ/kg$'+'\n\t\t    '+ r'$s= 0\ kJ/kg \cdot K$'+ '\n\t\t    '+r'$P= 1\ atm$'

props = dict(boxstyle='round',facecolor='white')
ax.text(0.05, 0.95, textoExplicatorio, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=props)

ax.set_ylim(75, 325,emit=True, auto=False)
#ax.set_yticks((75,95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275,285,195))
ax.yaxis.set_ticks(np.arange(75, 325, 10))
plt.yticks (fontsize=8)
plt.xticks(fontsize=7)
ax.set_xticks((-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4))
plt.grid()
figura.show()
metadatos={ 'Author': 'Juan Carlos Domínguez', 'Title': 'Diagrama T-s (Hausen) para el aire',
    'Creator': 'Creado con CoolProp en Python'}
figura.savefig('Diagrama de Hausen T-s para el aire.pdf',format='pdf',quality=95,metadata=metadatos,
               dpi=300)
metadatos_png={ 'Author': 'Juan Carlos Domínguez', 'Title': 'Diagrama T-s (Hausen) para el aire',
    'Creation Time':'01-Dic-2019','Software': 'Creado con CoolProp en Python'}

figura.savefig('Diagrama de Hausen T-s para el aire.png',format='png',quality=95,metadata=metadatos_png,
               dpi=300)

