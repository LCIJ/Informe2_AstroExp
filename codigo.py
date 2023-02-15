from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chisquare

'''
    CURVA DE ROTACIÓN
    '''

# Se abre el cubo de datos y después se extraen los datos y el header.

cubo = fits.open(
    "/home/javier/Escritorio/Semestres Universidad/Primavera 2022/Astronomía Experimental/Informe 2/20221006105173B7553D38160542__Cubo_de_Datos.fits")
data = cubo[0].data
 header = cubo[0].header

  print(repr(header))

   # Lo que hace esta función es traducir los índices discretos que indexan la variable
   # (l,b o v_LSR) a sus valores reales. Por ejemplo, si en el header la variable v_LSR
   # es la 1, y el header lo cargan en el objeto “h”:
   # v = values(h,1) dará como output un vector con los valores reales que toma v_LSR.

   def values(h, j):
        N = h['NAXIS'+str(j)]
        val = np.zeros(N)
        for i in range(0, N):
            val[i] = (i+1-float(h['CRPIX'+str(j)])) * \
                float(h['CDELT'+str(j)]) + float(h['CRVAL'+str(j)])
        return val

    # Se determina el rango de la velocidad, longitud y latitud.

    velocidad = values(header, 1)
    longitud = values(header, 2)
    latitud = values(header, 3)

    print('Longitudes = {' + str(np.min(longitud)) +
          ', ' + str(np.max(longitud)) + '}')
    print('Latitudes = {' + str(np.min(latitud)) +
          ', ' + str(np.max(latitud)) + '}')
    print('Velocidades = {' + str(np.min(velocidad)) +
          ', ' + str(np.max(velocidad)) + '}')
    print("Es interesante notar que por el rango de las longitudes, los datos del cubo corresponden al cuarto cuadrante del plano galáctico")

    # Se calcula el RMS promedio de las velocidades, para eso se tiene que para
    # una longitud l fija, se recorre latitud b y se calcula el rms de las velocidades.
    # Cada rms es almacenado para posteriormente calcular el RMS promedio
    # El RMS promedio va a ser usado para evitar almacenar datos que sean ruido

    RMS = []
    for lon in range(len(longitud)):
        for lat in range(len(latitud)):
            T = data[lat][lon][:]
            rms = np.sqrt(np.mean(T**2))
            RMS.append(rms)

    mean_rms = np.mean(RMS)
    print('RMS = ', mean_rms)

    # Se hace una tabla para todas las combinaciones de los valores de l, b,
    # con el resultado de la velocidad tangencial y/o terminal en cada caso

    datos = []
    b = 0
    l = 0
    for lon in range(len(longitud)):
        l = longitud[lon]
        vb = []
        for lat in range(len(latitud)):
            vb_i = []
            for vel in range(len(velocidad)):
                # Descartamos el error usando un margen de 3 sigma
                if data[lat][lon][vel] >= 3*mean_rms:
                    vb_i.append(velocidad[vel])
            if len(vb_i) == 0:
                vb.append(np.inf)
            else:
                # Se obtiene la velocidad minima para cada latitud b
                vb.append(np.min(vb_i))
        b = latitud[np.argmin(vb)]
        datos.append([l, b, np.min(vb)])

    datos_df = pd.DataFrame(datos, columns=['l', 'b', 'v_ter'])

    # Se grafica la velocidad terminal con respecto a la longitud

    plt.plot(datos_df['l'], datos_df['v_ter'], 'g.')
    plt.xlim(348, 300.)
    plt.ylim(0, datos_df['v_ter'].min())
    plt.xlabel(r'$Longitud \,\,\, [^o]$', fontsize=15)
    plt.ylabel(r'$V_{ter} \,\,\, [\frac{km}{s}]$', fontsize=15)
    plt.title('Velocidad terminal ' +
              r'$V_{ter}$' + ' en\nfunción de la longitud', fontsize=20)
    plt.grid(True, which="both", ls="-")
    plt.savefig("v_ter vs longitud.png", bbox_inches='tight')
    plt.show()

    R_0 = 8.5  # kPc
    v_sol = 220  # km/s
    omega_sol = v_sol / R_0

    # Se obtienen la velocidad orbital y la velocidad rotacional

    datos_df['sin_l'] = np.sin(datos_df['l'] * (np.pi/180.))
    datos_df['v_orb'] = datos_df['v_ter'] * \
        (np.abs(datos_df['sin_l']) / datos_df['sin_l']) + \
        (v_sol * np.abs(datos_df['sin_l']))
    datos_df['v_rot'] = (datos_df['v_ter'] /
                         (R_0 * datos_df['sin_l'])) + omega_sol
    # Al estar en el cuarto cuadrante, R = - R0 * sin(l)
    datos_df['R'] = - R_0 * datos_df['sin_l']
    datos_df.head()

    # Se grafica la velocidad orbital

    plt.plot(datos_df['R'], datos_df['v_orb'], 'r.')
    plt.title("Velocidad orbital " +
              r'$V_{orb}$' + " en función de\nla distancia galactocéntrica R", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$V_{orb} \,\,\, [\frac{km}{s}]$', fontsize=15)
    plt.grid(True, which="both", ls="-")
    plt.savefig("v_orb vs r.png", bbox_inches='tight')
    plt.show()

    # Se obtiene la velocidad rotacional, de la cual se extrae la famosa curva de rotación

    plt.plot(datos_df['R'], datos_df['v_rot'], 'b.')
    plt.title("Velocidad rotacional " + r'$\omega (R)$' +
              " en función\nde la distancia galactocéntrica R", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$\omega (R) \,\,\, [\frac{rad}{s}]$', fontsize=15)
    plt.legend(["Curva de Rotación"])
    plt.grid(True, which="both", ls="-")
    plt.savefig("v_rot vs r.png", bbox_inches='tight')
    plt.show()

    '''
    CORRUGACIÓN DEL PLANO
    '''

    # Se obtiene la corrugación del plano, tanto con Z = R0 * cos(l) * tan(b_max)
    # como con Z = R0 * cos(l) * b_rad

    datos_df['cos_l'] = np.cos(datos_df['l'] * (np.pi/180.))
    datos_df['tan_bmax'] = np.tan(datos_df['b'] * (np.pi/180.))
    datos_df['Z_tan'] = R_0 * datos_df['cos_l'] * datos_df['tan_bmax']
    datos_df['Z'] = R_0 * datos_df['cos_l'] * (datos_df['b'] * (np.pi/180.))
    datos_df.head()

    # Se grafica la corrugación del plano con respecto a R

    plt.plot(datos_df['R'], datos_df['Z_tan'], 'y.',
             label=r'$Z(R) = R_0 \cdot cos(l \cdot \frac{\pi}{180}) \cdot tan(b_{max} \cdot \frac{\pi}{180})$')
    plt.plot(datos_df['R'], datos_df['Z'], 'm.',
             label=r'$Z(R) = R_0 \cdot cos(l \cdot \frac{\pi}{180}) \cdot (b \cdot \frac{\pi}{180})$')
    plt.hlines(0, datos_df['R'].min(), datos_df['R'].max(), colors='k')
    plt.title("Distancia vertical al plano galáctico " + r'$Z(R)$' +
              " en\nfunción de la distancia galactocéntrica R", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$Z (R) \,\,\, [kpc]$', fontsize=15)
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig("corrugacion vs r.png", bbox_inches='tight')
    plt.show()

    '''
    MODELOS CURVA DE ROTACIÓN
    '''

    G = 4.302e-6

    # Modelo de curva de rotación con una masa puntual.

    def masapuntual(R, M0):
        M = M0
        v = np.sqrt((G * M) / R**3)
        return v

    mpuntual, covmpuntual = curve_fit(
        masapuntual, datos_df['R'], datos_df['v_rot'])

    plt.plot(datos_df['R'], datos_df['v_rot'], 'b.', label='Datos obtenidos')
    plt.plot(datos_df['R'], masapuntual(datos_df['R'],
             mpuntual[0]), 'r-', label='Modelo ajustado')
    plt.title("Velocidad rotacional " + r'$\omega (R)$' +
              " en función de\nla distancia galactocéntrica R ajustada\ncomo masa puntual", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$\omega (R) \,\,\, [\frac{rad}{s}]$', fontsize=15)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("modelo1.png", bbox_inches='tight')
    plt.show()

    print('M_0 = ', mpuntual[0])
    print("-----------------")

    # Modelo de curva de rotación con un disco uniforme.

    def disco_uniforme(R, S):
        M = np.pi * (R**2) * S
        v = np.sqrt((G * M) / R**3)
        return v

    dis_uniforme, covdis_uniforme = curve_fit(
        disco_uniforme, datos_df['R'], datos_df['v_rot'])

    plt.plot(datos_df['R'], datos_df['v_rot'], 'b.', label='Datos obtenidos')
    plt.plot(datos_df['R'], disco_uniforme(datos_df['R'],
             dis_uniforme[0]), 'r-', label='Modelo ajustado')
    plt.title("Velocidad rotacional " + r'$\omega (R)$' +
              " en función de\nla distancia galactocéntrica R ajustada\ncomo disco uniforme", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$\omega (R) \,\,\, [\frac{rad}{s}]$', fontsize=15)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("modelo2.png", bbox_inches='tight')
    plt.show()

    print('S = ', dis_uniforme[0])
    print("-----------------")

    # Modelo de curva de rotación con una esfera uniforme.

    def esfera_uniforme(R, rho):
        M = (4/3) * np.pi * (R**3) * rho
        v = np.sqrt((G * M) / R**3)
        return v

    esf_uniforme, covesf_uniforme = curve_fit(
        esfera_uniforme, datos_df['R'], datos_df['v_rot'])

    plt.plot(datos_df['R'], datos_df['v_rot'], 'b.', label='Datos obtenidos')
    plt.plot(datos_df['R'], esfera_uniforme(datos_df['R'],
             esf_uniforme[0]), 'r-', label='Modelo ajustado')
    plt.title("Velocidad rotacional " + r'$\omega (R)$' +
              " en función de\nla distancia galactocéntrica R ajustada\ncomo esfera uniforme", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$\omega (R) \,\,\, [\frac{rad}{s}]$', fontsize=15)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("modelo3.png", bbox_inches='tight')
    plt.show()

    print('rho = ', esf_uniforme[0])
    print("-----------------")

    # Modelo de curva de rotación con una masa puntual + esfera uniforme.

    def masa_esfera(R, M0, rho):
        M = M0 + ((4/3) * np.pi * (R**3) * rho)
        v = np.sqrt((G * M) / R**3)
        return v

    mas_esf, covmas_esf = curve_fit(
        masa_esfera, datos_df['R'], datos_df['v_rot'])

    plt.plot(datos_df['R'], datos_df['v_rot'], 'b.', label='Datos obtenidos')
    plt.plot(datos_df['R'], masa_esfera(datos_df['R'],
             mas_esf[0], mas_esf[1]), 'r-', label='Modelo ajustado')
    plt.title("Velocidad rotacional " + r'$\omega (R)$' +
              " en función de\nla distancia galactocéntrica R ajustada\ncomo masa puntual + esfera uniforme", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$\omega (R) \,\,\, [\frac{rad}{s}]$', fontsize=15)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("modelo4.png", bbox_inches='tight')
    plt.show()

    print('M_0 = ', mas_esf[0])
    print('rho = ', mas_esf[1])
    print("-----------------")

    # Modelo de curva de rotación con una masa puntual + disco uniforme.

    def masa_disco(R, M0, S):
        M = M0 + (np.pi * (R**2) * S)
        v = np.sqrt((G * M) / R**3)
        return v

    mas_dis, covmas_dis = curve_fit(
        masa_disco, datos_df['R'], datos_df['v_rot'])

    plt.plot(datos_df['R'], datos_df['v_rot'], 'b.', label='Datos obtenidos')
    plt.plot(datos_df['R'], masa_disco(datos_df['R'],
             mas_dis[0], mas_dis[1]), 'r-', label='Modelo ajustado')
    plt.title("Velocidad rotacional " + r'$\omega (R)$' +
              " en función de\nla distancia galactocéntrica R ajustada\ncomo masa puntual + disco uniforme", fontsize=20)
    plt.xlabel(r'$R \,\,\, [kpc]$', fontsize=15)
    plt.ylabel(r'$\omega (R) \,\,\, [\frac{rad}{s}]$', fontsize=15)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("modelo5.png", bbox_inches='tight')
    plt.show()

    print('M_0 = ', mas_dis[0])
    print('S = ', mas_dis[1])
    print("-----------------")

    # Para analizar cual se asemeja más a los datos obtenidos, calcularemos la prueba de chi cuadrado a cada curva.

    yr = datos_df['v_rot']  # Datos reales
    ym1 = masapuntual(datos_df['R'], mpuntual[0])  # Modelo 1
    ym2 = disco_uniforme(datos_df['R'], dis_uniforme[0])  # Modelo 2
    ym3 = esfera_uniforme(datos_df['R'], esf_uniforme[0])  # Modelo 3
    ym4 = masa_esfera(datos_df['R'], mas_esf[0], mas_esf[1])  # Modelo 4
    ym5 = masa_disco(datos_df['R'], mas_dis[0], mas_dis[1])  # Modelo 5

    chisquare1 = sum(((yr - ym1)**2) / ym1)
    chisquare2 = sum(((yr - ym2)**2) / ym2)
    chisquare3 = sum(((yr - ym3)**2) / ym3)
    chisquare4 = sum(((yr - ym4)**2) / ym4)
    chisquare5 = sum(((yr - ym5)**2) / ym5)

    print('Chi cuadrado modelo 1 = ', chisquare1)
    print('Chi cuadrado modelo 2 = ', chisquare2)
    print('Chi cuadrado modelo 3 = ', chisquare3)
    print('Chi cuadrado modelo 4 = ', chisquare4)
    print('Chi cuadrado modelo 5 = ', chisquare5)

    '''
    MAPA DEL CUBO
    '''

    # Mapa del cubo con la velocidad integrada.
    # Para integrar la velocidad, el eje de velocidad es colapsado para que quede un fits en 2 dimensiones: latitud y longitud.

    data_2 = np.zeros((len(latitud), len(longitud)))
    for i in range(len(latitud)):
        for j in range(len(longitud)):
            data_2[i][j] = np.trapz(data[i][j][:])

    plt.figure(figsize=(18, 3))
    plt.imshow(data_2, interpolation='nearest', aspect='auto', cmap='jet', extent = [min(longitud), max(longitud), min(latitud), max(latitud)])
    plt.title("Mapa del cubo con la velocidad integrada", fontsize=20)
    plt.xlabel(r'$Longitud \,\,\, [^o]$', fontsize=15)
    plt.ylabel(r'$Latitud \,\,\, [^o]$', fontsize=15)
    clb = plt.colorbar()
    clb.set_label(r'$Velocidad \,\,\, [\frac{km}{s}$]', labelpad=-25, y=1.15, rotation=0, fontsize = 15)
    plt.gca().invert_xaxis()  # Por defecto el eje esta al revés
    plt.savefig("mapa_cubo.png", bbox_inches='tight')
    plt.show()
