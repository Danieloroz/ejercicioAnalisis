import math
import numpy as np
import matplotlib.pyplot as plt

def volumen_tanque(h, R):
    """Calcula el volumen de un tanque esférico"""
    return (math.pi * h**2 * (3*R - h)) / 3

def newton_raphson(R, volumen_objetivo, h_inicial, tolerancia=0.003/100, max_iteraciones=100):
    """
    Metodo de Newton-Raphson con registro de convergencia

    Retorna:
    - Altura final
    - Lista de estimaciones de altura
    - Lista de errores relativos
    - Número de iteraciones
    """
    def f(h):
        """Función para encontrar raíz: diferencia entre volumen calculado y objetivo"""
        return volumen_tanque(h, R) - volumen_objetivo

    def df(h):
        """Derivada de la función de volumen"""
        return (math.pi * (6*R*h - 3*h**2)) / 3

    h = h_inicial
    iteraciones = 0

    # Listas para tracking
    alturas = [h]
    errores = [1.0]  # Primer error inicial
    print("\nMétodo de Newton-Raphson:")
    print("Iteración | Altura (m) | Error relativo (%)")

    while iteraciones < max_iteraciones:
        f_h = f(h)
        df_h = df(h)

        # Evitar división por cero
        if abs(df_h) < 1e-10:
            break

        # Calcular nueva estimación de altura
        h_nuevo = h - f_h / df_h

        # Calcular error relativo
        error_relativo = abs((h_nuevo - h) / h_nuevo) * 100

        # Registrar altura y error
        alturas.append(h_nuevo)
        errores.append(error_relativo)
        print(f"{iteraciones+1:<9} | {h_nuevo:<10.6f} | {error_relativo:<10.6f}")

        # Verificar convergencia
        if error_relativo < tolerancia * 100:
            return h_nuevo, alturas, errores, iteraciones + 1

        h = h_nuevo
        iteraciones += 1

    return None, alturas, errores, iteraciones

def secante(R, volumen_objetivo, h0, h1, tolerancia=0.003/100, max_iteraciones=100):
    """
    Metodo de la Secante con registro de convergencia

    Retorna:
    - Altura final
    - Lista de estimaciones de altura
    - Lista de errores relativos
    - Número de iteraciones
    """
    def f(h):
        """Función para encontrar raíz: diferencia entre volumen calculado y objetivo"""
        return volumen_tanque(h, R) - volumen_objetivo

    iteraciones = 0

    # Listas para tracking
    alturas = [h0, h1]
    errores = [1.0, 1.0]  # Errores iniciales

    print("\nMétodo de la Secante:")
    print("Iteración | Altura (m) | Error relativo (%)")
    while iteraciones < max_iteraciones:
        f_h0 = f(alturas[-2])
        f_h1 = f(alturas[-1])

        # Evitar división por cero
        if abs(f_h1 - f_h0) < 1e-10:
            break

        # Calcular nueva estimación de altura
        h_nuevo = alturas[-1] - f_h1 * (alturas[-1] - alturas[-2]) / (f_h1 - f_h0)

        # Calcular error relativo
        error_relativo = abs((h_nuevo - alturas[-1]) / h_nuevo) * 100

        # Registrar altura y error
        alturas.append(h_nuevo)
        errores.append(error_relativo)
        print(f"{iteraciones+1:<9} | {h_nuevo:<10.6f} | {error_relativo:<10.6f}")

        # Verificar convergencia
        if error_relativo < tolerancia * 100:
            return h_nuevo, alturas, errores, iteraciones + 1

        iteraciones += 1

    return None, alturas, errores, iteraciones

def graficar_convergencia(metodo_nr, metodo_secante):
    """
    Grafica la convergencia de ambos métodos
    """
    plt.figure(figsize=(15, 6))

    # Gráfica de alturas
    plt.subplot(1, 2, 1)
    plt.plot(metodo_nr[1], label='Newton-Raphson', marker='o')
    plt.plot(metodo_secante[1], label='Método de la Secante', marker='s')
    plt.title('Convergencia de Alturas')
    plt.xlabel('Iteraciones')
    plt.ylabel('Altura (m)')
    plt.legend()
    plt.grid(True)

    # Gráfica de errores
    plt.subplot(1, 2, 2)
    plt.semilogy(metodo_nr[2], label='Newton-Raphson', marker='o')
    plt.semilogy(metodo_secante[2], label='Método de la Secante', marker='s')
    plt.title('Convergencia de Errores')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error Relativo (Log Scale)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def graficar_volumen(R, volumen_objetivo, altura_nr, altura_secante):
    """
    Grafica la función de volumen del tanque y marca las soluciones encontradas.
    """
    # Crear rango de alturas para graficar
    alturas = np.linspace(0, R, 200)
    volumenes = [volumen_tanque(h, R) for h in alturas]

    # Crear la figura
    plt.figure(figsize=(12, 7))
    plt.plot(alturas, volumenes, label='Función de Volumen', color='blue')
    plt.title('Volumen del Tanque Esférico', fontsize=14)
    plt.xlabel('Altura (m)', fontsize=12)
    plt.ylabel('Volumen (m³)', fontsize=12)

    # Línea horizontal para el volumen objetivo
    plt.axhline(y=volumen_objetivo, color='r', linestyle='--', label='Volumen Objetivo')

    # Marcar soluciones de los métodos con mayor visibilidad
    plt.scatter([altura_nr], [volumen_tanque(altura_nr, R)], color='green',
                s=100, edgecolors='black', linewidth=2,
                label='Newton-Raphson', zorder=5)
    plt.scatter([altura_secante], [volumen_tanque(altura_secante, R)], color='purple',
                s=100, edgecolors='black', linewidth=2,
                label='Método de la Secante', zorder=5)

    # Agregar anotaciones con las alturas
    plt.annotate(f'NR: {altura_nr:.4f} m',
                 (altura_nr, volumen_tanque(altura_nr, R)),
                 xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Secante: {altura_secante:.4f} m',
                 (altura_secante, volumen_tanque(altura_secante, R)),
                 xytext=(10, -15), textcoords='offset points')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# Datos del problema
R = 3  # Radio del tanque [m]
volumen_objetivo = 30  # Volumen deseado [m³]

# Resolver con Newton-Raphson
resultado_nr = newton_raphson(R, volumen_objetivo, h_inicial=1)
altura_nr = resultado_nr[0]
print("Newton-Raphson:")
print(f"Altura: {altura_nr:.4f} m")
print(f"Iteraciones: {resultado_nr[3]}")
print(f"Volumen calculado: {volumen_tanque(altura_nr, R):.4f} m³")

# Resolver con Metodo de la Secante
resultado_secante = secante(R, volumen_objetivo, h0=1, h1=2)
altura_secante = resultado_secante[0]
print("\nMétodo de la Secante:")
print(f"Altura: {altura_secante:.4f} m")
print(f"Iteraciones: {resultado_secante[3]}")
print(f"Volumen calculado: {volumen_tanque(altura_secante, R):.4f} m³")

# Graficar volumen
graficar_volumen(R, volumen_objetivo, altura_nr, altura_secante)
# Graficar convergencia
graficar_convergencia(resultado_nr, resultado_secante)