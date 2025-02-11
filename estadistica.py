import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Definir la población: una distribución normal con media 50 y desviación estándar 15
np.random.seed(42)  # Para reproducibilidad
poblacion = np.random.normal(loc=50, scale=15, size=10000)  # Población de 10,000 individuos

# El parámetro real de la población (desconocido en la práctica)
media_poblacional_real = np.mean(poblacion)
print(f"Media real de la población: {media_poblacional_real:.2f}")

# Tomar una muestra aleatoria de la población
tamano_muestra = 100
muestra = np.random.choice(poblacion, size=tamano_muestra, replace=False)

# Estimar el parámetro (media poblacional) usando la muestra
media_muestral = np.mean(muestra)
print(f"Estimación de la media poblacional basada en la muestra: {media_muestral:.2f}")

# Función de verosimilitud para la media de la población
def verosimilitud(media, datos):
    """Calcula la verosimilitud de una media dada los datos muestrales."""
    verosimilitudes = stats.norm.pdf(datos, loc=media, scale=np.std(datos, ddof=1))
    return np.prod(verosimilitudes)  # Producto de las probabilidades individuales

# Evaluar la verosimilitud en un rango de valores de la media
medias_evaluadas = np.linspace(40, 60, 100)
verosimilitudes = [verosimilitud(media, muestra) for media in medias_evaluadas]

# Graficar la función de verosimilitud
plt.figure(figsize=(8, 5))
plt.plot(medias_evaluadas, verosimilitudes, label="Función de verosimilitud", color="blue")
plt.axvline(media_poblacional_real, color='red', linestyle='dashed', label="Media real de la población")
plt.axvline(media_muestral, color='green', linestyle='dotted', label="Media estimada de la muestra")
plt.xlabel("Media Poblacional")
plt.ylabel("Verosimilitud")
plt.legend()
plt.title("Función de Verosimilitud para la Media Poblacional")
plt.show()
