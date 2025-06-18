#librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from IPython.display import Audio, display

# Se carga y se muestra la información básica del audio.
archivo = 'AnalisisTextos.wav'
audio, sr = sf.read('AnalisisTextos.wav')

# Información del audio:
print("Vector de la señal segmentada:", audio)
print("Cantidad total de elementos en la muestra:", len(audio))
print("Frecuencia de Muestreo (Medida en MegaHerzios):", sr)
duracion = len(audio) / sr
print("Duración del audio:", duracion)

# Audio original:
print("Audio original (Sin modificaciones):")
display(Audio(audio, rate=sr))
# Se imprime la señal sonora del audio original.
plt.figure(figsize=(10, 4))
plt.plot(audio, color='red')
plt.title("Espectograma del audio:")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# Audio con velocidad duplicada:
print("Audio acelerado (velocidad duplicada / al X2):")
display(Audio(audio, rate=sr * 2))

audio_fast = librosa.resample(audio, orig_sr=sr, target_sr=sr * 2)
plt.figure(figsize=(10, 4))
plt.plot(audio_fast, color='blue')
plt.title("Forma de onda - Audio acelerado (x2)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# Audio con la velocidad reducida a la mitad:
print("Audio ralentizado - velocidad reducida a la mitad:")
display(Audio(audio, rate=sr // 2))

audio_slow = librosa.resample(audio, orig_sr=sr, target_sr=sr // 2)
plt.figure(figsize=(10, 4))
plt.plot(audio_slow, color='green')
plt.title("Forma de onda - Audio ralentizado (x0.5)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

#Reducción de la tasa de bits:
audio_baja_calidad = (audio * 2**3).astype(np.int8)
print("Audio con menor calidad (se reducen la cantidad de bits):")
display(Audio(audio_baja_calidad, rate=sr))

"""Se usará este codigo en caso de querer guardar el audio.
sf.write('AnalisisTexto2.wav', audio, samplerate=sr)
print("Archivo guardado como 'AnalisisTextos2.wav'")"""