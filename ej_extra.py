import numpy as np
import matplotlib.pyplot as plt


def process_generated_bits_file(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        bits = [int(float(line.split(sep=' ')[0].strip())) for line in file.readlines()]
    return np.array(bits)

def process_osc_measurement_file(filename: str) -> tuple[np.ndarray, np.ndarray]:
    measurement = np.loadtxt(filename, dtype=float, delimiter=' ', skiprows=1)
    return measurement[:,0], measurement[:,1]

def get_dead_zone_range(time: np.ndarray, v: np.ndarray, dead_zone_min_length: float=300e-6) -> tuple[int, int]:
    T = time[1] - time[0]
    dead_zone_min_samples = int(dead_zone_min_length / T)
    threshold = (v.max() + v.min())/2

    for i, _ in enumerate(time):
        if i + dead_zone_min_samples >= len(time):
            break
        for j in range(i, i+dead_zone_min_samples):
            if v[j] > threshold:
                break
        else:   # encontre zona muerta
            while j+1 < len(time) and v[j+1] < threshold:
                j +=1
            return time[i], time[j]
    return np.nan, np.nan

if __name__ == '__main__':
    bits = process_generated_bits_file("measurements/BitsGenerados.txt")
    t, v = process_osc_measurement_file('measurements/NuevoDiaMaximaPotenciaEmisor4Analogico.txt')
    dz_start, dz_end = get_dead_zone_range(t[::10],v[::10])
    plt.plot(t, v)
    plt.axvline(dz_start, color='r')
    plt.axvline(dz_end, color='r')
    plt.show()