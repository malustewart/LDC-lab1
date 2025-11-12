import numpy as np
import matplotlib.pyplot as plt
import argparse


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
            return i, j
    return np.nan, np.nan

def get_bit_and_error_count(t, v, mbps, bits, threshold=None):
    if threshold == None:
        threshold = (np.min(v)+ np.max(v))/2

    #skip first and last 0 bits which cannot be distinguished from dead zone:
    bits = np.trim_zeros(bits, trim='fb')

    _, dz_end = get_dead_zone_range(t, v)

    T = t[1]-t[0]
    step = 1/(1e6*mbps*T)
    bits_in_sample = int((len(v) - dz_end) / step)
    bits = bits[:bits_in_sample]
    sample_points = [dz_end + int((0.5+i)*step) for i, _ in enumerate(bits)]

    # despues de la zona muerta
    error_times = [time for time, meas, bit in zip(t[sample_points], v[sample_points], bits) if bit != int(meas > threshold)]
    
    # plots real bits:
    for i, b in enumerate(bits[:-1]):
        if b:
            start_t = t[dz_end + int((i)*step)]
            end_t = t[dz_end + int((i+1)*step)]
            plt.axvspan(start_t, end_t, color='#e0e0e0')

    # plot measured signal
    plt.plot(t, v, color='#aaaadd', linewidth=0.7, zorder=99)

    # plot decision threshold
    plt.axhline(threshold, color='k', alpha=0.7, linewidth=0.7, label="Decision threshold")

    # plot sample points
    plt.scatter(t[sample_points], v[sample_points], color='b', label="Decision samples", zorder=100)

    # plot erronous sample points
    for error_time in error_times:
        plt.scatter(error_time, color='r', alpha=1, zorder=101)

    return len(error_times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--mbps",
        type=float,
        default=1.0,
        help="Bandwidth in Mbps (default: 1.0)"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="List of input files"
    )

    parser.add_argument(
        "--bitfile",
        metavar="FILE",
        help="File with generated bits"
    )

    args = parser.parse_args()
    mbps = args.mbps
    files = args.files
    bitfile = args.bitfile

    print(f"Using bandwidth: {args.mbps} Mbps")
    print(f"Parsing following measurements: {args.files}")
    print(f"Comparing measurements with following bits: {args.bitfile}")

    for file in files:
        plt.figure()
        bits = process_generated_bits_file(bitfile)
        t, v = process_osc_measurement_file(file)
        error_count = get_bit_and_error_count(t, v, mbps, bits)

        print(error_count)
        plt.show()