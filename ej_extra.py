import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import erfc
import datetime

def params_to_str(params:dict, sep:str = " - "):
    return sep.join(f"{k}: {v}" for k,v in params.items())

def process_generated_bits_file(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        bits = [int(float(line.split(sep=' ')[0].strip())) for line in file.readlines()]
    return np.array(bits+bits)

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

def get_threshold(v_0, v_1):
    sigma_0, sigma_1 = np.std(v_0), np.std(v_1)
    mu_0, mu_1 = np.mean(v_0), np.mean(v_1)
    threshold = (sigma_0*mu_1 + sigma_1*mu_0)/(sigma_0+sigma_1)
    return threshold

def get_q(v_0, v_1):
    sigma_0, sigma_1 = np.std(v_0), np.std(v_1)
    mu_0, mu_1 = np.mean(v_0), np.mean(v_1)
    q = (mu_1 + mu_0)/(sigma_0+sigma_1)
    return q

def get_ber_estimate(q):
    ber = 0.5*(erfc(q/np.sqrt(2)))
    return ber

def get_sample_points(v, t, mbps, bits, dz_start, dz_end):
    bits = np.trim_zeros(bits, trim='fb')

    T = np.abs(t[1]-t[0])
    step = 1/(1e6*mbps*T)
    bits_in_sample_before = int(dz_start / step)
    bits_before_dz = bits[:-bits_in_sample_before:-1]
        
    sp_0_before = np.array([dz_start - int((0.5+i)*step) for i, bit in enumerate(bits_before_dz) if not bit], dtype=np.int64)
    sp_1_before = np.array([dz_start - int((0.5+i)*step) for i, bit in enumerate(bits_before_dz) if bit], dtype=np.int64)

    bits_in_sample_after = int((len(v) - dz_end) / step)
    bits_after_dz = bits[:bits_in_sample_after]
    sp_0_after = np.array([dz_end + int((0.5+i)*step) for i, bit in enumerate(bits_after_dz) if not bit], dtype=np.int64)
    sp_1_after = np.array([dz_end + int((0.5+i)*step) for i, bit in enumerate(bits_after_dz) if bit], dtype=np.int64)

    sp_0 = np.concatenate((sp_0_before, sp_0_after), dtype=np.int64)
    sp_1 = np.concatenate((sp_1_before, sp_1_after), dtype=np.int64)

    return sp_0, sp_1

def get_sample_points_ok_and_errors(v, sp_0, sp_1):
    v_0 = v[sp_0]
    v_1 = v[sp_1]

    sp_0_read_0 = np.array([sp for sp, v in zip(sp_0, v_0) if v < threshold], dtype=np.int64)
    sp_0_read_1 = np.array([sp for sp, v in zip(sp_0, v_0) if v >= threshold], dtype=np.int64)    # read 1 when it was a 0
    sp_1_read_1 = np.array([sp for sp, v in zip(sp_1, v_1) if v >= threshold], dtype=np.int64)
    sp_1_read_0 = np.array([sp for sp, v in zip(sp_1, v_1) if v < threshold], dtype=np.int64)     # read 0 when it was a 1

    return sp_0_read_0, sp_0_read_1, sp_1_read_1, sp_1_read_0

def get_signal_power(v, t):
    energy = np.sum(np.abs(v)**2)
    delta_t = t[-1]-t[0]
    power = energy/delta_t
    return power

def plot_signal(v, t, sp_0_read_0, sp_0_read_1, sp_1_read_1, sp_1_read_0, bits, dz_start, dz_end, filename=None):
    
    bits = np.trim_zeros(bits, 'fb')
    
    plt.figure()
    
    # plots real bits:
    T = np.abs(t[1]-t[0])
    step = 1/(1e6*mbps*T)
    bits_in_sample_after = int((len(v) - dz_end) / step)
    bits_after_dz = bits[:bits_in_sample_after]
    bits_in_sample_before = int(dz_start / step)
    bits_before_dz = bits[:-bits_in_sample_before:-1]

    for i, b in enumerate(bits_after_dz[:-1]):
        if b:
            start_t = t[dz_end + int((i)*step)]
            end_t = t[dz_end + int((i+1)*step)]
            plt.axvspan(start_t, end_t, color='#e0e0e0', zorder=0)
    for i, b in enumerate(bits_before_dz[:-1]):
        if b:
            start_t = t[dz_start - int((i)*step)]
            end_t = t[dz_start - int((i+1)*step)]
            plt.axvspan(start_t, end_t, color='#e0e0e0', zorder=0)

    # plot measured signal
    plt.plot(t, v, color='#aaaadd', linewidth=0.7, zorder=50)

    # plot decision threshold
    plt.axhline(threshold, color='k', alpha=0.7, linewidth=0.7, label="Decision threshold", zorder=99)

    # plot sample points
    sp_ok = np.concatenate((sp_0_read_0, sp_1_read_1), dtype=np.int64)
    sp_error = np.concatenate((sp_0_read_1, sp_1_read_0), dtype=np.int64)
    plt.scatter(t[sp_ok], v[sp_ok], color='b', label="Decision samples", zorder=100)
    plt.scatter(t[sp_error], v[sp_error], color='r', label="Bit errors", zorder=101)

    plt.title(params_to_str(params), fontsize=8)
    lgd = plt.legend(bbox_to_anchor=(1.1, 0.5), loc='center left')
    plt.tight_layout()
    plt.tick_params(axis='x', rotation=45)
    plt.xlabel("t[s]")
    plt.ylabel("V[V]")
    if filename:
        plt.savefig(f"figs/{filename}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        xlim = plt.xlim()
        plt.xlim((0.01528, 0.01530))
        plt.savefig(f"figs/{filename}_detail.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.xlim(xlim)


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
        "--logfile",
        type=str,
        default="log.txt",
        help="log to store results"
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
    logfile = args.logfile

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log = []
    log.append(f"\n\n\n****===================****\n    {timestamp}\n****===================****")
    log.append(f"Using bandwidth: {args.mbps} Mbps")
    log.append(f"Parsing following measurements: {args.files}")
    log.append(f"Comparing measurements with following bits: {args.bitfile}")

    for file in files:
        log.append(f"\n*** Processing {file}... ***")
        bits = process_generated_bits_file(bitfile)
        t, v = process_osc_measurement_file(file)
        params = {"Filename": file}
        
        # Measure BER directly
        dz_start, dz_end = get_dead_zone_range(t, v)
        sp_0, sp_1 = get_sample_points(v, t, mbps, bits, dz_start, dz_end)
        threshold = get_threshold(v[sp_0], v[sp_1])
        sp_0_read_0, sp_0_read_1, sp_1_read_1, sp_1_read_0 = get_sample_points_ok_and_errors(v, sp_0, sp_1)
        error_count = len(sp_0_read_1) + len(sp_1_read_0)
        ok_count = len(sp_0_read_0) + len(sp_1_read_1)
        bit_count = error_count + ok_count
        log.append(f"Measured BER: {error_count}/{bit_count} = {error_count/bit_count}")
        
        # Calculate BER through q
        q = get_q(v[sp_0], v[sp_1])
        ber_estimate = get_ber_estimate(q)
        log.append(f"q: {q}")
        log.append(f"Estimated BER: {ber_estimate}")

        plot_signal(v, t, sp_0_read_0, sp_0_read_1, sp_1_read_1, sp_1_read_0, bits, dz_start, dz_end, file)

        # Calculate power:
        power = get_signal_power(v, t)
        log.append(f"Electric power (W): {power}V²/R")

    with open(logfile, 'a') as f:
        for line in log:
            f.write(line+"\n")

    plt.show()