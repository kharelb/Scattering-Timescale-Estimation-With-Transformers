import simpulse 
import matplotlib.pyplot as plt
import numpy as np
import json
import chime_frb_constants as const
from fitburst.routines.manipulate import downsample_2d
from syn_frbs import CreateSyntheticFRBs




def scattering_time_at_nu(scatt_time=10, ref_freq=400, nu=750):
    return scatt_time * ((nu/ref_freq)**-4)


def create_power_law_bursts(n=2000, starting_value=0):
    create_syn_frbs = CreateSyntheticFRBs()
    intrinsic_width_range = np.linspace(0.0001, 0.005)   # in seconds
    spectral_index_range = np.arange(-1, 2)
    scattering_timescale_range = np.linspace(0, 20, 50)   # in milliseconds
    rms_noise_range = np.arange(1, 51)
    arrival_time_range = np.linspace(1.5, 3.5)   # higher than 2 moves towards left from center and 2 is at center

    
    for j in range(n):
        i = starting_value + j
        pulse_create = True
        intrinsic_width = round(np.random.choice(intrinsic_width_range), 5)
        arrival_time = round(np.random.choice(arrival_time_range), 2)
    
        scattering_timescale = round(np.random.choice(scattering_timescale_range), 2)

        # if scattering_timescale >= 5:
        #     arrival_time = round(np.random.choice(np.linspace(2.2, 3.5, 20)), 2)
            

        scattering_timescale_seconds = scattering_timescale / 1000
    
        if intrinsic_width >= round(scattering_timescale_seconds, 5):
            scattering_timescale = 0

        spectral_index = round(np.random.choice(spectral_index_range), 2)
   
        fluence = np.random.choice([0.001, 0.0015, 0.0020, 0.0025, 0.003])

        noise_rms = np.random.choice(rms_noise_range)


        while pulse_create:
            

            syn_frbs_noise, syn_frbs_noiseless, snr = create_syn_frbs.power_law_spectrum(fluence_jy_s=fluence, spectral_index=spectral_index, intrinsic_width_s=intrinsic_width, noise_rms=noise_rms, scat_time_ms=scattering_timescale)

            if (snr <= 37) and (noise_rms > 1):
                noise_rms -= 1
                continue

            elif (snr <= 37) and (noise_rms <= 1):
                noise_rms -= 0.1
                continue

            elif snr >= 155:
                noise_rms += 1
                continue

            else:
                print(f"{i} -- SI: {spectral_index}, Fluence: {fluence}, Intrinsic Width: {intrinsic_width}, Scat Time: {scattering_timescale}", end='\r')
                
                pulse_parameters = {
                    "intrinsic_width(s)": float(intrinsic_width),
                    "fluence": float(fluence),
                    "spectral index": float(spectral_index),
                    "scattering time(ms)": float(scattering_timescale),
                    "rms noise": float(noise_rms),
                    "snr": float(snr)
                }
                np.save(f"synthetic_dynamic_spectrum/ds_{i}.npy", syn_frbs_noise)
                np.save(f"synthetic_bursts_noiseless/noiseless_{i}.npy", syn_frbs_noiseless)

                with open(f"synthetic_parameters/parameter_{i}.json", 'w') as f:
                    json.dump(pulse_parameters, f)
                pulse_create = False

        

    
    
    