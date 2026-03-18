"""
cosmic-piano: Generate audio samples of various astrophysical phenomena from data.
Fits seamlessly for playing with a MIDI-compatable instrument or a regular computer keyboard.
"""

import os, sys, argparse, warnings
import numpy as np
from scipy.signal import butter, sosfilt, resample as scipy_resample
import soundfile as sf

warnings.filterwarnings("ignore")

SAMPLE_RATE    = 44100
SAMPLE_DURATION = 4.0
SAMPLES_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# ═══════════════════════════════════════════════════════════════════════════════
#  OBJECT CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════

OBJECTS = {

    # ── GRAVITATIONAL WAVES ────────────────────────────────────────────────────
    36: {
        "name": "GW150914", "type": "grav_wave",
        "label": "Black Hole Merger — GW150914",
        "data_source": "LIGO Open Science Center — real H1 strain, GPS 1126259462",
        "description": "First gravitational wave detected (14 Sep 2015). Two black holes of 36\n"
                       "and 29 solar masses merged 399 Mpc (1.3 billion ly) away. You are hearing\n"
                       "the actual distortion of spacetime recorded by LIGO Hanford.",
        "params": {
            "gps_time": 1126259462.4, "detector": "H1",
            "m1_solar": 36, "m2_solar": 29,
            "f_start": 35, "f_end": 350, "chirp_duration": 0.45,
        },
    },
    38: {
        "name": "GW170817", "type": "grav_wave",
        "label": "Neutron Star Merger — GW170817",
        "data_source": "LIGO Open Science Center — real H1 strain, GPS 1187008882",
        "description": "Two neutron stars merging (17 Aug 2017). Unlike black hole mergers\n"
                       "this one also flashed as a kilonova — seen by 70 telescopes worldwide.\n"
                       "The in-band chirp lasts over a minute in the raw LIGO data.",
        "params": {
            "gps_time": 1187008882.4, "detector": "H1",
            "m1_solar": 1.46, "m2_solar": 1.27,
            "f_start": 25, "f_end": 800, "chirp_duration": 30.0,
        },
    },
    40: {
        "name": "GW190521", "type": "grav_wave",
        "label": "Massive BH Merger — GW190521",
        "data_source": "LIGO Open Science Center — real L1 strain, GPS 1242442967",
        "description": "The most massive merger detected (21 May 2019). Black holes of 85 and\n"
                       "66 solar masses produced a ~150 solar-mass intermediate-mass black hole.\n"
                       "So massive the chirp is barely visible — just a short, deep thud.",
        "params": {
            "gps_time": 1242442967.4, "detector": "L1",
            "m1_solar": 85, "m2_solar": 66,
            "f_start": 25, "f_end": 80, "chirp_duration": 0.1,
        },
    },

    # ── PULSARS ────────────────────────────────────────────────────────────────
    48: {
        "name": "PSR_Vela", "type": "pulsar",
        "label": "Vela Pulsar — PSR B0833-45",
        "data_source": "EPN database — real 1400 MHz pulse profile, JNAME J0835-4510",
        "description": "One of the brightest pulsars in the sky, 307 pc (1000 ly) away.\n"
                       "Spins 11 times per second. The pulse shape comes from real radio\n"
                       "telescope measurements stored in the EPN database.",
        "params": {
            "jname": "J0835-4510", "bname": "B0833-45",
            "period_s": 0.0893, "duty_cycle": 0.08, "pulse_sharpness": 3.0,
        },
    },
    50: {
        "name": "PSR_LGM1", "type": "pulsar",
        "label": "First Pulsar — PSR B1919+21",
        "data_source": "EPN database — real pulse profile, JNAME J1921+2153",
        "description": "Discovered in 1967 by Jocelyn Bell Burnell — the first pulsar ever.\n"
                       "So regular it was nicknamed 'Little Green Men'. Ticks every 1.3373\n"
                       "seconds, exactly as measured by radio telescopes.",
        "params": {
            "jname": "J1921+2153", "bname": "B1919+21",
            "period_s": 1.3373, "duty_cycle": 0.04, "pulse_sharpness": 5.0,
        },
    },
    52: {
        "name": "PSR_MSP", "type": "pulsar",
        "label": "Millisecond Pulsar — PSR B1937+21",
        "data_source": "EPN database — real pulse profile, JNAME J1939+2134",
        "description": "First millisecond pulsar discovered. Spins 642 times per second —\n"
                       "more stable than an atomic clock. Individual pulses blur into a\n"
                       "continuous buzz at this rotation rate.",
        "params": {
            "jname": "J1939+2134", "bname": "B1937+21",
            "period_s": 0.00155, "duty_cycle": 0.10, "pulse_sharpness": 2.5,
        },
    },
    53: {
        "name": "PSR_Crab", "type": "pulsar",
        "label": "Crab Pulsar — PSR B0531+21",
        "data_source": "EPN database — real pulse profile, JNAME J0534+2200",
        "description": "Born in the supernova of 1054 AD, recorded by Chinese astronomers.\n"
                       "Spins 30 times per second inside the Crab Nebula, 1.99 kpc (6500 ly)\n"
                       "away. Notable double-peaked pulse profile.",
        "params": {
            "jname": "J0534+2200", "bname": "B0531+21",
            "period_s": 0.0334, "duty_cycle": 0.12, "pulse_sharpness": 2.0,
        },
    },

    # ── STARS ──────────────────────────────────────────────────────────────────
    60: {
        "name": "Star_O_type", "type": "star_synth",
        "label": "O-type Star (like Rigel)",
        "data_source": "Blackbody synthesis — published Teff = 30,000 K (spectral classification tables)",
        "description": "The hottest, most massive stars — over 30,000 K. Blue-white.\n"
                       "Live only a few million years before exploding as a supernova.",
        "params": {"temp_k": 30000},
    },
    62: {
        "name": "Star_B_type", "type": "star_synth",
        "label": "B-type Star (like Spica)",
        "data_source": "Blackbody synthesis — published Teff = 22,000 K",
        "description": "Hot blue-white giants. Spica (Alpha Virginis), the brightest star\n"
                       "in Virgo, is a classic example at 22,000 K.",
        "params": {"temp_k": 22000},
    },
    64: {
        "name": "Sirius_A", "type": "star_synth",
        "label": "Sirius — brightest star in the sky",
        "data_source": "Blackbody synthesis — published Teff = 9,940 K",
        "description": "The brightest night-sky star, 2.64 pc (8.6 ly) away. White A-type\n"
                       "star, twice the Sun's mass, 25× more luminous.",
        "params": {"temp_k": 9940},
    },
    65: {
        "name": "Star_F_type", "type": "star_synth",
        "label": "F-type Star (like Procyon)",
        "data_source": "Blackbody synthesis — published Teff = 7,350 K",
        "description": "Yellow-white stars slightly hotter than the Sun. Procyon, 3.37 pc (11 ly)\n"
                       "is one of the nearest F-type stars to Earth.",
        "params": {"temp_k": 7350},
    },
    67: {
        "name": "Sun", "type": "star_helio",
        "label": "The Sun — G-type star",
        "data_source": "GONG helioseismology — real p-mode frequencies, Chaplin et al. 2002",
        "description": "Our Sun resonates in thousands of acoustic modes with periods ~5\n"
                       "minutes. These are measured by the GONG telescope network and sped\n"
                       "up ~133× to bring them into audible range.",
        "params": {"temp_k": 5778},
    },
    69: {
        "name": "Kepler186", "type": "star_kepler",
        "label": "Kepler-186 — hosts first habitable-zone Earth-sized planet",
        "data_source": "NASA Kepler archive — real 4-year photometric light curve",
        "description": "Orange K-type dwarf 178 pc (582 ly) away. Kepler-186f orbits in\n"
                       "its habitable zone. The sound is the star's real brightness variation\n"
                       "recorded by Kepler over four years.",
        "params": {"kepler_id": "Kepler-186", "mission": "Kepler", "temp_k": 4000},
    },
    71: {
        "name": "ProximaCen", "type": "star_kepler",
        "label": "Proxima Centauri — nearest star to the Sun",
        "data_source": "TESS archive — real light curve including stellar flares",
        "description": "The nearest star at 1.30 pc (4.24 ly). A faint red dwarf that flares\n"
                       "violently — those flares appear as sudden brightness spikes in the\n"
                       "TESS light curve, which you can hear as audio transients.",
        "params": {"kepler_id": "Proxima Cen", "mission": "TESS", "temp_k": 3050},
    },

    # ── GALAXIES ───────────────────────────────────────────────────────────────
    72: {
        "name": "MilkyWay_analog", "type": "galaxy_sdss",
        "label": "Milky Way analog — NGC 3198",
        "data_source": "SDSS DR18 spectrum — NGC 3198, RA=154.98, Dec=45.55",
        "description": "A barred spiral galaxy 14.4 Mpc (47 million ly) away, similar to our\n"
                       "Milky Way. Its SDSS spectrum shows a mix of old yellow stars and\n"
                       "young blue star-forming regions in the spiral arms.",
        "params": {"ra": 154.9817, "dec": 45.5497, "galaxy_type": "spiral"},
    },
    74: {
        "name": "M87", "type": "galaxy_sdss",
        "label": "M87 — Giant Elliptical Galaxy",
        "data_source": "SDSS DR18 spectrum — M87/Virgo A, RA=187.706, Dec=12.391",
        "description": "Giant elliptical galaxy 16.2 Mpc (53 million ly) away — famous for\n"
                       "the first ever black hole image (EHT, 2019). Its spectrum is\n"
                       "dominated by old red stars: warm, smooth, bass-heavy sound.",
        "params": {"ra": 187.7059, "dec": 12.3911, "galaxy_type": "elliptical"},
    },
    76: {
        "name": "Arp220", "type": "galaxy_sdss",
        "label": "Arp 220 — Starburst Galaxy",
        "data_source": "SDSS DR18 spectrum — Arp 220, RA=233.738, Dec=23.503",
        "description": "Two galaxies caught merging 76.6 Mpc (250 million ly) away, forming\n"
                       "stars 200× faster than the Milky Way. SDSS spectrum blazes with\n"
                       "emission lines from glowing gas — bright, complex, chaotic sound.",
        "params": {"ra": 233.7380, "dec": 23.5033, "galaxy_type": "starburst"},
    },
    77: {
        "name": "3C273", "type": "galaxy_sdss",
        "label": "3C 273 — Brightest Quasar",
        "data_source": "SDSS DR18 spectrum — 3C 273, RA=187.278, Dec=2.052",
        "description": "A quasar 736 Mpc (2.4 billion ly) away — a galaxy whose supermassive\n"
                       "black hole is actively feeding. Its SDSS spectrum has a flat power-law\n"
                       "continuum with intense broad emission lines on top.",
        "params": {"ra": 187.2779, "dec": 2.0524, "galaxy_type": "agn"},
    },

    # ── SOLAR SYSTEM ───────────────────────────────────────────────────────────
    84: {
        "name": "Jupiter", "type": "planet_radio",
        "label": "Jupiter — Radio Emissions",
        "data_source": "Plasma physics model based on Voyager 1/2 and Juno PDS observations",
        "description": "Jupiter emits intense radio bursts from its powerful magnetosphere\n"
                       "and interactions with the volcanic moon Io. S-bursts and L-bursts\n"
                       "modelled from real spacecraft emission structure.",
        "params": {"planet": "jupiter"},
    },
    86: {
        "name": "Saturn", "type": "planet_radio",
        "label": "Saturn — Magnetosphere Whistlers",
        "data_source": "Model based on Cassini RPWS instrument data (NASA PDS)",
        "description": "Saturn's magnetosphere produces ghostly whistler waves — plasma\n"
                       "waves that disperse as descending tones. Cassini's radio instrument\n"
                       "recorded these throughout its 13 years at Saturn.",
        "params": {"planet": "saturn"},
    },
    88: {
        "name": "EarthChorus", "type": "planet_radio",
        "label": "Earth — Chorus Waves",
        "data_source": "Model based on Van Allen Probes and Cluster satellite observations",
        "description": "Plasma waves in Earth's Van Allen belts, generated by energetic\n"
                       "electrons. They genuinely sound like birdsong — rising chirps at\n"
                       "1–5 kHz — one of the strangest discoveries of the space age.",
        "params": {"planet": "earth"},
    },
}

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def note_name(n): return f"{NOTE_NAMES[n%12]}{n//12-1}"
def sample_path(note):
    return os.path.join(SAMPLES_DIR, f"{OBJECTS[note]['name']}_note{note}.wav") if note in OBJECTS else None

def bandpass(audio, lo, hi, sr=SAMPLE_RATE):
    lo = max(lo, 10); hi = min(hi, sr/2 - 100)
    if lo >= hi: return audio
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio)

def normalise(audio, target=0.80):
    pk = np.max(np.abs(audio))
    return (audio / pk * target).astype(np.float32) if pk > 0 else audio.astype(np.float32)

def safe_audio(audio):
    return np.nan_to_num(np.clip(np.array(audio, dtype=np.float64), -1.0, 1.0),
                         nan=0.0).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
#  GRAVITATIONAL WAVES — real LIGO strain from GWOSC
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gwosc_strain(gps_time, detector, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """
    Download real gravitational wave strain data from the LIGO Open Science Center.
    gwpy fetches HDF5 files directly from gwosc.org — no API key needed.

    Processing pipeline (standard LIGO data release procedure):
      1. Fetch raw strain in a window around the event
      2. Bandpass 20–1800 Hz (detector sensitive band)
      3. Notch 60/120 Hz (US power-line noise)
      4. Whiten by dividing by the noise PSD — this is what makes the chirp audible
      5. Crop to our output window
      6. Resample from 4096 Hz → 44100 Hz
    """
    try:
        from gwpy.timeseries import TimeSeries
        pad = max(duration * 4, 16)
        print(f"    Fetching {detector} strain from GWOSC (GPS {gps_time:.1f})...")
        raw = TimeSeries.fetch_open_data(
            detector, gps_time - pad/2, gps_time + pad/2,
            sample_rate=4096, cache=True, verbose=False
        )
        filtered  = raw.bandpass(20, 1800).notch(60).notch(120)
        whitened  = filtered.whiten(fftlength=4, overlap=2)
        cropped   = whitened.crop(gps_time - duration/2, gps_time + duration/2)
        audio     = np.array(cropped.value, dtype=np.float64)
        n_out     = int(len(audio) * sr / 4096)
        audio     = scipy_resample(audio, n_out)
        audio     = normalise(audio)
        print(f"    ✓ GWOSC strain: {len(audio)/sr:.1f}s at {sr} Hz")
        return audio
    except ImportError:
        print("    ⚠ gwpy not installed — pip install gwpy")
        return None
    except Exception as e:
        print(f"    ⚠ GWOSC download failed: {e}")
        return None


def analytic_chirp(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """
    Fallback: post-Newtonian inspiral chirp computed from the real published
    component masses. Gives the correct frequency evolution even without gwpy.
    """
    G, C, M_SUN = 6.674e-11, 3.0e8, 1.989e30
    m1 = params["m1_solar"] * M_SUN
    m2 = params["m2_solar"] * M_SUN
    m_chirp = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)
    f_start, f_end = params["f_start"], params["f_end"]
    chirp_dur = min(params["chirp_duration"], duration * 0.8)

    n_total   = int(duration * sr)
    n_chirp   = int(chirp_dur * sr)
    audio     = np.zeros(n_total)
    t         = np.linspace(0, chirp_dur, n_chirp, endpoint=False)
    tau       = np.maximum(chirp_dur - t, 1e-6)
    f_inst    = np.clip(
        (5/256)**(3/8) / np.pi * (G*m_chirp/C**3)**(-5/8) * tau**(-3/8),
        f_start, f_end
    )
    phase     = 2 * np.pi * np.cumsum(f_inst) / sr
    amp       = (tau / chirp_dur) ** (-1/4)
    amp      /= amp.max()
    fade      = min(int(0.05*sr), n_chirp)
    amp[:fade] *= np.linspace(0, 1, fade)

    offset  = int(0.1 * sr)
    end_idx = min(offset + n_chirp, n_total)
    audio[offset:end_idx] = np.cos(phase) * amp[:end_idx - offset]

    ring_start = offset + n_chirp
    if ring_start < n_total:
        n_r = n_total - ring_start
        t_r = np.arange(n_r) / sr
        audio[ring_start:] += np.sin(2*np.pi*f_end*t_r) * np.exp(-t_r/0.15) * 0.3

    audio = bandpass(audio, max(20, f_start*0.8), min(f_end*1.2, 2000))
    return normalise(audio)


def gen_grav_wave(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    audio = fetch_gwosc_strain(params["gps_time"], params["detector"], duration, sr)
    if audio is None:
        print("    Falling back to analytic PN chirp (real masses)")
        audio = analytic_chirp(params, duration, sr)
    return safe_audio(audio)


# ═══════════════════════════════════════════════════════════════════════════════
#  PULSARS — real pulse profiles from the EPN database
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_epn_profile(jname, bname=None):
    """
    Fetch a real pulse profile from the European Pulsar Network (EPN) database.
    https://www.epta.eu.org/epndb/

    The EPN stores radio telescope measurements of how pulse intensity varies
    across one full rotation period. We take the profile closest to 1400 MHz
    (L-band, the most common observing frequency).

    Returns a normalised numpy array or None on failure.
    """
    try:
        import requests
    except ImportError:
        print("    ⚠ requests not installed — pip install requests")
        return None

    names = [jname] + ([bname] if bname else [])
    for name in names:
        try:
            url = f"https://www.epta.eu.org/epndb/json?name={name}"
            print(f"    Fetching EPN profile for {name}...")
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                continue

            # Pick entry closest to 1400 MHz
            best, best_diff = None, float("inf")
            for entry in data:
                try:
                    diff = abs(float(entry.get("freq", 0)) - 1400)
                    if diff < best_diff:
                        best_diff = diff; best = entry
                except (ValueError, TypeError):
                    continue
            if best is None:
                best = data[0]

            profile_data = best.get("data", [])
            if not profile_data:
                continue

            profile = np.array([[float(r[0]), float(r[1])]
                                 for r in profile_data if len(r) >= 2])
            intensities = profile[:, 1]
            intensities -= intensities.min()
            peak = intensities.max()
            if peak > 0:
                intensities /= peak
            print(f"    ✓ EPN: {len(intensities)} phase bins at {best.get('freq','?')} MHz")
            return intensities

        except Exception as e:
            print(f"    ⚠ EPN failed for {name}: {e}")
    return None


def synth_pulse_profile(n_bins, duty_cycle, sharpness):
    x = np.linspace(0, 1, n_bins, endpoint=False)
    profile  = np.exp(-0.5 * ((x - 0.05) / (duty_cycle/2))**2) ** sharpness
    profile += np.exp(-0.5 * ((x - 0.55) / duty_cycle)**2) ** sharpness * 0.3
    profile -= profile.min()
    return profile / profile.max()


def gen_pulsar(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """
    Tile a real EPN pulse profile at the real rotation period to create audio.
    The rhythm IS the real spin rate. The shape IS the real measured profile.
    """
    period_s      = params["period_s"]
    period_samples = max(2, int(period_s * sr))
    n_total        = int(duration * sr)

    profile_raw = fetch_epn_profile(params.get("jname"), params.get("bname"))

    if profile_raw is not None and len(profile_raw) > 4:
        profile = scipy_resample(profile_raw, period_samples)
        profile -= profile.min()
        if profile.max() > 0: profile /= profile.max()
    else:
        print("    Using synthesised pulse profile (real period)")
        profile = synth_pulse_profile(
            period_samples, params["duty_cycle"], params["pulse_sharpness"]
        )

    carrier_freq = (
        80   if period_s > 0.1  else
        300  if period_s > 0.01 else
        1200 if period_s > 0.001 else 3000
    )

    tiled    = np.tile(profile, int(np.ceil(n_total/period_samples)) + 1)[:n_total]
    t        = np.arange(n_total) / sr
    harmonics = sum(w * np.sin(2*np.pi*carrier_freq*i*t)
                    for i, w in enumerate([1, .5, .25, .1], 1)) / 1.85
    audio    = harmonics * tiled
    if period_s > 0.05:
        audio += np.random.randn(n_total) * 0.15 * tiled

    sos = butter(4, min(carrier_freq*6, sr*0.45), btype="low", fs=sr, output="sos")
    audio = sosfilt(sos, audio)
    fade  = int(0.05 * sr)
    audio[:fade]  *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return safe_audio(normalise(audio))


# ═══════════════════════════════════════════════════════════════════════════════
#  STARS — real Kepler/TESS light curves + helioseismology
# ═══════════════════════════════════════════════════════════════════════════════

def planck_spectrum(temp_k, n=32):
    wl_m = np.linspace(300, 900, n) * 1e-9
    h, c, k = 6.626e-34, 3e8, 1.381e-23
    exp  = np.clip((h*c) / (wl_m*k*temp_k), 0, 500)
    power = (2*h*c**2) / (wl_m**5 * (np.exp(exp) - 1))
    return power / power.max()

def star_carrier(temp_k, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """Additive synthesis weighted by blackbody spectrum — the star's intrinsic voice."""
    power = planck_spectrum(temp_k)
    f0    = np.clip(80 * (temp_k/3000)**0.5, 60, 1200)
    n     = int(duration * sr)
    t     = np.arange(n) / sr
    audio = np.zeros(n)
    for i, amp in enumerate(power, 1):
        freq = f0 * i * (1 + 0.002 * i**1.5)
        if freq >= sr/2 - 100: break
        audio += amp * np.sin(2*np.pi*freq*t + np.random.uniform(0, 2*np.pi))
    sos     = butter(2, 10, btype="low", fs=sr, output="sos")
    flicker = sosfilt(sos, np.random.randn(n))
    flicker /= (np.max(np.abs(flicker)) + 1e-10)
    return audio * (1 + 0.03 * flicker)

def gen_star_synth(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    return safe_audio(normalise(star_carrier(params["temp_k"], duration, sr)))

def gen_star_helio(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """
    Sun: real GONG p-mode oscillation frequencies from Chaplin et al. 2002.
    Sped up ~133× to bring 5-minute oscillations (~3000 μHz) to ~400 Hz.
    """
    n  = int(duration * sr)
    t  = np.arange(n) / sr
    # (frequency_μHz, relative_amplitude) — from Table 1 of Chaplin et al. 2002
    pmodes = [
        (2093,.30),(2231,.50),(2388,.70),(2559,.90),(2735,1.0),(2912,.95),
        (3094,.85),(3277,.70),(3462,.55),(3650,.40),(3840,.28),(4032,.18),
        (2185,.25),(2335,.45),(2480,.65),(2655,.85),(2829,.90),(3008,.80),
    ]
    speedup = 400.0 / 3000.0 * 1e6
    audio   = np.zeros(n)
    for freq_uHz, amp in pmodes:
        f_audio = freq_uHz * 1e-6 * speedup
        if f_audio >= sr / 2: continue
        mod    = 0.8 + 0.2 * np.sin(2*np.pi*np.random.uniform(0.1, 0.5)*t)
        audio += amp * mod * np.sin(2*np.pi*f_audio*t + np.random.uniform(0, 2*np.pi))
    sos   = butter(4, [30, 500], btype="band", fs=sr, output="sos")
    audio += sosfilt(sos, np.random.randn(n)) * 0.08
    return safe_audio(normalise(audio))

def gen_star_kepler(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """
    Download a real Kepler/TESS light curve and use it to amplitude-modulate
    a blackbody carrier tone. The star's real brightness variations become
    the rhythm and texture of the sound.
    """
    try:
        import lightkurve as lk
        kid, mission = params["kepler_id"], params.get("mission", "Kepler")
        print(f"    Downloading light curve: {kid} ({mission})...")
        search = lk.search_lightcurve(kid, mission=mission)
        if len(search) == 0:
            raise ValueError("No light curves found")
        lc   = search[0].download().remove_nans().remove_outliers(sigma=5)
        flux = np.nan_to_num(lc.flux.value, nan=1.0)
        flux = np.clip(flux / np.median(flux), 0.5, 1.5)
        n    = int(duration * sr)
        flux_r = scipy_resample(flux, n)
        sos    = butter(2, 50, btype="low", fs=sr, output="sos")
        flux_smooth = np.clip(sosfilt(sos, flux_r), 0.3, 2.0)
        audio  = star_carrier(params.get("temp_k", 5000), duration, sr) * flux_smooth
        print(f"    ✓ Light curve: {len(flux)} flux points → audio")
    except ImportError:
        print("    ⚠ lightkurve not installed — pip install lightkurve")
        audio = star_carrier(params.get("temp_k", 5000), duration, sr)
    except Exception as e:
        print(f"    ⚠ Kepler download failed ({e}) — using synthesis")
        audio = star_carrier(params.get("temp_k", 5000), duration, sr)
    return safe_audio(normalise(np.array(audio)))


# ═══════════════════════════════════════════════════════════════════════════════
#  GALAXIES — real SDSS DR18 optical spectra
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_sdss_spectrum(ra, dec):
    """
    Fetch a real optical spectrum from SDSS DR18 by sky position.

    SDSS spectra cover ~3800–9200 Å at resolving power R~2000.
    Each pixel is a (wavelength, flux) pair — we treat flux(λ) as the
    Fourier magnitude at the corresponding audio frequency, so the
    galaxy's colour literally becomes its sonic timbre:
      - Red, old stars (ellipticals) → bass-heavy, warm
      - Blue, young stars (spirals/starbursts) → bright, treble-rich
      - Emission lines → distinct tonal peaks

    Returns (wavelengths, flux) arrays or None on failure.
    """
    try:
        from astroquery.sdss import SDSS
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        print(f"    Querying SDSS DR18 at RA={ra:.3f} Dec={dec:.3f}...")

        xid = SDSS.query_region(pos, radius=10*u.arcsec, spectro=True, data_release=18)
        if xid is None or len(xid) == 0:
            xid = SDSS.query_region(pos, radius=30*u.arcsec, spectro=True, data_release=18)
        if xid is None or len(xid) == 0:
            raise ValueError("No SDSS spectrum at this position within 30 arcsec")

        print(f"    Found match — downloading spectrum...")
        sp = SDSS.get_spectra(matches=xid[:1], data_release=18)
        if not sp:
            raise ValueError("Empty spectrum download")

        hdu  = sp[0]
        flux = np.array(hdu[1].data["flux"],   dtype=np.float64)
        loglam = np.array(hdu[1].data["loglam"], dtype=np.float64)
        wl   = 10 ** loglam  # log-wavelength → Å

        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        flux = np.maximum(flux, 0)
        print(f"    ✓ SDSS spectrum: {len(flux)} pixels, {wl.min():.0f}–{wl.max():.0f} Å")
        return wl, flux

    except ImportError:
        print("    ⚠ astroquery not installed — pip install astroquery astropy")
        return None
    except Exception as e:
        print(f"    ⚠ SDSS failed: {e}")
        return None


def sdss_to_audio(wl, flux, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    """
    Inverse-FFT sonification of an SDSS spectrum.

    Mapping: shorter wavelength (blue) → higher audio frequency
             longer wavelength (red)  → lower audio frequency
    The flux at each wavelength sets the amplitude of that audio frequency.
    Random phases produce a stationary noise-like texture whose colour
    matches the galaxy's spectral energy distribution.
    """
    n       = int(duration * sr)
    n_pos   = n // 2 + 1

    # Normalised wavelength axis: 0=red(low freq), 1=blue(high freq)
    log_wl       = np.log10(wl)
    log_wl_norm  = 1 - (log_wl - log_wl.min()) / (log_wl.max() - log_wl.min() + 1e-10)
    flux_norm    = flux / (flux.max() + 1e-10)

    # Audio frequency axis (log-spaced for perceptual uniformity)
    audio_freqs  = np.fft.rfftfreq(n, d=1/sr)
    f_min, f_max = 80, min(8000, sr/2 - 100)
    af_safe      = np.clip(audio_freqs, f_min, f_max)
    log_f_norm   = (np.log10(af_safe) - np.log10(f_min)) / (np.log10(f_max) - np.log10(f_min))
    log_f_norm[0] = 0

    # Interpolate spectrum flux onto the audio frequency axis
    magnitudes   = np.interp(log_f_norm, log_wl_norm, flux_norm)
    magnitudes[0] = 0  # zero DC

    phases   = np.random.uniform(0, 2*np.pi, n_pos)
    spectrum = magnitudes * np.exp(1j * phases)
    full     = np.zeros(n, dtype=complex)
    full[:n_pos] = spectrum
    full[n_pos:] = np.conj(spectrum[1:n-n_pos+1][::-1])
    audio    = np.real(np.fft.ifft(full))

    fade = int(0.1 * sr)
    audio[:fade]  *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return audio


def sed_template_fallback(galaxy_type, n):
    """Template SED used when SDSS download fails."""
    x = np.linspace(0, 1, n)
    if galaxy_type == "elliptical":
        wl_norm = np.linspace(100, 10000, n) / 800
        sed = wl_norm**(-3) / (np.exp(1.0/wl_norm) - 1e-6)
    elif galaxy_type == "spiral":
        wl_m = np.linspace(300, 900, n) * 1e-9
        h, c, k = 6.626e-34, 3e8, 1.381e-23
        exp = np.clip((h*c)/(wl_m*k*4000), 0, 500)
        sed = (2*h*c**2)/(wl_m**5*(np.exp(exp)-1)) * 0.6 + x**1.5 * 0.3
        for pos in [0.3, 0.5, 0.65, 0.8]:
            idx = int(pos*n); sed[max(0,idx-2):idx+3] += 0.15
    elif galaxy_type == "starburst":
        sed = np.exp(-0.5*((x-0.8)/0.15)**2)*0.9 + np.exp(-0.5*((x-0.1)/0.2)**2)*0.7
        for pos in [0.2, 0.35, 0.55, 0.7, 0.85]:
            idx = int(pos*n); sed[max(0,idx-1):idx+2] += 0.25
    else:  # agn
        sed = np.linspace(0.01, 1, n)**(-0.3); sed /= sed.max()
        for pos, s in [(0.15,.8),(0.3,.6),(0.45,.9),(0.6,.5),(0.75,.7)]:
            idx = int(pos*n); sed[max(0,idx-1):idx+2] += s
    sed = np.maximum(sed, 0)
    return sed / (sed.max() + 1e-10)


def gen_galaxy_sdss(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    result = fetch_sdss_spectrum(params["ra"], params["dec"])
    n = int(duration * sr)

    if result is not None:
        wl, flux = result
        audio = sdss_to_audio(wl, flux, duration, sr)
    else:
        print(f"    Using SED template for {params['galaxy_type']}")
        sed   = sed_template_fallback(params["galaxy_type"], 256)
        n_pos = n // 2 + 1
        mags  = np.interp(np.linspace(0,1,n_pos), np.linspace(0,1,256), sed)
        phases = np.random.uniform(0, 2*np.pi, n_pos)
        spec  = mags * np.exp(1j * phases)
        full  = np.zeros(n, dtype=complex)
        full[:n_pos] = spec
        full[n_pos:] = np.conj(spec[1:n-n_pos+1][::-1])
        audio = np.real(np.fft.ifft(full))
        fade  = int(0.1 * sr)
        audio[:fade]  *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)

    audio = bandpass(audio, 80, 8000)
    return safe_audio(normalise(audio))


# ═══════════════════════════════════════════════════════════════════════════════
#  PLANETS — plasma physics synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def gen_planet_radio(params, duration=SAMPLE_DURATION, sr=SAMPLE_RATE):
    n = int(duration * sr)
    t = np.arange(n) / sr
    planet = params.get("planet", "jupiter")
    audio  = np.zeros(n)

    if planet == "jupiter":
        audio += np.sin(2*np.pi*80*t) * 0.3
        for bt in np.arange(0, duration, 1/2.5):
            idx = int((bt + np.random.uniform(-0.1,0.1)) * sr)
            bl  = int(0.08 * sr)
            if idx + bl < n:
                audio[idx:idx+bl] += (np.random.randn(bl)
                    * np.exp(-np.arange(bl)/(bl*0.3))
                    * np.random.uniform(0.4, 1.0))
        for _ in range(int(8 * duration)):
            idx = np.random.randint(0, max(1, n-200))
            sl  = int(0.003 * sr)
            if idx + sl < n:
                st = np.arange(sl) / sr
                audio[idx:idx+sl] += (
                    np.sin(2*np.pi*np.random.uniform(300,1200)*st)
                    * np.exp(-np.arange(sl)/(sl*0.2))
                    * np.random.uniform(0.3, 0.8))
        sos = butter(4, [40, 4000], btype="band", fs=sr, output="sos")

    elif planet == "saturn":
        for _ in range(int(duration / 0.8)):
            s   = np.random.uniform(0, duration - 0.6)
            dw  = np.random.uniform(0.3, 0.7)
            f0, f1 = np.random.uniform(40,100), np.random.uniform(200,600)
            nw  = int(dw * sr); idx = int(s * sr)
            if idx + nw < n:
                tw = np.linspace(0, dw, nw)
                audio[idx:idx+nw] += (
                    np.sin(2*np.pi*(f0*tw + (f1-f0)/(2*dw)*tw**2))
                    * np.hanning(nw)
                    * np.random.uniform(0.3, 0.8))
        audio += sosfilt(
            butter(4,[30,500],btype="band",fs=sr,output="sos"),
            np.random.randn(n) * 0.1)
        audio *= 0.7 + 0.3 * np.sin(2*np.pi*0.4*t)
        sos = butter(2, [20, 3000], btype="band", fs=sr, output="sos")

    else:  # earth chorus
        for _ in range(int(4 * duration)):
            s  = np.random.uniform(0, duration - 0.3)
            de = np.random.uniform(0.05, 0.25)
            f0 = np.random.uniform(500, 1500)
            f1 = min(f0 * np.random.uniform(1.5, 4.0), 5000)
            ne = int(de * sr); idx = int(s * sr)
            if idx + ne < n:
                te     = np.linspace(0, de, ne)
                freq_t = f0 + (f1 - f0) * (te/de) ** 1.5
                audio[idx:idx+ne] += (
                    np.sin(2*np.pi * np.cumsum(freq_t)/sr)
                    * np.hanning(ne) ** 0.5
                    * np.random.uniform(0.3, 1.0))
        audio += sosfilt(
            butter(4,[200,6000],btype="band",fs=sr,output="sos"),
            np.random.randn(n) * 0.05)
        sos = butter(2, [100, 8000], btype="band", fs=sr, output="sos")

    audio = sosfilt(sos, audio)
    return safe_audio(normalise(audio))


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "grav_wave":   gen_grav_wave,
    "pulsar":      gen_pulsar,
    "star_synth":  gen_star_synth,
    "star_helio":  gen_star_helio,
    "star_kepler": gen_star_kepler,
    "galaxy_sdss": gen_galaxy_sdss,
    "planet_radio":gen_planet_radio,
}

def generate_all(duration=SAMPLE_DURATION):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print(f"\n🔭  Generating {len(OBJECTS)} samples  →  {SAMPLES_DIR}/\n")
    ok, fail = 0, []
    for note in sorted(OBJECTS):
        obj = OBJECTS[note]
        print(f"  [{note:3d}]  {obj['label']}")
        print(f"         {obj['data_source']}")
        try:
            audio  = GENERATORS[obj["type"]](obj["params"], duration)
            audio  = safe_audio(audio)
            stereo = np.ascontiguousarray(np.column_stack([audio, audio]))
            sf.write(sample_path(note), stereo, SAMPLE_RATE, subtype="PCM_16")
            print(f"         ✓ saved\n")
            ok += 1
        except Exception as e:
            import traceback
            print(f"         ✗ FAILED: {e}")
            traceback.print_exc()
            fail.append((note, obj["name"]))
            print()
    print(f"{'─'*60}")
    print(f"  ✓ {ok} samples generated,  ✗ {len(fail)} failed")
    if fail:
        for n, nm in fail: print(f"    note {n}: {nm}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MIDI / KEYBOARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_library(master_vol=0.85):
    lib, missing = {}, []
    for note in OBJECTS:
        fp = sample_path(note)
        if fp and os.path.exists(fp):
            try:
                audio, _ = sf.read(fp, dtype="float32")
                if audio.ndim == 1:
                    audio = np.column_stack([audio, audio])
                lib[note] = (audio * master_vol).astype(np.float32)
            except Exception:
                missing.append(note)
        else:
            missing.append(note)
    if missing:
        names = [OBJECTS[n]["name"] for n in missing]
        print(f"  ⚠  Missing: {', '.join(names[:6])}"
              + ("..." if len(names) > 6 else ""))
        print("  →  Run with --generate first\n")
    return lib


def _play_note(lib, note, vel, active):
    import pygame
    audio = lib.get(note)
    if audio is None:
        print(f"  (note {note} — no sample, run --generate)"); return
    int16 = np.ascontiguousarray((audio * (vel/127.0) * 32767).astype(np.int16))
    sound = pygame.sndarray.make_sound(int16)
    ch    = pygame.mixer.find_channel(force=True)
    ch.play(sound, loops=-1)
    active[note] = ch
    obj = OBJECTS.get(note, {})
    print(f"\n  ♪  {note_name(note)}  —  {obj.get('label','')}")
    print(f"     Source: {obj.get('data_source','')}")
    desc_line1 = obj.get("description","").split("\n")[0].strip()
    print(f"     {desc_line1}\n")


def play_midi(device_name=None, master_vol=0.85):
    import pygame
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init(); pygame.mixer.set_num_channels(32); pygame.init()
    print("\n🔭  SOUNDS OF SPACE — loading samples...")
    lib = load_library(master_vol)
    print(f"  ✓  {len(lib)}/{len(OBJECTS)} samples ready\n")
    active = {}; sustain = False; sustained = set()

    try:
        import mido
    except ImportError:
        print("  ✗  pip install mido python-rtmidi"); return

    ports = mido.get_input_names()
    if not ports:
        print("  ✗  No MIDI devices. Use --keyboard instead."); return
    port = next((p for p in ports if device_name and device_name.lower() in p.lower()), ports[0])
    print(f"  ✓  MIDI: {port}\n  Ctrl+C to stop\n")

    with mido.open_input(port) as p:
        for msg in p:
            if msg.type == "note_on":
                if msg.velocity == 0:
                    #if not sustain: active.pop(msg.note, None) and active.get(msg.note) and active[msg.note].stop()
                    #else: sustained.add(msg.note)
                    if not sustain:
                        ch = active.pop(msg.note, None)
                        if ch: ch.stop()
                    else: sustained.add(msg.note)
                else:
                    _play_note(lib, msg.note, msg.velocity, active)
                    if sustain: sustained.add(msg.note)
            elif msg.type == "note_off":
                if sustain: sustained.add(msg.note)
                else:
                    ch = active.pop(msg.note, None)
                    if ch: ch.stop()
            elif msg.type == "control_change" and msg.control == 64:
                sustain = msg.value >= 64
                if not sustain:
                    for n in list(sustained):
                        ch = active.pop(n, None)
                        if ch: ch.fadeout(200)
                    sustained.clear()


def play_keyboard(master_vol=0.85):
    import pygame
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init(); pygame.mixer.set_num_channels(32); pygame.init()
    lib = load_library(master_vol)

    KEY_MAP = {
        pygame.K_z:36, pygame.K_x:38, pygame.K_c:40,
        pygame.K_v:48, pygame.K_b:50, pygame.K_n:52, pygame.K_m:53,
        pygame.K_a:60, pygame.K_s:62, pygame.K_d:64, pygame.K_f:65,
        pygame.K_g:67, pygame.K_h:69, pygame.K_j:71,
        pygame.K_q:72, pygame.K_w:74, pygame.K_e:76, pygame.K_r:77,
        pygame.K_t:84, pygame.K_y:86, pygame.K_u:88,
    }

    screen = pygame.display.set_mode((720, 240))
    pygame.display.set_caption("🔭 Sounds of Space")
    font   = pygame.font.SysFont("monospace", 13)
    active = {}

    print("\n🔭  Keyboard Mode")
    print("  Z/X/C=Black holes | V/B/N/M=Pulsars | A-J=Stars | Q-R=Galaxies | T/Y/U=Planets\n")

    running = True
    while running:
        screen.fill((5, 5, 20))
        for i, (txt, col) in enumerate([
            ("🔭  SOUNDS OF SPACE", (150,200,255)),
            ("Z/X/C = Black holes  |  V/B/N/M = Pulsars  |  A/S/D/F/G/H/J = Stars", (120,160,200)),
            ("Q/W/E/R = Galaxies   |  T/Y/U = Solar system   |  ESC to quit", (100,130,170)),
        ]):
            screen.blit(font.render(txt, True, col), (20, 20 + i*40))

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key in KEY_MAP and event.key not in active:
                note  = KEY_MAP[event.key]
                audio = lib.get(note)
                if audio is not None:
                    sound = pygame.sndarray.make_sound(
                        np.ascontiguousarray((audio * 32767).astype(np.int16)))
                    ch = pygame.mixer.find_channel(force=True)
                    ch.play(sound, loops=-1); active[event.key] = ch
                    obj = OBJECTS.get(note, {})
                    print(f"  ♪  {note_name(note)}  {obj.get('label','')}")
                    print(f"     {obj.get('data_source','')}\n")
            elif event.type == pygame.KEYUP and event.key in active:
                active.pop(event.key).stop()

        pygame.display.flip()
        pygame.time.wait(10)
    pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Sounds of Space — Piano")
    p.add_argument("--generate",  action="store_true", help="Download data and generate all WAV samples")
    p.add_argument("--play",      action="store_true", help="Start MIDI engine")
    p.add_argument("--keyboard",  action="store_true", help="Computer keyboard mode (no MIDI needed)")
    p.add_argument("--list",      action="store_true", help="List all objects and data sources")
    p.add_argument("--preview",   type=int,   default=None, help="Preview MIDI note number")
    p.add_argument("--midi",      type=str,   default=None, help="MIDI device name (partial match)")
    p.add_argument("--volume",    type=float, default=0.85,  help="Master volume 0–1")
    p.add_argument("--duration",  type=float, default=SAMPLE_DURATION, help="Sample duration in seconds")
    args = p.parse_args()

    if args.list:
        print(f"\n{'Note':>5}  {'Label':<44}  Data source")
        print("─" * 115)
        for note in sorted(OBJECTS):
            o = OBJECTS[note]
            print(f"{note:>5}  {o['label']:<44}  {o['data_source']}")
        print()
        return

    if args.generate:
        generate_all(args.duration)
        return

    if args.preview is not None:
        fp = sample_path(args.preview)
        if not fp or not os.path.exists(fp):
            print(f"Note {args.preview} not found — run --generate first")
            return
        try:
            import sounddevice as sd
            audio, sr = sf.read(fp)
            print(f"Playing {note_name(args.preview)} — {OBJECTS.get(args.preview,{}).get('label','')}")
            sd.play(audio, sr); sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")
        return

    if args.keyboard:
        play_keyboard(args.volume)
        return

    if args.play:
        play_midi(args.midi, args.volume)
        return

    p.print_help()
    print("\nQuick start:")
    print("  python space_piano_standalone.py --generate")
    print("  python space_piano_standalone.py --keyboard")


if __name__ == "__main__":
    main()
