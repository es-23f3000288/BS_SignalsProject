# =================================================================
# Soumendu Ray | 23F3000288 | Signals Project | BS ES May 2026 Tem
# =================================================================

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import streamlit as st

# ==========================================
# 1. CORE MATHEMATICAL LOGIC FUNCTIONS
# ==========================================

def calculate_resampling_factors(fs_in, fs_out):
    """Calculates rational fraction integers P, Q and Nyquist metrics."""
    gcd = np.gcd(fs_in, fs_out)
    P = fs_out // gcd
    Q = fs_in // gcd
    return P, Q, fs_in / 2.0, fs_out / 2.0


def convert_int16_to_float64(audio_data):
    """Symmetrically normalizes signed int16 audio to float64 (-1.0 to 1.0)."""
    max_val = float(abs(np.iinfo(np.int16).min))
    return audio_data.astype(np.float64) / max_val


def convert_float64_to_int16(audio_float):
    """Symmetrically scales and rounds float64 back to signed int16 arrays."""
    max_val = float(abs(np.iinfo(np.int16).min))
    clipped = np.clip(audio_float, -1.0, 1.0)
    return np.round(clipped * max_val).astype(np.int16)


def execute_resampling(audio_float, P, Q):
    """Performs polyphase resampling across the input array."""
    return signal.resample_poly(audio_float, P, Q, axis=0)


def evaluate_bandwidth_distortion(audio_float, fs_in, target_nyquist):
    """Quantifies high-frequency energy suppression downsampling penalties (Full Track)."""
    fft_orig = np.fft.rfft(audio_float)
    freqs_orig = np.fft.rfftfreq(len(audio_float), d=1/fs_in)
    energy_orig = np.abs(fft_orig) ** 2

    lost_mask = freqs_orig > target_nyquist
    total_energy = np.sum(energy_orig)
    lost_energy = np.sum(energy_orig[lost_mask]) if total_energy > 0 else 0
    return (lost_energy / total_energy) * 100


def evaluate_window_bandwidth_distortion(audio_float, fs_in, target_nyquist, window_sec=0.02):
    """Quantifies bandwidth suppression inside a strict window to capture spectral leakage."""
    samples = int(fs_in * window_sec)
    windowed_data = audio_float[:samples]
    
    fft_win = np.fft.rfft(windowed_data)
    freqs_win = np.fft.rfftfreq(len(windowed_data), d=1/fs_in)
    energy_win = np.abs(fft_win) ** 2
    
    lost_mask = freqs_win > target_nyquist
    total_energy = np.sum(energy_win)
    lost_energy = np.sum(energy_win[lost_mask]) if total_energy > 0 else 0
    return (lost_energy / total_energy) * 100


# ==========================================
# 2. SIGNAL GENERATION & VISUALIZATION
# ==========================================

def generate_test_tone(fs_in, duration=2.0):
    """Generates 2 seconds of a synthetic audio mixture."""
    t_in = np.linspace(0, duration, int(fs_in * duration), endpoint=False)
    tone_440 = 0.6 * np.sin(2 * np.pi * 440.0 * t_in)
    tone_6000 = 0.3 * np.sin(2 * np.pi * 6000.0 * t_in)
    return tone_440 + tone_6000


def build_analysis_plots(audio_float, resampled_float, fs_in, fs_out, target_nyquist, plot_window_sec=0.02):
    """Constructs time and frequency domain subplots matching a short window size."""
    samples_orig = int(fs_in * plot_window_sec)
    samples_resamp = int(fs_out * plot_window_sec)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time-Domain Window Construction
    t_in_ms = np.linspace(0, plot_window_sec, samples_orig, endpoint=False) * 1000
    t_out_ms = np.linspace(0, plot_window_sec, samples_resamp, endpoint=False) * 1000

    ax1.plot(t_in_ms, audio_float[:samples_orig], label=f"Original ({fs_in} Hz)", color="#38bdf8", linewidth=2)
    ax1.stem(t_out_ms, resampled_float[:samples_resamp], linefmt="#f97316", markerfmt="or", label=f"Resampled ({fs_out} Hz)", basefmt=" ")
    ax1.set_xlim(0, 15)
    ax1.set_title("Time-Domain Waveform Signature Comparisons", fontweight='bold')
    ax1.set_xlabel("Time (milliseconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Frequency-Domain Spectrum Construction (Matching the 20ms view slice)
    fft_orig_p = np.fft.rfft(audio_float[:samples_orig])
    freqs_orig_p = np.fft.rfftfreq(samples_orig, d=1/fs_in)
    fft_resamp_p = np.fft.rfft(resampled_float[:samples_resamp])
    freqs_resamp_p = np.fft.rfftfreq(samples_resamp, d=1/fs_out)

    ax2.plot(freqs_orig_p, np.abs(fft_orig_p) / samples_orig, label="Original Spectrum", color="#38bdf8", linewidth=2)
    ax2.plot(freqs_resamp_p, np.abs(fft_resamp_p) / samples_resamp, label="Resampled Spectrum", color="#f97316", linewidth=2, linestyle="--")
    ax2.axvline(x=target_nyquist, color="#ef4444", linestyle=":", linewidth=2, label=f"Target Nyquist ({target_nyquist} Hz)")
    ax2.set_xlim(0, 10000)
    ax2.set_title("Frequency Spectrum Distortion Analysis", fontweight='bold')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Normalized Magnitude")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.subplots_adjust(hspace=0.4)
    return fig


# ==========================================
# 3. STREAMLIT APPLICATION INTERFACE
# ==========================================

def render_streamlit_app():
    """Renders the Streamlit frontend layout, sliders, and audio modules."""
    st.set_page_config(page_title="Audio Sampling Rate conversion", layout="wide")

    st.markdown(
        "<h1 style='text-align: center; color: #1e3a8a; font-weight: bold;'>BS Signals Project: 23F3000288</h1>", 
        unsafe_allow_html=True
    )
    
    st.title("🎛️ Audio Sampling Rate Conversion")
    st.write("Input sample rates, listen to the audio outputs, and evaluate energy distortion.")

    # Sidebar Controls
    st.sidebar.header("Signal Configuration")
    fs_in = st.sidebar.slider("Original Sampling Rate (Hz)", min_value=8000, max_value=48000, value=44100, step=1000)
    fs_out = st.sidebar.slider("Target Sampling Rate (Hz)", min_value=4000, max_value=48000, value=11256, step=1000)

    # Core Execution Flow
    tone_mix = generate_test_tone(fs_in)
    audio_data_int16 = convert_float64_to_int16(tone_mix)
    audio_float = convert_int16_to_float64(audio_data_int16)

    P, Q, orig_nyquist, target_nyquist = calculate_resampling_factors(fs_in, fs_out)
    resampled_float = execute_resampling(audio_float, P, Q)
    resampled_int16 = convert_float64_to_int16(resampled_float)

    # Render Metric Dashboards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rational Ratio (P/Q)", f"{P} / {Q}")
    with col2:
        st.metric("Target Nyquist Limit", f"{target_nyquist} Hz")
    with col3:
        if fs_out < fs_in:
            distortion_full = evaluate_bandwidth_distortion(audio_float, fs_in, target_nyquist)
            st.metric("Full-Track Distortion Loss", f"{distortion_full:.2f}%")
        else:
            st.metric("Full-Track Distortion Loss", "0.00%", delta="Clean Track")
    with col4:
        if fs_out < fs_in:
            # New Metric tracking the observed window footprint directly
            distortion_win = evaluate_window_bandwidth_distortion(audio_float, fs_in, target_nyquist, window_sec=0.02)
            st.metric("Observed 20ms Distortion Loss", f"{distortion_win:.2f}%", delta="-Leakage Active", delta_color="inverse")
        else:
            st.metric("Observed 20ms Distortion Loss", "0.00%", delta="Clean Track")

    # Render Playback Components
    st.subheader("🔊 Audio Footprint Verification")
    aud1, aud2 = st.columns(2)
    with aud1:
        st.write(f"Original Audio Mixed Tone ({fs_in} Hz):")
        st.audio(audio_data_int16, sample_rate=fs_in)
    with aud2:
        st.write(f"Resampled Output Array ({fs_out} Hz):")
        st.audio(resampled_int16, sample_rate=fs_out)

    # Render Signal Domains
    st.subheader("📊 Signal Analysis")
    fig = build_analysis_plots(audio_float, resampled_float, fs_in, fs_out, target_nyquist)
    st.pyplot(fig)


# --- ENTRY POINT CONTROL ---
if __name__ == "__main__":
    render_streamlit_app()
