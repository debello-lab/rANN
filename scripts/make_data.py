"""
make_data.py
============
Constructs the synthetic dataset of binaural features and source position labels, and provides utilities for visualizing the data.

Usage
─────
1. To create the synthetic dataset, call `create_synthetic_dataset()` with desired parameters. This returns the feature matrix `x_train`, target labels `y_train`, and the axes for frequency, ITD, and ILD.
2. To visualize the features and target distributions for a filtered subset of the data, use `plot_filtered_profiles()` with the dataset and desired azimuth/elevation ranges.

"""

import numpy as np
import matplotlib.pyplot as plt


def _periodic_gaussian_profile(axis, center, sigma, period, n_aliases=3):
    """Return a wrapped Gaussian profile with peaks repeated every `period`."""
    axis = np.asarray(axis, dtype=np.float32)
    aliases = np.arange(-n_aliases, n_aliases + 1, dtype=np.float32) * np.float32(period)
    deltas = axis[None, :] - (np.float32(center) + aliases[:, None])
    profile = np.exp(-0.5 * (deltas / (np.float32(sigma) + 1e-12)) ** 2).sum(axis=0)
    return profile.astype(np.float32)

# Extract features from real stereo audio into the same array-encoded format
def extract_binaural_features(
    left_audio,
    right_audio,
    sr=44100,
    n_freq_bins=32,
    f_min=200.0,
    f_max=8000.0,
    n_itd_bins=32,
    itd_min=-0.8,
    itd_max=0.8,
    n_ild_bins=32,
    ild_min=-20.0,
    ild_max=20.0,
    profile_sigma_bins=2.0,
    n_time_steps=8,
    frame_length=None,
    hop_length=None,
):
    """Extract time-varying frequency, ITD, and ILD profiles from stereo audio."""
    if frame_length is None:
        frame_length = max(len(left_audio) // n_time_steps, 1)
    if hop_length is None:
        hop_length = max(frame_length // 2, 1)

    if len(left_audio) < frame_length:
        left_audio = np.pad(left_audio, (0, frame_length - len(left_audio)))
        right_audio = np.pad(right_audio, (0, frame_length - len(right_audio)))

    freq_axis = np.linspace(f_min, f_max, n_freq_bins, dtype=np.float32)
    itd_axis = np.linspace(itd_min, itd_max, n_itd_bins, dtype=np.float32)
    ild_axis = np.linspace(ild_min, ild_max, n_ild_bins, dtype=np.float32)
    itd_sigma = (itd_max - itd_min) / max(n_itd_bins - 1, 1) * profile_sigma_bins
    ild_sigma = (ild_max - ild_min) / max(n_ild_bins - 1, 1) * profile_sigma_bins

    features_over_time = []
    for start in range(0, max(len(left_audio) - frame_length + 1, 1), hop_length):
        left_frame = left_audio[start:start + frame_length]
        right_frame = right_audio[start:start + frame_length]
        if len(left_frame) < frame_length:
            left_frame = np.pad(left_frame, (0, frame_length - len(left_frame)))
            right_frame = np.pad(right_frame, (0, frame_length - len(right_frame)))

        mono = 0.5 * (left_frame + right_frame)
        spectrum = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), 1 / sr)

        freq_profile = np.interp(freq_axis, freqs, spectrum).astype(np.float32)
        freq_profile /= (np.max(freq_profile) + 1e-12)

        correlation = np.correlate(left_frame, right_frame, mode='full')
        delay_samples = np.argmax(correlation) - len(left_frame) + 1
        itd = (delay_samples / sr) * 1000.0

        rms_left = np.sqrt(np.mean(left_frame ** 2))
        rms_right = np.sqrt(np.mean(right_frame ** 2))
        ild = 20 * np.log10((rms_left + 1e-12) / (rms_right + 1e-12))

        itd_profile = np.exp(-0.5 * ((itd_axis - itd) / (itd_sigma + 1e-12)) ** 2).astype(np.float32)
        ild_profile = np.exp(-0.5 * ((ild_axis - ild) / (ild_sigma + 1e-12)) ** 2).astype(np.float32)
        itd_profile /= (np.max(itd_profile) + 1e-12)
        ild_profile /= (np.max(ild_profile) + 1e-12)

        features_over_time.append(np.concatenate([freq_profile, itd_profile, ild_profile]).astype(np.float32))

    return np.concatenate(features_over_time).astype(np.float32)

# Generate synthetic data with owl-like coding: ITD~azimuth and ILD~elevation
def create_synthetic_dataset(
    n_samples=2000,
    sr=44100,
    n_freq_bins=32,
    f_min=2000.0,
    f_max=10000.0,
    n_itd_bins=32,
    itd_min=-0.18,
    itd_max=0.18,
    n_ild_bins=32,
    ild_min=-30.0,
    ild_max=30.0,
    profile_sigma_bins=2.0,
    freq_sigma_bins=4.0,
    freq_center_jitter=0.08,
    freq_amp_jitter=0.15,
    itd_hwhh=None,
    ild_hwhh=None,
    ipd_aliases=0,
    #ild_high_freq_start=3000.0,
    n_time_steps=8,
    time_jitter=0.35,
    temporal_wobble=0.18,
):
    """Generate synthetic binaural features and source position labels with owl-like cue mapping.

    The output for each sample is a flattened time series built from repeated
    frequency, ITD, and ILD profiles across n_time_steps.

    freq_sigma_bins controls the Gaussian width of the frequency profile.
    freq_center_jitter and freq_amp_jitter control how much the frequency
    center and amplitude vary across time steps.

    itd_hwhh controls the ITD profile half-width at half-height in ms. If None,
    bin-derived width from profile_sigma_bins is used.

    If ipd_aliases > 0, the ITD response is made periodic with period 1000 / f
    ms at each time step, so a pure tone produces ambiguous alias peaks at
    integer wavelength delays.
    """
    x_train = []
    y_train = []

    freq_axis = np.linspace(f_min, f_max, n_freq_bins, dtype=np.float32)
    itd_axis = np.linspace(itd_min, itd_max, n_itd_bins, dtype=np.float32)
    ild_axis = np.linspace(ild_min, ild_max, n_ild_bins, dtype=np.float32)

    # Gaussian width per axis (in axis units) from bin count
    itd_sigma = (itd_max - itd_min) / max(n_itd_bins - 1, 1) * profile_sigma_bins
    ild_sigma = (ild_max - ild_min) / max(n_ild_bins - 1, 1) * profile_sigma_bins

    # Optional direct control of ITD profile width using HWHH: HWHH = sigma*sqrt(2*ln(2))
    if itd_hwhh is not None:
        itd_sigma = max(float(itd_hwhh), 1e-9) / np.sqrt(2.0 * np.log(2.0))
    # Optional direct control of ILD profile width using HWHH: HWHH = sigma*sqrt(2*ln(2))
    if ild_hwhh is not None:
        ild_sigma = max(float(ild_hwhh), 1e-9) / np.sqrt(2.0 * np.log(2.0))

    itd_mid = 0.5 * (itd_min + itd_max)
    itd_half = 0.5 * (itd_max - itd_min)
    ild_mid = 0.5 * (ild_min + ild_max)
    ild_half = 0.5 * (ild_max - ild_min)

    for _ in range(n_samples):
        # Random source position
        azimuth = np.random.uniform(-90, 90)
        elevation = np.random.uniform(-45, 45)

        freq_frames = []
        itd_frames = []
        ild_frames = []

        # Create a short temporal trajectory for each cue and flatten it into one vector.
        time_points = np.linspace(-1.0, 1.0, n_time_steps, dtype=np.float32)
        for time_point in time_points:
            jitter = np.sin(np.pi * time_point) * time_jitter
            temporal_scale = 1.0 + 0.5 * temporal_wobble * np.cos(np.pi * time_point)

            freq_center_base = np.random.uniform(f_min + 0.2 * (f_max - f_min), f_max - 0.2 * (f_max - f_min))
            freq_center = np.clip(
                freq_center_base + jitter * freq_center_jitter * (f_max - f_min),
                f_min,
                f_max,
            )
            freq_sigma = max(freq_sigma_bins, 1e-6) * (f_max - f_min) / max(n_freq_bins - 1, 1)
            freq_profile = np.exp(-0.5 * ((freq_axis - freq_center) / freq_sigma) ** 2).astype(np.float32)
            freq_profile *= np.float32(1.0 + freq_amp_jitter * jitter)
            freq_profile = np.clip(freq_profile * np.float32(temporal_scale), 0.0, None)

            az_rad = np.radians(azimuth + jitter * 12.0)
            el_rad = np.radians(elevation + jitter * 8.0)
            center_freq = freq_center

            # Owl-like mapping
            # ITD is the principal cue for azimuth (left-right).
            itd = itd_mid + itd_half * np.sin(az_rad)

            # # ILD is the principal cue for elevation and gets stronger at high frequency.
            # hf_gain = np.clip(
            #     (center_freq - ild_high_freq_start) / (f_max - ild_high_freq_start + 1e-12),
            #     0.0,
            #     1.0,
            # )

            ild = ild_mid + ild_half * np.sin(el_rad) #* hf_gain

            # ITD/ILD profiles over their axes.
            # For pure tones, the ITD cue becomes periodic in delay with period 1/f.
            if ipd_aliases and center_freq > 1e-9:
                itd_period_ms = 1000.0 / float(center_freq)
                itd_profile = _periodic_gaussian_profile(
                    itd_axis,
                    itd,
                    itd_sigma,
                    itd_period_ms,
                    n_aliases=int(ipd_aliases),
                )
            else:
                itd_profile = np.exp(-0.5 * ((itd_axis - itd) / (itd_sigma + 1e-12)) ** 2).astype(np.float32)
            ild_profile = np.exp(-0.5 * ((ild_axis - ild) / (ild_sigma + 1e-12)) ** 2).astype(np.float32)
            itd_profile /= (np.max(itd_profile) + 1e-12)
            ild_profile /= (np.max(ild_profile) + 1e-12)

            freq_frames.append(freq_profile)
            itd_frames.append(itd_profile)
            ild_frames.append(ild_profile)

        # Features: flattened time-varying [frequency..., ITD..., ILD...] profiles
        features = np.concatenate(
            [
                np.stack(freq_frames, axis=0).reshape(-1),
                np.stack(itd_frames, axis=0).reshape(-1),
                np.stack(ild_frames, axis=0).reshape(-1),
            ]
        ).astype(np.float32)
        x_train.append(features)

        # Targets: [azimuth, elevation]
        y_train.append([azimuth, elevation])

    return (
        np.array(x_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        freq_axis,
        itd_axis,
        ild_axis,
    )

def plot_filtered_profiles(
    x_data,
    y_data,
    freq_axis,
    itd_axis,
    ild_axis,
    azimuth_range=(-75, -25),
    elevation_range=(-15, 15),
    max_samples=200,
    random_seed=42,
    show_individual=True,
    individual_alpha=0.08,
    n_time_steps=8,
):
    """Plot filtered frequency/ITD/ILD profiles and target locations for selected samples."""
    mask = (
        (y_data[:, 0] >= azimuth_range[0]) & (y_data[:, 0] <= azimuth_range[1]) &
        (y_data[:, 1] >= elevation_range[0]) & (y_data[:, 1] <= elevation_range[1])
    )

    selected_idx = np.where(mask)[0]
    print(f'Matching samples: {len(selected_idx)}')

    if len(selected_idx) == 0:
        print('No samples matched these ranges. Widen the filters and rerun.')
        return selected_idx

    rng = np.random.default_rng(random_seed)
    if len(selected_idx) > max_samples:
        selected_idx = rng.choice(selected_idx, size=max_samples, replace=False)

    selected_x = x_data[selected_idx]
    selected_y = y_data[selected_idx]

    n_f = len(freq_axis)
    n_i = len(itd_axis)
    n_l = len(ild_axis)
    per_step = n_f + n_i + n_l
    expected_len = n_time_steps * per_step
    if selected_x.shape[1] != expected_len:
        raise ValueError(
            f'Expected flattened feature length {expected_len}, got {selected_x.shape[1]}. '
            'Pass the correct n_time_steps for this dataset.'
        )

    freq_batch = selected_x[:, : n_time_steps * n_f].reshape(len(selected_x), n_time_steps, n_f)
    itd_batch = selected_x[:, n_time_steps * n_f : n_time_steps * (n_f + n_i)].reshape(len(selected_x), n_time_steps, n_i)
    ild_batch = selected_x[:, n_time_steps * (n_f + n_i) :].reshape(len(selected_x), n_time_steps, n_l)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Show the flattened frequency block as a time-frequency spectrogram.
    freq_spectrogram = freq_batch.mean(axis=0).T
    im = axes[0, 0].imshow(
        freq_spectrogram,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, n_time_steps - 1, float(freq_axis[0]), float(freq_axis[-1])],
    )
    axes[0, 0].set_title('Frequency Spectrogram (Filtered Batch Mean)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    cbar = plt.colorbar(im, ax=axes[0, 0])
    cbar.set_label('Normalized Amplitude')

    if show_individual:
        for sample in itd_batch:
            axes[0, 1].plot(itd_axis, sample.mean(axis=0), color='tab:orange', alpha=individual_alpha)
    axes[0, 1].plot(itd_axis, itd_batch.mean(axis=(0, 1)), color='darkorange', linewidth=2.5, label='Mean')
    axes[0, 1].set_title('ITD Profiles (Filtered Batch)')
    axes[0, 1].set_xlabel('ITD (ms)')
    axes[0, 1].set_ylabel('Normalized Response')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    if show_individual:
        for sample in ild_batch:
            axes[1, 0].plot(ild_axis, sample.mean(axis=0), color='tab:green', alpha=individual_alpha)
    axes[1, 0].plot(ild_axis, ild_batch.mean(axis=(0, 1)), color='darkgreen', linewidth=2.5, label='Mean')
    axes[1, 0].set_title('ILD Profiles (Filtered Batch)')
    axes[1, 0].set_xlabel('ILD (dB)')
    axes[1, 0].set_ylabel('Normalized Response')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].scatter(y_data[:, 0], y_data[:, 1], s=8, c='lightgray', alpha=0.35, label='All samples')
    sc = axes[1, 1].scatter(
        selected_y[:, 0],
        selected_y[:, 1],
        s=35,
        edgecolor='black',
        linewidth=0.2,
        label='Filtered samples',
    )
    axes[1, 1].set_title('Filtered Target Locations')
    axes[1, 1].set_xlabel('Azimuth (deg)')
    axes[1, 1].set_ylabel('Elevation (deg)')
    axes[1, 1].set_xlim(-95, 95)
    axes[1, 1].set_ylim(-50, 50)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc='upper right')

    fig.suptitle(
        f'Filtered batch: az={azimuth_range}, el={elevation_range} | n={len(selected_idx)}',
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()

    return selected_idx

if __name__ == "__main__":
    # Example usage:
    x_train, y_train, freq_axis, itd_axis, ild_axis = create_synthetic_dataset(n_samples=2000)
    plot_filtered_profiles(x_train, y_train, freq_axis, itd_axis, ild_axis)
