from collections import deque
import time
import numpy as np
import bci_runtime_env
import mne
from pylsl import StreamInlet
from Utils.preprocessing import (
    initialize_filter_bank,
    apply_streaming_filters,
    get_valid_channel_mask_and_metadata,
    select_channels,
)

class EEGStreamState:
    def __init__(self, inlet: StreamInlet, config, mode = "motor", logger=None):
        self.inlet = inlet
        self.config = config
        self.logger = logger
        self.mode = mode

        # Override with ERRP-specific values if selected
        self.lowcut = config.LOWCUT
        self.highcut = config.HIGHCUT
        if self.mode == "errp":
            self.lowcut = config.LOWCUT_ERRP
            self.highcut = config.HIGHCUT_ERRP


        # Filtering
        self.filter_bank = initialize_filter_bank(
            fs=config.FS,
            lowcut=self.lowcut,
            highcut=self.highcut,
            notch_freqs=[60],
            notch_q=30,
            order=getattr(config, "ONLINE_FILTER_ORDER", 4),
        )
        self.filter_state = {}

        # Buffers and state
        self.filtered_buffer = deque(maxlen=config.FILTER_BUFFER_SIZE)
        self.timestamps = deque(maxlen=config.FILTER_BUFFER_SIZE)
        self.baseline_mean = None
        self.last_chunk_monotonic = None

        # Channel selection
        self.channel_names = None
        self.valid_channel_indices = None
        self.subset_indices = None
        self.final_indices = None  # legacy/original-stream indices
        self.car_reference_indices = None
        self.car_reference_channel_names = None
        self.car_reference_mode = getattr(config, "ONLINE_CAR_REFERENCE", "selected")

        self.first_chunk_processed = False

    def update(self):
        try:
            # === Pull new chunk from LSL stream ===
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.0)
            if not chunk or not timestamps:
                return  # No new data
            raw_chunk = np.array(chunk).T  # shape: (n_channels, n_samples)

            # === One-time channel selection logic ===
            if not self.first_chunk_processed:
                all_ch_names = self._get_channel_names()

                # Get valid EEG data, channel names, and MNE Raw object
                valid_channel_names, valid_raw, valid_indices = get_valid_channel_mask_and_metadata(
                    raw_chunk, all_ch_names, fs=self.config.FS, drop_mastoids=True
                )
                # Store indices of valid EEG channels in original stream
                            
                self.valid_channel_indices = valid_indices
                self.channel_names = valid_channel_names

                # Optional: select only motor-related EEG channels.  Keep
                # subset_indices relative to the valid EEG array so we can
                # optionally perform CAR on all valid EEG first, matching the
                # offline MotorCap pipeline.
                if self.mode == "motor":
                    motor_raw = select_channels(valid_raw, keep_channels = self.config.MOTOR_CHANNEL_NAMES)
                    self.subset_indices = [valid_channel_names.index(ch) for ch in motor_raw.ch_names]
                    self.channel_names = motor_raw.ch_names
                    self.final_indices = [self.valid_channel_indices[i] for i in self.subset_indices]
                elif self.mode == "errp":
                    errp_raw = select_channels(valid_raw, keep_channels = self.config.ERRP_CHANNEL_NAMES)
                    self.subset_indices = [valid_channel_names.index(ch) for ch in errp_raw.ch_names]
                    self.channel_names = errp_raw.ch_names
                    self.final_indices = [self.valid_channel_indices[i] for i in self.subset_indices]
                
                else:
                    self.subset_indices = None
                    self.final_indices = self.valid_channel_indices

                # Match the offline TimePoints CAR base: after removing
                # non-EEG/mastoids upstream, also exclude edge/frontal-temporal
                # channels before computing the online CAR.  This keeps the
                # operational model channels unchanged; it only changes the
                # reference average used when ONLINE_CAR_REFERENCE="all_valid_eeg".
                car_drop_channels = set(
                    getattr(self.config, "ONLINE_CAR_DROP_CHANNELS", [])
                )
                self.car_reference_channel_names = [
                    ch for ch in valid_channel_names if ch not in car_drop_channels
                ]
                self.car_reference_indices = [
                    self.valid_channel_indices[valid_channel_names.index(ch)]
                    for ch in self.car_reference_channel_names
                ]
                if self.car_reference_mode == "all_valid_eeg" and self.channel_names:
                    self.subset_indices = [
                        self.car_reference_channel_names.index(ch)
                        for ch in self.channel_names
                    ]

                self.first_chunk_processed = True
            # === Fast real-time slicing using precomputed indices ===
            # Keep the current online operational pipeline unchanged:
            # choose the reference set, apply CAR on the raw chunk, then run
            # the causal streaming filters. If ONLINE_CAR_REFERENCE=
            # "all_valid_eeg", CAR is computed across all valid EEG channels
            # before selecting the motor/model subset; "selected" keeps the
            # legacy selected-channel reference.
            car_all_valid = self.car_reference_mode == "all_valid_eeg"
            if car_all_valid and self.car_reference_indices is not None:
                raw_chunk = raw_chunk[self.car_reference_indices]
            elif self.final_indices is not None:
                raw_chunk = raw_chunk[self.final_indices]

            # === CAR — Common Average Reference BEFORE filtering ===
            raw_chunk = raw_chunk - raw_chunk.mean(axis=0, keepdims=True)

            # === Apply streaming filters ===
            filtered_chunk, self.filter_state = apply_streaming_filters(
                raw_chunk, self.filter_bank, self.filter_state
            )

            # === Guard: reset filter and buffer if artifact produced NaN/Inf ===
            if not np.isfinite(filtered_chunk).all():
                self.filter_state = {}
                self.filtered_buffer.clear()
                self.baseline_mean = None
                return

            if car_all_valid and self.subset_indices is not None:
                filtered_chunk = filtered_chunk[self.subset_indices]

            # === Append filtered samples to buffer ===
            for i in range(filtered_chunk.shape[1]):
                self.filtered_buffer.append(filtered_chunk[:, i])
                self.timestamps.append(timestamps[i])
            self.last_chunk_monotonic = time.monotonic()

        except Exception as e:
            if self.logger:
                self.logger.log_event(f"⚠️ Failed to update EEG stream: {e}")


    def compute_baseline(self, duration_sec=1.0, end_offset_sec=0.0):
        self.assert_stream_fresh()
        samples_needed = int(duration_sec * self.config.FS)
        offset_samples = int(end_offset_sec * self.config.FS)
        total_needed = samples_needed + offset_samples
        if len(self.filtered_buffer) < total_needed:
            raise ValueError("Not enough data in buffer to compute baseline.")

        buffer = np.array(self.filtered_buffer)
        end_idx = len(buffer) - offset_samples if offset_samples > 0 else len(buffer)
        start_idx = end_idx - samples_needed
        if start_idx < 0 or end_idx <= start_idx:
            raise ValueError("Invalid baseline window.")

        buffer_array = buffer[start_idx:end_idx]
        self.baseline_mean = buffer_array.mean(axis=0, keepdims=True).T  # shape: (n_channels, 1)

    def get_baseline_corrected_window(self, window_size_samples):
        self.assert_stream_fresh()
        if len(self.filtered_buffer) < window_size_samples:
            raise ValueError("Not enough data in buffer for window.")

        window = np.array(self.filtered_buffer)[-window_size_samples:].T  # shape: (n_channels, samples)
        if self.baseline_mean is not None:
            window -= self.baseline_mean
        return window, list(self.timestamps)[-window_size_samples:]

    def assert_stream_fresh(self, max_age_sec=None):
        """Fail closed instead of classifying a stale LSL buffer."""
        if max_age_sec is None:
            max_age_sec = float(
                getattr(self.config, "EEG_STREAM_MAX_AGE_S", 1.0)
            )
        if self.last_chunk_monotonic is None:
            raise RuntimeError("EEG stream has not delivered any samples.")
        age = time.monotonic() - self.last_chunk_monotonic
        if age > max_age_sec:
            raise RuntimeError(
                f"EEG stream is stale: last chunk received {age:.3f} s ago "
                f"(limit={max_age_sec:.3f} s)."
            )
    
    def _get_channel_names(self):
        """
        Read raw channel labels from the LSL stream metadata without any renaming.
        Renaming/normalization should be done in the preprocessing helper only.
        """
        info = self.inlet.info()
        ch = info.desc().child("channels").child("channel")
        names = []

        while ch.name():
            label_node = ch.child("label").first_child()
            if not label_node:
                raise RuntimeError("Channel label missing in LSL stream metadata")
            names.append(label_node.value())  # <-- raw, unmodified label
            ch = ch.next_sibling()

        return names




    def _make_dummy_info(self):
        ch_names = self._get_channel_names()
        sfreq = self.config.FS
        ch_types = ['eeg'] * len(ch_names)
        return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
