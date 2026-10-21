from pathlib import Path
import mne
import numpy as np

def load_and_preprocess(path, l_freq=1.0, h_freq=45.0, target_sfreq=256, picks="eeg"):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported EEG extension: {ext}")
    raw.pick(picks)
    raw.filter(l_freq, h_freq, method="iir",
               iir_params={"order": 4, "ftype": "butter"}, phase="zero",
               verbose="ERROR")
    if target_sfreq and raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    return raw

def fixed_epochs(raw, epoch_seconds=2.0, overlap=0.0):
    events = mne.make_fixed_length_events(raw, duration=epoch_seconds, overlap=overlap)
    return mne.Epochs(raw, events, tmin=0, tmax=epoch_seconds, baseline=None,
                      preload=True, reject_by_annotation=True, verbose="ERROR")
