#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:14:44 2025

@author: jameslim
"""

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300       # for on-screen display
mpl.rcParams['savefig.dpi'] = 300      # for saving figures


# In[]
import numpy as np

# Example: Replace this with your actual path
file_path = r"/Volumes/PRESTIGE/Slice - Rat/Slice_250521/1463A/tissue/stim/SC_B5-B6_DIV2/STIM_CC5.0_PC1.0_PW0.1_PositiveFirst_PS1_CS0_IPD0.2/Outputs_LFP_zoomed/extracted_segments_LFP_Ch37.npy"

# Load the specified .npy file
try:
    extracted_segments1 = np.load(file_path)
    print("File loaded successfully.")
    print("Data shape:", extracted_segments1.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")

# Example: Replace this with your actual path
file_path = r"/Volumes/PRESTIGE/Slice - Rat/Slice_250521/1463A/tissue/stim/SC_B5-B6_DIV2/STIM_CC5.0_PC1.0_PW0.1_PositiveFirst_PS1_CS0_IPD0.2/Outputs_LFP_zoomed/extracted_segments_LFP_Ch38.npy"
# Load the specified .npy file
try:
    extracted_segments2 = np.load(file_path)
    print("File loaded successfully.")
    print("Data shape:", extracted_segments2.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")

# Example: Replace this with your actual path
file_path = r"/Volumes/PRESTIGE/Slice - Rat/Slice_250521/1463A/notissue/CW/TEMP/STIM_CC5.0_PC1.0_PW0.1_PositiveFirst_PS1_CS0_IPD0.2/Outputs_LFP_zoomed/extracted_segments_LFP_Ch37.npy"

# Load the specified .npy file
try:
    extracted_segments_mediaonly1 = np.load(file_path)
    print("File loaded successfully.")
    print("Data shape:", extracted_segments_mediaonly1.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")

# Example: Replace this with your actual path
file_path = r"/Volumes/PRESTIGE/Slice - Rat/Slice_250521/1463A/notissue/CW/TEMP/STIM_CC5.0_PC1.0_PW0.1_PositiveFirst_PS1_CS0_IPD0.2/Outputs_LFP_zoomed/extracted_segments_LFP_Ch38.npy"

# Load the specified .npy file
try:
    extracted_segments_mediaonly2 = np.load(file_path)
    print("File loaded successfully.")
    print("Data shape:", extracted_segments_mediaonly2.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")

# Load corresponding baseline recording already concatenated
"""Notch and downsampling to 2kHz has already been done on this array"""
# Example: Replace this with your actual path
file_path = r"/Volumes/PRESTIGE/Slice - Rat/Slice_250521/1463A/tissue/baseline/DIV1_250522_081314/concatenated_signal_low_pass_ds.npy"
# Load the specified .npy file
try:
    baseline_activity = np.load(file_path)
    print("File loaded successfully.")
    print("Data shape:", baseline_activity.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")
    
# Load impedance info
# Example: Replace this with your actual path
file_path = r"/Volumes/PRESTIGE/Slice - Rat/Slice_250521/1463A/tissue/stim/SC_B5-B6_DIV2/impedance.csv"
# Load the specified .npy file
try:
    # Load the CSV file into a NumPy array
    impedance_info = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    print("File loaded successfully.")
    print("Data:", impedance_info)
    print('Channels and Headers: ', impedance_info.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")
# In[] Exclude channels with impedances greater than 2Mohm
import numpy as np

IMPEDANCE_THRESH = 2e6  # 2 Megaohm

def get_low_impedance_channels(impedances, thresh=IMPEDANCE_THRESH):
    """
    Return indices of channels with impedance <= threshold.
    """
    impedances = np.asarray(impedances)
    return np.where(impedances <= thresh)[0]

low_imp_chs = get_low_impedance_channels(impedance_info[:,4])
print(f"{len(low_imp_chs)} channels pass impedance threshold")

extracted_segments1_imp = extracted_segments1[:, low_imp_chs, :]
extracted_segments2_imp = extracted_segments2[:, low_imp_chs, :]
extracted_segments_mediaonly1_imp = extracted_segments_mediaonly1[:, low_imp_chs, :]
extracted_segments_mediaonly2_imp = extracted_segments_mediaonly2[:, low_imp_chs, :]

# In[]
import scipy.signal
from scipy import signal
# ========== APPLY NOTCH FILTER ==========
def apply_notch_filter(data, notch_freq, sampling_rate, quality_factor=30):
    """
    Applies a notch filter to each channel in the data array.

    Parameters:
    - data: numpy array with shape (channels, samples)
    - notch_freq: float, frequency to remove (e.g., 60 Hz)
    - sampling_rate: int, samples per second
    - quality_factor: float, controls notch width (higher = narrower)

    Returns:
    - filtered_data: numpy array with the same shape as input
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
    filtered_data = np.zeros_like(data)

    for channel in range(data.shape[0]):
        signal_data = data[channel, :]
        if len(signal_data) < 3 * max(len(a), len(b)):
            raise ValueError("Signal too short for filtfilt.")
        filtered_data[channel, :] = signal.filtfilt(b, a, signal_data)

    return filtered_data

fs = 2000  # Hz

extracted_segments1_imp = apply_notch_filter(extracted_segments1_imp, 60, fs)
extracted_segments2_imp = apply_notch_filter(extracted_segments2_imp, 60, fs)
extracted_segments_mediaonly1_imp = apply_notch_filter(extracted_segments_mediaonly1_imp, 60, fs)
extracted_segments_mediaonly2_imp = apply_notch_filter(extracted_segments_mediaonly2_imp, 60, fs)


# In[] highpass filter some segments
"""use these segments for selecting active channels. The purpose is to screen for activity without the lingering 
effects of capacitive discharge and artifact in the delta range. But the original extracted_segments_imp arrays 
will be used for PSD calculation and feature extraction"""
from scipy.signal import butter, filtfilt

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

lowcut=4
highcut=30
extracted_segments1_lowband = bandpass_filter(extracted_segments1_imp, lowcut, highcut, fs)
extracted_segments2_lowband = bandpass_filter(extracted_segments2_imp, lowcut, highcut, fs)
extracted_segments1_430 = bandpass_filter(extracted_segments1, lowcut, highcut, fs)
extracted_segments2_430 = bandpass_filter(extracted_segments2, lowcut, highcut, fs)

extracted_segments_mediaonly1_lowband = bandpass_filter(extracted_segments_mediaonly1_imp, lowcut, highcut, fs)
extracted_segments_mediaonly2_lowband = bandpass_filter(extracted_segments_mediaonly2_imp, lowcut, highcut, fs)

# In[3] Filter theta, gamma, delta, and sharp wave ripple bands and use as input features for classification
"""Filter theta, gamma, delta, and sharp wave ripple bands and use as input features for classification"""

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

# Bandpass filter function
def lowpass_filter(data, highcut, fs, order=4):
    nyquist = 0.5 * fs
    # low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, high, btype="lowpass")
    return filtfilt(b, a, data)

# In[] correct way to split theta and beta bins
import numpy as np
from scipy.signal import welch

def compute_psd_bins_theta_beta(signal, bin_size, start_time=2, end_time=7, fs=2000, nperseg=256):
    """
    Compute PSD features for theta (4-12 Hz) and beta (12-30 Hz) bands
    per bin, per channel, for each trial.
    
    Returns:
        psd_bins: np.array, shape [trials, bins, channels, 2] (theta, beta)
    """
    trials, channels, num_samples = signal.shape

    # Compute start and end indices
    start_idx = int(start_time * fs)
    end_idx   = int(end_time * fs)

    num_bins = (end_idx - start_idx) // bin_size

    # Output array: trials x bins x channels x 2 (theta, beta)
    psd_bins = np.zeros((trials, num_bins, channels, 2))

    for i in range(num_bins):
        start = start_idx + i * bin_size
        stop  = start_idx + (i + 1) * bin_size

        for trial in range(trials):
            for channel in range(channels):
                segment = signal[trial, channel, start:stop]

                if len(segment) == 0:
                    continue

                # Compute PSD
                freqs, psd = welch(segment, fs=fs, nperseg=nperseg)

                # Band masks
                theta_mask = (freqs >= 4) & (freqs <= 12)
                beta_mask  = (freqs > 12) & (freqs <= 30)

                # Integrate power in each band
                theta_power = np.trapz(psd[theta_mask], freqs[theta_mask])
                beta_power  = np.trapz(psd[beta_mask],  freqs[beta_mask])

                psd_bins[trial, i, channel, 0] = theta_power
                psd_bins[trial, i, channel, 1] = beta_power

    # Log-transform for stability
    psd_bins = np.log10(psd_bins + 1e-12)

    # Flatten for classifier: trials x (bins*channels*2)
    return psd_bins
# unflattened
X_tb1 = compute_psd_bins_theta_beta(extracted_segments1_lowband, bin_size=1000, start_time=4, end_time=7, nperseg=1024) # excludes last 32 and ch13 and 15
X_tb2 = compute_psd_bins_theta_beta(extracted_segments2_lowband, bin_size=1000, start_time=4, end_time=7, nperseg=1024)

theta_psd_bins1 = X_tb1[..., 0]
beta_psd_bins1  = X_tb1[..., 1]

theta_psd_bins2 = X_tb2[..., 0]
beta_psd_bins2  = X_tb2[..., 1]

#reshape/flatten for classifiers
thetabeta1 = X_tb1.reshape(X_tb1.shape[0], -1)
theta_psd_bins1 = theta_psd_bins1.reshape(theta_psd_bins1.shape[0], -1)
beta_psd_bins1 = beta_psd_bins1.reshape(beta_psd_bins1.shape[0], -1)

thetabeta2 = X_tb2.reshape(X_tb2.shape[0], -1)
theta_psd_bins2 = theta_psd_bins2.reshape(theta_psd_bins2.shape[0], -1)
beta_psd_bins2 = beta_psd_bins2.reshape(beta_psd_bins2.shape[0], -1)

# In[]z-score to baseline activity recorded before experiment
import numpy as np

# omit high impedance channels
baseline_activity_imp = baseline_activity[low_imp_chs,:]

def make_pseudo_trials(
    baseline_activity,
    fs=2000,
    trial_duration_sec=3.0,
    drop_remainder=True
):
    """
    baseline_activity: (channels, samples)
    returns: (trials, channels, samples_per_trial)
    """

    channels, total_samples = baseline_activity.shape
    samples_per_trial = int(trial_duration_sec * fs)

    n_trials = total_samples // samples_per_trial

    if drop_remainder:
        baseline_activity = baseline_activity[:, :n_trials * samples_per_trial]

    # reshape
    baseline_trials = baseline_activity.reshape(
        channels,
        n_trials,
        samples_per_trial
    ).transpose(1, 0, 2)

    return baseline_trials
baseline_trials = make_pseudo_trials(
    baseline_activity_imp,
    fs=2000,
    trial_duration_sec=3.0
)

print(baseline_trials.shape)
# (n_baseline_trials, channels, 2000)

X_baseline = compute_psd_bins_theta_beta(
    baseline_trials,
    bin_size=1000,     # same as post-stimulus
    start_time=0,      # baseline starts at beginning of pseudo-trial
    end_time=3.0,      # match trial length
    fs=2000,
    nperseg=1024
)
# Collapse baseline across trials
baseline_mean = X_baseline.mean(axis=0)  # (bins, channels, 2)
baseline_std  = X_baseline.std(axis=0)   # (bins, channels, 2)

# Apply feature-wise normalization
X_tb1_z = (X_tb1 - baseline_mean) / baseline_std
X_tb2_z = (X_tb2 - baseline_mean) / baseline_std

# Flatten for classifiers
thetabeta1_z = X_tb1_z.reshape(X_tb1_z.shape[0], -1)
thetabeta2_z = X_tb2_z.reshape(X_tb2_z.shape[0], -1)

theta_psd_bins1_z = X_tb1_z[..., 0].reshape(X_tb1_z.shape[0], -1)
beta_psd_bins1_z  = X_tb1_z[..., 1].reshape(X_tb1_z.shape[0], -1)

theta_psd_bins2_z = X_tb2_z[..., 0].reshape(X_tb2_z.shape[0], -1)
beta_psd_bins2_z  = X_tb2_z[..., 1].reshape(X_tb2_z.shape[0], -1)

# In[] z-score the early and late window
early_bins = [0, 1, 2]
late_bins  = [3, 4, 5]

X1_early = X_tb1_z[:, early_bins, :, :]
X1_late  = X_tb1_z[:, late_bins, :, :]

X2_early = X_tb2_z[:, early_bins, :, :]
X2_late  = X_tb2_z[:, late_bins, :, :]

X1_early_flat = X1_early.reshape(X1_early.shape[0], -1)
X1_late_flat  = X1_late.reshape(X1_late.shape[0], -1)

X2_early_flat = X2_early.reshape(X2_early.shape[0], -1)
X2_late_flat  = X2_late.reshape(X2_late.shape[0], -1)

# In[]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
}

# In[]: block-aware holdout + CV with AB-pair grouping
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_features(
    X1, X2, label,
    # classifiers,
    dim_reduction=None,
    n_components=10,
    block_size=2,     # number of trials per block; 2 = one AB pair
    n_splits=5,
    random_state=42,
    plot_roc=True
):
    """
    Evaluate classifiers on PSD features with:
      - Block-aware holdout (GroupShuffleSplit) to prevent adjacency leakage
      - Block-aware cross-validation (GroupKFold) with AB-pair grouping

    Assumptions:
      - Trials alternate strictly A, B, A, B, ...
      - X1 are A-trial features, X2 are B-trial features
      - We pair A[i] with B[i] into the same group; block_size=2 groups one AB pair
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # ---------- Validate inputs ----------
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of features (columns).")

    # For strict alternation, ensure pairs align; if unequal, truncate to min length
    n1, n2 = X1.shape[0], X2.shape[0]
    n_pairs = min(n1, n2)
    if n1 != n2:
        warnings.warn(f"X1 and X2 have unequal lengths (n1={n1}, n2={n2}); truncating to {n_pairs} pairs.")
        X1 = X1[:n_pairs]
        X2 = X2[:n_pairs]

    # block_size must be an even number of trials under AB alternation (e.g., 2, 4, 6 ...)
    if block_size % 2 != 0:
        raise ValueError("block_size must be even (each AB pair = 2 trials).")

    pairs_per_block = block_size // 2

    # ---------- Construct dataset (stack A then B) ----------
    X = np.vstack((X1, X2))
    y = np.concatenate((np.zeros(n_pairs, dtype=int), np.ones(n_pairs, dtype=int)))

    # ---------- Build AB-pair-based group labels ----------
    # A[i] and B[i] share the same pair index i
    pair_idx = np.arange(n_pairs, dtype=int)
    block_ids = pair_idx // pairs_per_block  # group consecutive AB pairs into a block

    # Map to sample-wise groups for stacked X = [A...; B...]
    groups = np.empty(2 * n_pairs, dtype=int)
    groups[:n_pairs] = block_ids        # A block ids
    groups[n_pairs:] = block_ids        # B block ids  (same as paired A)

    n_unique_groups = np.unique(groups).size

    # ---------- Containers ----------
    report_rows = []
    roc_info = {}
    confusion_rows = []

    # ===============================
    # BLOCK-AWARE HOLDOUT (GroupShuffleSplit)
    # ===============================
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    (train_idx, test_idx), = gss.split(X, y, groups=groups)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for clf_name, clf in classifiers.items():
        steps = [("scaler", StandardScaler())]
        if dim_reduction == "PCA":
            steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
        steps.append(("clf", clf))
        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # ---- Classification report (per-class & macro/weighted) ----
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    report_rows.append({
                        "Condition": label,
                        "DimReduction": dim_reduction or "None",
                        "Classifier": clf_name,
                        "Class": cls,
                        "Metric": metric_name,
                        "Value": metric_value,
                        "Type": "Holdout"
                    })

        # ---- Confusion matrix (row-normalized, safe divide) ----
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        confusion_rows.append({
            "Condition": label,
            "DimReduction": dim_reduction or "None",
            "Classifier": clf_name,
            "TN": cm_norm[0, 0],
            "FP": cm_norm[0, 1],
            "FN": cm_norm[1, 0],
            "TP": cm_norm[1, 1],
            "Type": "Holdout"
        })

        # ---- ROC / AUC (skip if single-class in test) ----
        try:
            if len(np.unique(y_test)) < 2:
                raise ValueError("Single-class holdout test set; ROC/AUC undefined.")
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_score = pipe.predict_proba(X_test)[:, 1]
            else:
                y_score = pipe.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_val = auc(fpr, tpr)
            roc_info[clf_name] = (fpr, tpr, auc_val)

            report_rows.append({
                "Condition": label,
                "DimReduction": dim_reduction or "None",
                "Classifier": clf_name,
                "Class": "overall",
                "Metric": "AUC",
                "Value": auc_val,
                "Type": "Holdout"
            })
        except Exception as e:
            print(f"[WARN] ROC/AUC skipped for {clf_name}: {e}")

    # ---------- Plot ROC ----------
    if plot_roc and len(roc_info) > 0:
        plt.figure(figsize=(8, 6))
        for clf_name, (fpr, tpr, auc_val) in roc_info.items():
            plt.plot(fpr, tpr, lw=2, label=f"{clf_name} (AUC={auc_val:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {label}, DimRed: {dim_reduction or 'None'}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ===============================
    # BLOCK-AWARE CROSS-VALIDATION (GroupKFold)
    # ===============================
    n_splits_eff = min(n_splits, n_unique_groups)
    if n_splits_eff < 2:
        raise ValueError(
            f"Not enough unique groups ({n_unique_groups}) for CV with n_splits={n_splits}. "
            "Increase data or reduce n_splits."
        )

    gkf = GroupKFold(n_splits=n_splits_eff)
    cv_results = {}

    for clf_name, clf in classifiers.items():
        steps = [("scaler", StandardScaler())]
        if dim_reduction == "PCA":
            steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
        steps.append(("clf", clf))
        pipe = Pipeline(steps)

        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=gkf,
            groups=groups,   # enforces block exclusivity
            scoring="accuracy",
            n_jobs=None
        )

        cv_results[f"{label}_{dim_reduction or 'None'}_{clf_name}"] = scores

        report_rows.append({
            "Condition": label,
            "DimReduction": dim_reduction or "None",
            "Classifier": clf_name,
            "Class": "overall",
            "Metric": "CV_Mean",
            "Value": scores.mean(),
            "Type": "CrossVal"
        })

        report_rows.append({
            "Condition": label,
            "DimReduction": dim_reduction or "None",
            "Classifier": clf_name,
            "Class": "overall",
            "Metric": "CV_Std",
            "Value": scores.std(ddof=1),
            "Type": "CrossVal"
        })

    # ---------- Output ----------
    summary_table = pd.DataFrame(report_rows)
    confusion_table = pd.DataFrame(confusion_rows)

    return summary_table, confusion_table, X, y, cv_results

# In[] run evaluate features and create tables for PSD on slice - z-scored
# Collect all results
all_summaries = []
all_confusion = []
# Theta/Beta
thetabeta_none, confusion_table_tb, x_scaled, y, thetabeta_results_none = evaluate_features(thetabeta1_z, thetabeta2_z, "ThetaBeta PSD", dim_reduction=None, n_components=10)
thetabeta_pca, _, _, _, thetabeta_results_pca = evaluate_features(thetabeta1_z, thetabeta2_z, "ThetaBeta PSD", dim_reduction="PCA", n_components=2)
all_summaries += [thetabeta_none]
all_confusion += [confusion_table_tb]

# Theta
theta_none, confusion_table_t,_, _, theta_results_none = evaluate_features(theta_psd_bins1_z, theta_psd_bins2_z, "Theta PSD", dim_reduction=None, n_components=10)
theta_pca, _, _, _, theta_results_pca = evaluate_features(theta_psd_bins1_z, theta_psd_bins2_z, "Theta PSD", dim_reduction="PCA", n_components=2)

# all_summaries += [theta_none, theta_pca]
all_summaries += [theta_none]
all_confusion += [confusion_table_t]

# Beta
beta_none, confusion_table_b, _, _, beta_results_none = evaluate_features(beta_psd_bins1_z, beta_psd_bins2_z, "Beta PSD", dim_reduction=None, n_components=10)
beta_pca, _, _, _, beta_results_pca = evaluate_features(beta_psd_bins1_z, beta_psd_bins2_z, "Beta PSD", dim_reduction="PCA", n_components=2)
 
# all_summaries += [beta_none, beta_pca]
all_summaries += [beta_none]
all_confusion += [confusion_table_b]

# Merge
final_summary = pd.concat(all_summaries, ignore_index=True)

# Pivot for readability (CV + holdout all together)
final_pivot = final_summary.pivot_table(
    index=["Condition","DimReduction","Classifier","Class"],
    columns="Metric",
    values="Value"
).reset_index()

print(final_pivot.head(20))
import os
import pandas as pd

# # Example: path to one of your source files
source_file = file_path #"/path/to/your/source_file.csv"
# Get the folder containing the source file
source_folder = os.path.dirname(source_file)
# Create the full path for the CSV to save
save_path = os.path.join(source_folder, "slice02_classification_summary_pivot_052125_B5B6.csv")
# Save the pivot table
final_pivot.to_csv(save_path, index=False)
print(f"Saved pivot table to: {save_path}")

# ###########################################################################
final_confusion = pd.concat(all_confusion, ignore_index=True)
final_confusion.to_csv(
    os.path.join(source_folder, "slice02_confusion_matrices_normalized_052125_B5B6.csv"),
    index=False
)

###########################################################################

pipeline_results = {**thetabeta_results_none, **theta_results_none, **beta_results_none}

# Results DataFrame
df = pd.DataFrame([
    {"Pipeline": key, "Accuracy": acc} 
    for key, values in pipeline_results.items() for acc in values
])

# Box plot
plt.figure(figsize=(20, 12))
sns.boxplot(data=df, y="Pipeline", x="Accuracy", orient="h", palette="Set2")
plt.title("Post-stimulus PSD Features - Z-scored",fontsize=50, fontweight="bold")
plt.xlabel("Accuracy", fontsize = 25, fontweight="bold")
plt.ylabel("Pipeline", fontsize = 25, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")  # adjust 16 → larger/smaller as needed
plt.yticks(fontsize=20, fontweight="bold")  # adjust 16 → larger/smaller as needed
plt.xlim(0, 1)
plt.axvline(0.5, linestyle='--',color='black')
# Add horizontal grid lines aligned with y ticks
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# In[] run updated evaluate features and create tables for early and late window PSD on slice - z scored
# Collect all results
all_summaries = []
all_confusion = []


# Early
early_none, confusion_table_early,_, _, early_results_none = evaluate_features(X1_early_flat, X2_early_flat, "Early PSD", dim_reduction=None, n_components=10)
early_pca, _, _, _, early_results_pca = evaluate_features(X1_early_flat, X2_early_flat, "Early PSD", dim_reduction="PCA", n_components=2)

# all_summaries += [theta_none, theta_pca]
all_summaries += [early_none]
all_confusion += [confusion_table_early]

# Late
late_none, confusion_table_late, _, _, late_results_none = evaluate_features(X1_late_flat, X2_late_flat, "Late PSD", dim_reduction=None, n_components=10)
late_pca, _, _, _, late_results_pca = evaluate_features(X1_late_flat, X2_late_flat, "Late PSD", dim_reduction="PCA", n_components=2)
 
# all_summaries += [beta_none, beta_pca]
all_summaries += [late_none]
all_confusion += [confusion_table_late]

# Merge
final_summary = pd.concat(all_summaries, ignore_index=True)

# Pivot for readability (CV + holdout all together)
final_pivot = final_summary.pivot_table(
    index=["Condition","DimReduction","Classifier","Class"],
    columns="Metric",
    values="Value"
).reset_index()

print(final_pivot.head(20))

import os
import pandas as pd

# # Example: path to one of your source files
# source_file = file_path #"/path/to/your/source_file.csv"
# # Get the folder containing the source file
# source_folder = os.path.dirname(source_file)
# # Create the full path for the CSV to save
# save_path = os.path.join(source_folder, "slice03_classification_summary_pivot_052125_B5B6.csv")
# # Save the pivot table
# final_pivot.to_csv(save_path, index=False)
# print(f"Saved pivot table to: {save_path}")

# ###########################################################################
# final_confusion = pd.concat(all_confusion, ignore_index=True)
# final_confusion.to_csv(
#     os.path.join(source_folder, "slice03_confusion_matrices_normalized_052125_B5B6.csv"),
#     index=False
# )

###########################################################################

# pipeline_results = {**early_results_none,**late_results_none}
pipeline_results = {**early_results_none,**late_results_none}


# Results DataFrame
df = pd.DataFrame([
    {"Pipeline": key, "Accuracy": acc} 
    for key, values in pipeline_results.items() for acc in values
])

# Box plot
plt.figure(figsize=(20, 12))
sns.boxplot(data=df, y="Pipeline", x="Accuracy", orient="h", palette="Set2")
plt.title("Post-stimulus PSD Features - Early vs Late Window",fontsize=50, fontweight="bold")
plt.xlabel("Accuracy", fontsize = 25, fontweight="bold")
plt.ylabel("Pipeline", fontsize = 25, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")  # adjust 16 → larger/smaller as needed
plt.yticks(fontsize=20, fontweight="bold")  # adjust 16 → larger/smaller as needed
plt.xlim(0, 1)
plt.axvline(0.5, linestyle='--',color='black')
# Add horizontal grid lines aligned with y ticks
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# In[] cross validated ROC curves
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
def compute_cv_roc(clf, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        # interpolate TPR to common FPR grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc  = np.std(aucs)

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, aucs
plt.figure(figsize=(8, 6))

for clf_name, clf in classifiers.items():
    fpr, tpr, std_tpr, mean_auc, std_auc, aucs = compute_cv_roc(clf, x_scaled, y)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{clf_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
    )

    plt.fill_between(
        fpr,
        tpr - std_tpr,
        tpr + std_tpr,
        alpha=0.2
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate", fontsize=20, fontweight="bold")
plt.ylabel("True Positive Rate", fontsize=20, fontweight="bold")
plt.title("Cross-Validated ROC Curves - Theta", fontsize=25, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()


# In[]: Improved, publication-safe, block-aware permutation testing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
    cross_val_score
)
from sklearn.base import clone


# ---------------------------
# Helpers
# ---------------------------

def build_X_y_groups(X1, X2, block_size=2):
    """
    Build (X, y, groups) for strictly alternating A,B,A,B,... design.

    - X1: features for A trials, shape (nA, d)
    - X2: features for B trials, shape (nB, d)
    - block_size: number of *trials* per CV block. 2 = one AB pair, 4 = ABAB, etc.

    Returns
    -------
    X : (2*n_pairs, d)
    y : (2*n_pairs,)
    groups : (2*n_pairs,)  # block IDs, same for paired A[i], B[i]
    """
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have same number of features (columns).")

    if block_size % 2 != 0:
        raise ValueError("block_size must be even (each AB pair = 2 trials).")

    n_pairs = min(X1.shape[0], X2.shape[0])
    if X1.shape[0] != X2.shape[0]:
        print(f"[WARN] Unequal counts (A={X1.shape[0]}, B={X2.shape[0]}). "
              f"Truncating to {n_pairs} AB pairs to respect alternation.")

    X1 = X1[:n_pairs]
    X2 = X2[:n_pairs]

    # Stack A then B
    X = np.vstack((X1, X2))
    y = np.concatenate((np.zeros(n_pairs, dtype=int), np.ones(n_pairs, dtype=int)))

    # Grouping: A[i] and B[i] share the same AB-pair index; then we form larger blocks if requested
    pairs_per_block = block_size // 2
    pair_idx = np.arange(n_pairs, dtype=int)
    block_ids = pair_idx // pairs_per_block  # group consecutive AB pairs into a block

    groups = np.empty(2 * n_pairs, dtype=int)
    groups[:n_pairs] = block_ids      # A trials
    groups[n_pairs:] = block_ids      # B trials -> same block as paired A

    return X, y, groups


def make_group_cv(groups, y, n_splits=5, random_state=42):
    """
    Prefer StratifiedGroupKFold; fallback to GroupKFold if not available.
    Clamp n_splits to number of unique groups.
    """
    n_unique_groups = np.unique(groups).size
    n_splits_eff = max(2, min(n_splits, n_unique_groups))

    try:
        cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    except Exception:
        # scikit-learn < 1.1 or environments without SGKF
        cv = GroupKFold(n_splits=n_splits_eff)
    return cv


def materialize_splits(cv, X, y, groups):
    """Materialize CV splits so the exact same folds are reused for permutations."""
    return list(cv.split(X, y, groups))


def permute_labels(y, groups, rng, mode="group"):
    """
    Create a permuted label vector that respects grouping.

    mode:
      - "group": shuffle group-level labels and broadcast to members
      - "within-group": permute labels within each group independently
      - "global": full free permutation (IID; not recommended for temporally dependent data)
    """
    y = np.asarray(y)

    if groups is None or mode == "global":
        return rng.permutation(y)

    groups = np.asarray(groups)

    if mode == "group":
        unique_groups, inv = np.unique(groups, return_inverse=True)
        # Majority (or mean) label per group; for AB pairs this is usually 0.5 (one A, one B).
        # If blocks can have variable composition, majority is robust.
        grp_vals = np.zeros(unique_groups.shape[0], dtype=float)
        for gi, g in enumerate(unique_groups):
            grp_vals[gi] = y[groups == g].mean()
        grp_labels = (grp_vals >= 0.5).astype(int)
        perm_grp_labels = grp_labels.copy()
        rng.shuffle(perm_grp_labels)
        return perm_grp_labels[inv]

    if mode == "within-group":
        y_perm = y.copy()
        for g in np.unique(groups):
            mask = (groups == g)
            y_perm[mask] = rng.permutation(y[mask])
        return y_perm

    raise ValueError("mode must be one of {'group','within-group','global'}.")


def permutation_test_cv(
    X, y, pipeline, splits, groups="within-group",
    n_permutations=1000, random_state=42, scoring="accuracy",
    permute_mode="group", n_jobs_cv=None
):
    """
    Publication-safe permutation test for cross-validated score with group-aware options.

    Parameters
    ----------
    X, y : arrays
    pipeline : estimator or Pipeline
    splits : materialized CV splits (list of (train_idx, test_idx))
    groups : array or None (aligns with X/y)
    n_permutations : int
    random_state : int
    scoring : str
    permute_mode : "group" | "within-group" | "global"
    n_jobs_cv : int or None, forwarded to cross_val_score

    Returns
    -------
    dict with keys:
      - true_scores : array (n_splits,)
      - true_mean   : float
      - perm_scores : array (n_permutations,)
      - p_value     : float (Monte Carlo p with +1 correction)
    """
    rng = np.random.default_rng(random_state)

    true_scores = cross_val_score(
        clone(pipeline), X, y, cv=splits, groups=groups, scoring=scoring, n_jobs=n_jobs_cv
    )
    true_mean = true_scores.mean()

    perm_scores = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = permute_labels(y, groups, rng, mode=permute_mode)
        perm_scores[i] = cross_val_score(
            clone(pipeline), X, y_perm, cv=splits, groups=groups, scoring=scoring, n_jobs=n_jobs_cv
        ).mean()

    # Exact Monte-Carlo p-value with +1 correction
    p_value = (np.sum(perm_scores >= true_mean) + 1.0) / (n_permutations + 1.0)
    return {"true_scores": true_scores, "true_mean": float(true_mean),
            "perm_scores": perm_scores, "p_value": float(p_value)}


# ---------------------------
# User configuration
# ---------------------------

# Bands dictionary provided by you (must exist):
bands = {
    "ThetaBeta": (thetabeta1_z, thetabeta2_z),
    "Theta": (theta_psd_bins1_z, theta_psd_bins2_z),
    "Beta": (beta_psd_bins1_z, beta_psd_bins2_z),
}

block_size = 2          # 2 = one AB pair; enlarge (4,6,...) to enforce longer blocks
n_splits = 5
n_permutations = 500
random_state = 42
permute_mode = "within-group"  # "group" | "within-group" | "global" (use "group" for AB alternation)
var_threshold = 1e-12   # matches your previous variance filter


# Define classifier pipelines *without* leakage: all preprocessing in-pipeline
pipelines = {
    "Logistic Regression": Pipeline([
        ("vt",     VarianceThreshold(threshold=var_threshold)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=random_state))
    ]),
    "SVM": Pipeline([
        ("vt",     VarianceThreshold(threshold=var_threshold)),
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="linear"))
    ]),
    "Random Forest": Pipeline([
        ("vt",     VarianceThreshold(threshold=var_threshold)),
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(random_state=random_state))
    ]),
    "KNN": Pipeline([
        ("vt",     VarianceThreshold(threshold=var_threshold)),
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(n_neighbors=5))
    ]),
    "LDA": Pipeline([
        ("vt",     VarianceThreshold(threshold=var_threshold)),
        ("scaler", StandardScaler()),
        ("clf",    LinearDiscriminantAnalysis())
    ])
}


# ---------------------------
# Main loop across bands and classifiers
# ---------------------------

all_perm_results = {}
summary_rows = []
long_rows = []

for band_name, (X1, X2) in bands.items():
    # Build dataset and AB-pair block groups
    X, y, groups = build_X_y_groups(X1, X2, block_size=block_size)

    # Build group-aware CV
    cv = make_group_cv(groups, y, n_splits=n_splits, random_state=random_state)

    # Materialize splits ONCE and reuse across permutations
    splits = materialize_splits(cv, X, y, groups)

    for clf_name, pipe in pipelines.items():
        res = permutation_test_cv(
            X=X, y=y, pipeline=pipe, splits=splits, groups=groups,
            n_permutations=n_permutations, random_state=random_state,
            scoring="accuracy", permute_mode=permute_mode, n_jobs_cv=None
        )

        all_perm_results[f"{band_name} - {clf_name}"] = res

        # --- Plot null vs observed for quick QC ---
        perm = res["perm_scores"]
        true_mean = res["true_mean"]
        p_value = res["p_value"]

        plt.figure(figsize=(8, 5))
        plt.hist(perm, bins=40, alpha=0.7)
        plt.axvline(true_mean, color='red', linewidth=3, label="Observed")
        plt.title(f"Permutation Test — {band_name} - {clf_name}")
        plt.xlabel("Cross-validated accuracy")
        plt.ylabel("Count")
        plt.xlim(0, 1)
        plt.text(
            0.98, 0.02, f"p = {p_value:.4f}",
            ha='right', va='bottom', transform=plt.gca().transAxes,
            fontsize=14, bbox=dict(boxstyle="round", fc="white", alpha=0.8)
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- Add to summary tables ---
        summary_rows.append({
            "Band": band_name,
            "Classifier": clf_name,
            "Observed_CV_Accuracy": true_mean,
            "Permutation_Mean": float(np.mean(perm)),
            "Permutation_Std": float(np.std(perm, ddof=1)),
            "Permutation_p": p_value,
            "N_Permutations": len(perm),
            "Block_Size": block_size,
            "CV_Splits": len(splits),
            "Permute_Mode": permute_mode
        })

        for val in perm:
            long_rows.append({
                "Band": band_name,
                "Classifier": clf_name,
                "Permuted_Accuracy": float(val),
                "Block_Size": block_size,
                "Permute_Mode": permute_mode
            })


# ---------------------------
# Save CSVs
# ---------------------------

perm_summary_df = pd.DataFrame(summary_rows)
perm_long_df = pd.DataFrame(long_rows)

print(perm_summary_df.head())

# Update your paths as needed
summary_save_path = "/Volumes/PRESTIGE/Slice - Rat/manuscript_BMES/classification_results/permutation_summary_052125_1.csv"
long_save_path    = "/Volumes/PRESTIGE/Slice - Rat/manuscript_BMES/classification_results/permutation_distributions_052125_1.csv"

perm_summary_df.to_csv(summary_save_path, index=False)
perm_long_df.to_csv(long_save_path, index=False)

print(f"Saved:\n  {summary_save_path}\n  {long_save_path}")
