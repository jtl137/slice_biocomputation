#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:34:21 2025

@author: jameslim

This script is for analyzing the saved extracted_segments that are already 
filtered. This extracted segments array should have
the segments of data for each stim trial'. Arrays may be filtered to visualize LFPs 
(bandpassing anywhere between 0-300Hz) after stimulation.

All LFP extracted_segments arrays are downsampled to 2000Hz before being loaded
into this script.
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

plt.rcParams['figure.dpi'] = 300       # for on-screen display
plt.rcParams['savefig.dpi'] = 300      # for saving figures
# from intanutil.header import (read_header,
#                               header_to_result)
# from intanutil.data import (calculate_data_size,
#                             read_all_data_blocks,
#                             check_end_of_file,
#                             parse_data,
#                             data_to_result)
# from intanutil.filter import apply_notch_filter

# In[]:
Fs = 2000
# Create a Tkinter root window, which is required for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select .npy file", 
    filetypes=[("NumPy files", "*.npy")], 
    initialdir="."  # Optional: set the initial directory
)

# Check if a file was selected
if file_path:
    # Load the selected .npy file
    extracted_segments1 = np.load(file_path)
    print("File loaded successfully.")
    print("Data:", extracted_segments1)
else:
    print("No file selected.")

# In[]:
#Fs = 30000
# Create a Tkinter root window, which is required for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select before tissue .npy file", 
    filetypes=[("NumPy files", "*.npy")], 
    initialdir="."  # Optional: set the initial directory
)

# Check if a file was selected
if file_path:
    # Load the selected .npy file
    extracted_segments2 = np.load(file_path)
    print("File loaded successfully.")
    print("Data:", extracted_segments2)
else:
    print("No file selected.")   

# In[] media only
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select before tissue .npy file", 
    filetypes=[("NumPy files", "*.npy")], 
    initialdir="."  # Optional: set the initial directory
)

# Check if a file was selected
if file_path:
    # Load the selected .npy file
    extracted_segments_mediaonly = np.load(file_path)
    print("File loaded successfully.")
    print("Data:", extracted_segments_mediaonly)
else:
    print("No file selected.")   
# In[] extracted_segments+ttx
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select before tissue .npy file", 
    filetypes=[("NumPy files", "*.npy")], 
    initialdir="."  # Optional: set the initial directory
)

# Check if a file was selected
if file_path:
    # Load the selected .npy file
    extracted_segments_ttx = np.load(file_path)
    print("File loaded successfully.")
    print("Data:", extracted_segments_ttx)
else:
    print("No file selected.")   

# In[ ]: FILTER DEFINITIONS

def LP_IIR(signal, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.iirfilter(order, normal_cutoff,
                                  btype='lowpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def HP_IIR(signal, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.iirfilter(order, normal_cutoff,
                                  btype='highpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def BP_IIR(signal, cutoff1, cutoff2, fs, order):
    nyq = 0.5 * fs
    normal_cutoff1 = cutoff1 / nyq
    normal_cutoff2 = cutoff2 / nyq

    b, a = scipy.signal.iirfilter(order, [normal_cutoff1, normal_cutoff2],
                                  btype='bandpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def BS_IIR(signal, cutoff1, cutoff2, fs, order):
    nyq = 0.5 * fs
    normal_cutoff1 = cutoff1 / nyq
    normal_cutoff2 = cutoff2 / nyq
    b, a = scipy.signal.iirfilter(order, [normal_cutoff1, normal_cutoff2],
                                  btype='bandstop', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def AP_IIR(signal, fs, order):
    nyq = 0.5 * fs
    cutoff1 = 300
    cutoff2 = 3000
    normal_cutoff1 = cutoff1 / nyq
    normal_cutoff2 = cutoff2 / nyq

    b, a = scipy.signal.iirfilter(order, [normal_cutoff1, normal_cutoff2],
                                  btype='bandpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def LFP_IIR(signal, fs, order):
    nyq = 0.5 * fs
    cutoff = 1000

    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.iirfilter(order, normal_cutoff,
                                  btype='lowpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter
# In[] for Figure 2
def plot_avg_time_series(time_series_slice, layout, fs, title='Average Time-Series'):
    """
    Plots the average time-series across trials for each channel in the given layout.

    Parameters:
        time_series_slice: array of shape (trials, channels, samples)
        layout: 2D list of channel names (e.g., [['A0','A1',...], ...])
        fs: sampling rate (Hz)
    """
    num_trials, num_channels, num_samples = time_series_slice.shape
    num_rows = len(layout)
    num_cols = len(layout[0])

    # Build time axis
    time_axis = np.arange(num_samples) / fs

    # Bank -> channel index offsets
    bank_offsets = {"A": 0, "B": 32, "C": 64, "D": 96}

    # Precompute average across trials
    avg_data = np.mean(time_series_slice, axis=0)  # -> (channels, samples)
    std_segment = np.std(time_series_slice, axis=0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16), sharex=True, sharey=True)

    for i, row in enumerate(layout):
        for j, channel_name in enumerate(row):

            bank = channel_name[0]
            ch_num = int(channel_name[1:])
            channel_index = bank_offsets[bank] + ch_num

            ax = axes[i, j]
            ax.plot(time_axis, avg_data[channel_index], linewidth=3.0, color='black')
            ax.fill_between(time_axis, avg_data[channel_index] - std_segment[channel_index],avg_data[channel_index] + std_segment[channel_index], 
                            alpha=0.6 , color='blue')
            ax.set_ylim(-200,200)
            # Formatting
            # ax.set_title(channel_name, fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelsize=6)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return avg_data, std_segment

layout = [
            ['A7', 'A6', 'A14', 'A20', 'B24', 'B25', 'B17', 'B11'],
            ['A0', 'A4', 'A13', 'A22', 'B31', 'B27', 'B18', 'B9'],
            ['A23', 'A17', 'A27', 'A11', 'B6', 'B30', 'B10', 'B20'],
            ['A31', 'A19', 'A12', 'A18', 'B0', 'B2', 'B19', 'B13'],
            ['A28', 'A2', 'A29', 'A15', 'B3', 'B29', 'B12', 'B16'],
            ['A5', 'A21', 'A1', 'A25', 'B26', 'B4', 'B14', 'B8'],
            ['A24', 'A3', 'A10', 'A16', 'B7', 'B28', 'B21', 'B15'],
            ['A26', 'A30', 'A8', 'A9', 'B5', 'B1', 'B23', 'B22']
        ]
avg_data1, std_segment1 = plot_avg_time_series(extracted_segments1, layout, fs=2000)
avg_data2, std_segment2 = plot_avg_time_series(extracted_segments2, layout, fs=2000)

# In[ ]: band pass extracted_segments array into desired low frequency bands

def bandpass_extracted_segments(filtered_segments,fs,order=3):
    '''
    Create filtered_segments arrays dedicated to a certain low frequency band
    Parameters:
        - array: filtered_segments array [trials, channels, samples]
        - sampling frequency: fs
        - order: filter order
        
    Returns:
        - filtered_segments_delta
        - filtered_segments_theta
        - filtered_segments_beta
        - filtered_segments_gamma
        - filtered_segments_ripple
    '''
    # Delta
    cutoff_delta = 1.5  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_delta = 4
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_delta = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_delta[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_delta, cutoff1_delta, fs, order)

    # Theta
    cutoff_theta = 4  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_theta = 12
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_theta = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_theta[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_theta, cutoff1_theta, fs, order)

    # Beta
    cutoff_beta = 12  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_beta = 30
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_beta = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_beta[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_beta, cutoff1_beta, fs, order)

    # Gamma (30-100Hz)
    cutoff_gamma = 30  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_gamma = 100
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_gamma = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_gamma[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_gamma, cutoff1_gamma, fs, order)

    # Ripple (100-250Hz)
    cutoff_ripple = 100  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_ripple = 250
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_ripple = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_ripple[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_ripple, cutoff1_ripple, fs, order)

    
    return f_segments_delta, f_segments_theta, f_segments_beta, f_segments_gamma, f_segments_ripple
# In[]
"""Use bandpass extracted segments function on both filtered segments 1 and 2"""
# Example Usage:
array1 = extracted_segments1
fs = 2000
f_segments_delta1, f_segments_theta1, f_segments_beta1, f_segments_gamma1, f_segments_ripple1 = bandpass_extracted_segments(array1,fs,order=3)
print('Function Complete!')

array2 = extracted_segments2
fs = 2000
f_segments_delta2, f_segments_theta2, f_segments_beta2, f_segments_gamma2, f_segments_ripple2 = bandpass_extracted_segments(array2,fs,order=3)
print('Function Complete!')
# In[ ]: band pass extracted_segments array into desired low frequency bands

def plot_f_segments(ch, start, stop, f_segments_delta, f_segments_theta, f_segments_beta, f_segments_gamma, f_segments_ripple):
    '''
    Plot frequency bands from f_segments_arrays
    Parameters:
        - 
    Returns:
        - time_axis
    '''
    # Create time axis
    time_axis = np.arange(f_segments_beta.shape[2])/fs
    
    # Get shape of filtered_segments
    trials, channels, samples = f_segments_beta.shape
    print('trials, channels, samples: ', f_segments_beta.shape)

    # Plot individual trial responses and display each frequency band
    for trial in range(trials):
        plt.plot(time_axis[start:stop], f_segments_delta[trial,
                  ch, start:stop], label='delta(1-4Hz)')
        plt.plot(time_axis[start:stop], f_segments_theta[trial,
                  ch, start:stop], label='theta(4-10Hz)')
        plt.plot(time_axis[start:stop], f_segments_beta[trial,
                  ch, start:stop], label='beta(12-30Hz)')
        plt.plot(time_axis[start:stop], f_segments_gamma[trial,
                  ch, start:stop], label='gamma(30-100Hz)')
        plt.plot(time_axis[start:stop], f_segments_ripple[trial,
                  ch,start:stop], label='ripple(100-250Hz)')
        # plt.plot(time_axis[start:stop], filtered_segments1[trial,
        #           ch, start:stop], label='all(0-300Hz)', alpha=0.5)
        plt.xlabel('Time(s)')
        plt.ylabel('uV')
        plt.ylim(-500, 500)
        plt.axvline(x=stim_time, color='black',
                    linestyle='--', linewidth=2, label='STIM')
        plt.title(f'Frequency bands channel {ch} - Trial {trial+1} response')
        plt.grid(True)
        plt.legend(loc='upper right', fontsize='x-small', ncol=2)
        plt.show()
    
    return time_axis

# Example Usage:
ch = 0
start = int(0*Fs)
stop = int(10*Fs)
stim_time = 2
time_axis = plot_f_segments(ch, start, stop,
        f_segments_delta1, f_segments_theta1, f_segments_beta1, 
        f_segments_gamma1, f_segments_ripple1)
print('plot_f_segments COMPLETE!')

ch = 0
start = int(0*Fs)
stop = int(10*Fs)
stim_time = 2
time_axis = plot_f_segments(ch, start, stop,
        f_segments_delta2, f_segments_theta2, f_segments_beta2, 
        f_segments_gamma2, f_segments_ripple2)
print('plot_f_segments COMPLETE!')

# In[] for figure 3
def plot_f_segments_column(
    ch, start, stop,
    f_segments_delta, f_segments_theta, f_segments_beta,
    f_segments_gamma, f_segments_ripple,
    n_trials=5, trials_to_plot=None
):
    """
    Plot frequency band responses for a single channel, stacking trials vertically in one column.
    
    Parameters:
        ch : int
            Channel index to plot.
        start, stop : int
            Start and stop sample indices.
        f_segments_* : np.ndarray
            Arrays of shape (trials, channels, samples) for each frequency band.
        n_trials : int
            Number of trials to display if trials_to_plot=None.
        trials_to_plot : list[int] or None
            Specific trial indices to display. If None, takes the first n_trials.
    """
    # Create time axis
    time_axis = np.arange(f_segments_beta.shape[2]) / fs

    # Get shape
    trials, channels, samples = f_segments_beta.shape
    print('trials, channels, samples: ', f_segments_beta.shape)

    # Determine which trials to plot
    if trials_to_plot is None:
        selected_trials = np.arange(min(n_trials, trials))
    else:
        selected_trials = np.array(trials_to_plot)

    # Create subplots
    fig, axes = plt.subplots(len(selected_trials), 1, figsize=(8, 2 * len(selected_trials)), sharex=True, sharey=True)

    if len(selected_trials) == 1:
        axes = [axes]  # make iterable if only 1 trial

    for idx, trial in enumerate(selected_trials):
        ax = axes[idx]
        ax.plot(time_axis[start:stop], f_segments_delta[trial, ch, start:stop], label='delta (1–4Hz)', alpha=0.9,linewidth=3)
        ax.plot(time_axis[start:stop], f_segments_theta[trial, ch, start:stop], label='theta (4–12Hz)', alpha=0.8,linewidth=3)
        ax.plot(time_axis[start:stop], f_segments_gamma[trial, ch, start:stop], label='gamma (30–100Hz)', alpha=0.6,linewidth=3)
        
        ax.axvline(x=stim_time, color='black', linestyle='--', linewidth=4, label='STIM')
        ax.set_ylim(-200, 200)
        ax.set_title(f'Trial {trial+1}', fontsize=15, loc='left', fontweight="bold")

        # Hide ticks except last
        if idx != len(selected_trials) - 1:
            ax.set_xticklabels([], fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for ax in axes:
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(14)
                label.set_fontweight('bold')

    # Final subplot keeps axes
    axes[-1].set_xlabel('Time (s)', fontsize=20, fontweight="bold")
    axes[-1].set_ylabel('uV', fontsize=20, fontweight="bold")
    axes[0].legend(loc='upper right', fontsize=13,ncol=2)

    plt.tight_layout()
    plt.show()
plot_f_segments_column(ch, start, stop, f_segments_delta1, f_segments_theta1, f_segments_beta1, f_segments_gamma1, f_segments_ripple1, 
                       n_trials=5,trials_to_plot=[8,34,59,77,96])
plot_f_segments_column(ch, start, stop, f_segments_delta2, f_segments_theta2, f_segments_beta2, f_segments_gamma2, f_segments_ripple2, 
                       n_trials=5,trials_to_plot=[8,34,59,77,96])

# In[ plot avg of media only, slice, slice + TTX] - for supplementary figure 3
ch=35
avg_media_only = np.mean(extracted_segments_mediaonly[:,ch,:], axis=0)
avg_extracted_segments1 = np.mean(extracted_segments1[:,ch,:], axis=0)
avg_ttx = np.mean(extracted_segments_ttx[:,ch,:], axis=0)

std_media_only = np.std(extracted_segments_mediaonly[:, ch, :], axis=0)
std_extracted_segments1 = np.std(extracted_segments1[:, ch, :], axis=0)
std_ttx = np.std(extracted_segments_ttx[:, ch, :], axis=0)

def plot_avg_segments(media_only,extracted_segments1, avg_ttx):
    
    # Build a time axis in seconds
    n_samples = avg_media_only.shape[0]
    t = np.arange(n_samples) / fs

    plt.figure(figsize=(12, 4))
    # Media only
    plt.subplot(1, 3, 1)
    plt.plot(t, avg_media_only, color="blue", label="Mean", linewidth=3)
    plt.fill_between(t, avg_media_only - std_media_only, 
                        avg_media_only + std_media_only, 
                        color="blue", alpha=0.3, label="±1 SD")
    plt.title(f"Channel {ch} – Media only", fontweight="bold")
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("µV", fontweight="bold")
    plt.ylim(-500,2000)
    # Extracted Segments 1
    plt.subplot(1, 3, 2)
    plt.plot(t, avg_extracted_segments1, color="green", label="Mean", linewidth=3)
    plt.fill_between(t, avg_extracted_segments1 - std_extracted_segments1, 
                        avg_extracted_segments1 + std_extracted_segments1, 
                        color="green", alpha=0.3, label="±1 SD")
    plt.title(f"Channel {ch} – Slice", fontweight="bold")
    # plt.xlabel("Time (s)")
    plt.ylim(-500,2000)
    # TTX
    plt.subplot(1, 3, 3)
    plt.plot(t, avg_ttx, color="red", label="Mean", linewidth=3)
    plt.fill_between(t, avg_ttx - std_ttx, 
                        avg_ttx + std_ttx, 
                        color="red", alpha=0.3, label="±1 SD")
    plt.title(f"Channel {ch} – Slice + TTX", fontweight="bold")
    # plt.xlabel("Time (s)")
    plt.ylim(-500,2000)
fs = 2000
plot_avg_segments(avg_media_only,avg_extracted_segments1,avg_ttx)
# In[ plot avg of media only, slice, slice + TTX] - for supplementary figure 3
ch=37
avg_media_only = np.mean(extracted_segments_mediaonly[:,ch,:], axis=0)
avg_extracted_segments1 = np.mean(extracted_segments1[:,ch,:], axis=0)
avg_ttx1 = np.mean(extracted_segments_notissue1[:,ch,:], axis=0)

std_media_only = np.std(extracted_segments_mediaonly[:, ch, :], axis=0)
std_extracted_segments1 = np.std(extracted_segments1[:, ch, :], axis=0)
std_ttx1 = np.std(extracted_segments_notissue1[:, ch, :], axis=0)

avg_media_only = np.mean(extracted_segments_mediaonly[:,ch,:], axis=0)
avg_extracted_segments2 = np.mean(extracted_segments2[:,ch,:], axis=0)
avg_ttx2 = np.mean(extracted_segments_notissue1[:,ch,:], axis=0)

std_media_only = np.std(extracted_segments_mediaonly[:, ch, :], axis=0)
std_extracted_segments2 = np.std(extracted_segments2[:, ch, :], axis=0)
std_ttx2 = np.std(extracted_segments_notissue2[:, ch, :], axis=0)

def plot_avg_segments(media_only,extracted_segments1, avg_ttx, std_ttx):
    
    # Build a time axis in seconds
    n_samples = avg_media_only.shape[0]
    t = np.arange(n_samples) / fs

    plt.figure(figsize=(12, 4))
    # Media only
    plt.subplot(1, 3, 1)
    plt.plot(t, avg_media_only, color="blue", label="Mean", linewidth=3)
    plt.fill_between(t, avg_media_only - std_media_only, 
                        avg_media_only + std_media_only, 
                        color="blue", alpha=0.3, label="±1 SD")
    plt.title(f"Channel {ch} – Media only", fontweight="bold")
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("µV", fontweight="bold")
    plt.ylim(-500,2000)
    # Extracted Segments 1
    plt.subplot(1, 3, 2)
    plt.plot(t, avg_extracted_segments1, color="green", label="Mean", linewidth=3)
    plt.fill_between(t, avg_extracted_segments1 - std_extracted_segments1, 
                        avg_extracted_segments1 + std_extracted_segments1, 
                        color="green", alpha=0.3, label="±1 SD")
    plt.title(f"Channel {ch} – Slice", fontweight="bold")
    # plt.xlabel("Time (s)")
    plt.ylim(-500,2000)
    # TTX
    plt.subplot(1, 3, 3)
    plt.plot(t, avg_ttx, color="red", label="Mean", linewidth=3)
    plt.fill_between(t, avg_ttx - std_ttx, 
                        avg_ttx + std_ttx, 
                        color="red", alpha=0.3, label="±1 SD")
    plt.title(f"Channel {ch} – Slice + TTX", fontweight="bold")
    # plt.xlabel("Time (s)")
    plt.ylim(-500,2000)
fs = 2000
plot_avg_segments(avg_media_only,avg_extracted_segments1,avg_ttx1,std_ttx1)
plot_avg_segments(avg_media_only,avg_extracted_segments2,avg_ttx2,std_ttx2)
