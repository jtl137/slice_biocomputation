#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:29:16 2026

@author: jameslim
"""

import pandas as pd
import glob
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300       # for on-screen display
mpl.rcParams['savefig.dpi'] = 300      # for saving figures
# In[]
base_dir = "/Volumes/PRESTIGE/Slice - Rat/manuscript_BMES/classification_results/classifier_performance/slices"
# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(base_dir, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

# Path to your slice CSVs
slice_files = glob.glob(os.path.join(base_dir, "*.csv"))

dfs = []

for f in csv_files:
    df = pd.read_csv(f)

    # --- extract slice ID from filename ---
    fname = os.path.basename(f)
    slice_id = fname.split("_")[0]   # e.g., "slice01"
    
    df["Slice"] = slice_id           # ADD slice column
    dfs.append(df)
    
# Read and combine
all_slices = pd.concat(dfs, ignore_index=True)

import matplotlib.pyplot as plt
import numpy as np

# Filter overall class only
df = all_slices[all_slices["Class"] == "overall"]

conditions = df["Condition"].unique()
classifiers = df["Classifier"].unique()
slices = df["Slice"].unique()

fig, ax = plt.subplots(figsize=(8,6))
colors = sns.color_palette("Set2", n_colors=len(classifiers))

width = 0.2
x = np.arange(len(conditions))

for i, clf in enumerate(classifiers):
    means = []
    sems = []

    for cond in conditions:
        subset = df[(df["Condition"] == cond) & (df["Classifier"] == clf)]
        mean_acc = subset["CV_Mean"].mean()
        sem_acc  = subset["CV_Mean"].std() / np.sqrt(len(subset))
        means.append(mean_acc)
        sems.append(sem_acc)

    ax.bar(
        x + i*width,
        means,
        width=width,
        yerr=sems,
        capsize=5,
        label=clf,
        color=colors[i]
    )

# Optional: overlay individual slice points
for j, cond in enumerate(conditions):
    for i, clf in enumerate(classifiers):
        subset = df[(df["Condition"] == cond) & (df["Classifier"] == clf)]
        ax.scatter(
            np.full(len(subset), x[j] + i*width),
            subset["CV_Mean"],
            color="k",
            s=30,
            zorder=5
        )
ax = plt.gca()  # get current axis

ax.tick_params(axis='y', labelsize=14)  # increase y-tick font size

for label in ax.get_yticklabels():
    label.set_fontweight('bold')

ax.set_xticks(x + width*(len(classifiers)-1)/2)
ax.set_xticklabels(conditions, rotation=30,fontsize=16, fontweight="bold")
ax.set_ylim(0,1)
ax.axhline(0.5, color="k", linestyle="--", label="Chance")
ax.set_ylabel("Classification Accuracy", fontsize=16, fontweight="bold")
ax.set_title("CV Accuracy Across Datasets", fontsize=25, fontweight="bold")
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()

all_slices.to_csv(
    os.path.join(base_dir, "all_classifier_performance.csv"),
    index=False
)
# In[]
"""plot average AUC"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Keep only overall AUC rows
# auc_df = df[df["Class"] == "overall"].copy()
auc_df = all_slices[all_slices["Class"] == "overall"].copy()

# Conditions to include (order matters)
conditions = ["Theta PSD", "Beta PSD", "ThetaBeta PSD"]

# Compute mean / SEM per (Classifier × Condition)
summary = (
    auc_df[auc_df["Condition"].isin(conditions)]
    .groupby(["Classifier", "Condition"])["AUC"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)

summary["sem"] = summary["std"] / np.sqrt(summary["n"])

classifiers = summary["Classifier"].unique()
n_clf = len(classifiers)
n_cond = len(conditions)

x = np.arange(n_clf)
bar_width = 0.25


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 5))
# plt.figure(figsize=(6, 4))

colors = sns.color_palette("Set2", n_colors=len(conditions))

for i, cond in enumerate(conditions):
    cond_data = summary[summary["Condition"] == cond].set_index("Classifier")

    means = cond_data.loc[classifiers, "mean"]
    sems  = cond_data.loc[classifiers, "sem"]

    plt.bar(
        x + i * bar_width,
        means,
        bar_width,
        yerr=sems,
        capsize=5,
        label=cond,
        color=colors[i]
    )
ax = plt.gca()  # get current axis

ax.tick_params(axis='y', labelsize=11)  # increase y-tick font size

for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.axhline(0.5, linestyle="--", linewidth=1)
plt.ylabel("AUC", fontsize=12,fontweight="bold")
plt.title("Average AUC Across Datasets by Feature Set and Classifier",
          fontsize=20, fontweight="bold")

plt.xticks(x + bar_width, classifiers, rotation=30, ha="right", fontsize=12,fontweight="bold")

plt.legend(title="Feature Set")
plt.tight_layout()
plt.show()
# In[]
x_labels = []
x_positions = {}
pos = 0

for clf in classifiers:
    for cond in conditions:
        label = f"{clf}\n{cond}"
        x_labels.append(label)
        x_positions[(clf, cond)] = pos
        pos += 1

plt.figure(figsize=(12, 5))

for (clf, cond), g in auc_df.groupby(["Classifier", "Condition"]):
    if cond not in conditions:
        continue

    x = x_positions[(clf, cond)]

    plt.scatter(
        np.full(len(g), x),
        g["AUC"],
        alpha=0.7
    )

plt.axhline(0.5, linestyle="--", linewidth=1)
plt.ylim(0,1)
plt.ylabel("AUC", fontweight="bold")
plt.title("Slice-wise AUC by Classifier and Feature Set", fontweight="bold")

plt.xticks(
    ticks=list(x_positions.values()),
    labels=x_labels,
    rotation=45,
    ha="right"
)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# In[]
"""compile confusion matrices"""
base_dir = "/Volumes/PRESTIGE/Slice - Rat/manuscript_BMES/classification_results/confusion_matrices/slices"

csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
dfs_confusion = []

for f in csv_files:
    df = pd.read_csv(f)
    df["Slice"] = os.path.basename(f).split("_")[0]
    dfs_confusion.append(df)

conf_cols = ["TN", "FP", "FN", "TP"]
all_confusion = pd.concat(dfs_confusion, ignore_index=True)
conf_summary = (
    all_confusion
    .groupby(["Condition", "Classifier"])[conf_cols]
    .agg(["mean", "std"])
)
all_confusion.to_csv(
    os.path.join(base_dir, "all_confusion_matrices_normalized.csv"),
    index=False
)
# In[] plot confusion matrix from selected condition as representative for figure
import numpy as np

def extract_confusion_from_row(row):
    return np.array([
        [row["TN"], row["FP"]],
        [row["FN"], row["TP"]]
    ])


import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(cm, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.2f}",
                    ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=14, fontweight="bold")
    ax.set_yticklabels(["True 0", "True 1"], fontsize=14, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=18)
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


holdout_confusion = all_confusion[
    all_confusion["Type"] == "Holdout"
]

row = all_confusion[
    (all_confusion["Slice"] == "slice02") &
    (all_confusion["Condition"] == "ThetaBeta PSD") &
    (all_confusion["Classifier"] == "SVM")
].iloc[0]

cm = extract_confusion_from_row(row)


print(cm)

plot_confusion_matrix(
    cm,
    title="Dataset03 | ThetaBeta | SVM"
)
# In[] plot mean confusion matrices across slices
import numpy as np
import matplotlib.pyplot as plt

theta_svm = all_confusion[
    # (all_confusion["Condition"] == "Theta PSD") &
    (all_confusion["Condition"] == "ThetaBeta PSD") &
    (all_confusion["Classifier"] == "SVM") &
    (all_confusion["Type"] == "Holdout")
]
mean_theta_svm = theta_svm[["TN","FP","FN","TP"]].mean()

cm_mean = np.array([
    [mean_theta_svm["TN"], mean_theta_svm["FP"]],
    [mean_theta_svm["FN"], mean_theta_svm["TP"]]
])

plot_confusion_matrix(
    cm_mean,
    # title="Mean Confusion — ThetaBeta PSD | SVM"
    title="Mean Confusion — ThetaBeta | SVM"
)

# In[]
"""compile permutation results"""
base_dir = "/Volumes/PRESTIGE/Slice - Rat/manuscript_BMES/classification_results/permutation_summaries"

csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
dfs_perm = []

for f in csv_files:
    df = pd.read_csv(f)
    df["Slice"] = os.path.basename(f).split("_")[0]
    dfs_perm.append(df)

all_perm = pd.concat(dfs_perm, ignore_index=True)

perm_summary = (
    all_perm
    .groupby(["Band", "Classifier"])
    .agg(
        mean_observed=("Observed_CV_Accuracy", "mean"),
        mean_p=("Permutation_p", "mean"),
        n_slices=("Slice", "nunique")
    )
    .reset_index()
)
all_perm.to_csv(
    os.path.join(base_dir, "all_permutation_tests.csv"),
    index=False
)