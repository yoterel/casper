import matplotlib.pyplot as plt
from pathlib import Path
import csv
import numpy as np

root_path = Path("C:/src/augmented_hands")
# user_study_file = Path(root_path, "subject1.csv")
all_latencies = []
all_motion_types = []
all_pair1s = []
all_pair2s = []
all_choices = []
for file in root_path.glob("*.csv"):
    print(file)
    with open(str(file), mode="r") as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            latency, motion_type, pair1, pair2, choice = line
            all_latencies.append(float(latency))
            all_motion_types.append(int(motion_type))
            all_pair1s.append(int(pair1))
            all_pair2s.append(int(pair2))
            all_choices.append(int(choice))
# vanila vs ours
# for pair1s = 1 (vanila), plot per latency if choice = 1 (user thinks vanilla is worse)
mask = np.array(all_pair1s) == 1
latencies = np.array(all_latencies)
choices = np.array(all_choices)
latencies = latencies[mask]
unique_latencies, count_latencies = np.unique(latencies, return_counts=True)
choices = choices[mask]
percents = []
for latency, count in zip(unique_latencies, count_latencies):
    mask = latencies == latency
    percent_vanilla = 100 * np.sum(choices[mask] == 1) / count
    percents.append(percent_vanilla)
plt.scatter(unique_latencies, percents)
plt.savefig("vanilla_vs_ours.png")
plt.cla()
plt.clf()
# GT vs vanilla
# for pair1s = 0, pairs2s = 1 (GT vs vanilla), plot per latency if choice == 1 (user thinks vanilla is worse)
mask1 = np.array(all_pair1s) == 0
mask2 = np.array(all_pair2s) == 1
mask = mask1 & mask2
latencies = np.array(all_latencies)
choices = np.array(all_choices)
latencies = latencies[mask]
unique_latencies, count_latencies = np.unique(latencies, return_counts=True)
choices = choices[mask]
percents = []
for latency, count in zip(unique_latencies, count_latencies):
    mask = latencies == latency
    percent_vanilla = 100 * np.sum(choices[mask] == 1) / count
    percents.append(percent_vanilla)
plt.scatter(unique_latencies, percents)
plt.savefig("GT_vs_vanilla.png")
plt.cla()
plt.clf()
# GT vs ours
# for pair1s = 0, pairs2s = 2 (GT vs ours), plot per latency if choice == 2 (user thinks ours is worse)
mask1 = np.array(all_pair1s) == 0
mask2 = np.array(all_pair2s) == 2
mask = mask1 & mask2
latencies = np.array(all_latencies)
choices = np.array(all_choices)
latencies = latencies[mask]
unique_latencies, count_latencies = np.unique(latencies, return_counts=True)
choices = choices[mask]
percents = []
for latency, count in zip(unique_latencies, count_latencies):
    mask = latencies == latency
    percent_vanilla = 100 * np.sum(choices[mask] == 2) / count
    percents.append(percent_vanilla)
plt.scatter(unique_latencies, percents)
plt.savefig("GT_vs_ours.png")
plt.cla()
plt.clf()
