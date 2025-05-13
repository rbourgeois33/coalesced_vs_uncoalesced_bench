import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# === Parse command line arguments ===
parser = argparse.ArgumentParser(description='Plot GPU bandwidth ratios from a data file.')
parser.add_argument('filename', type=str, help='Input data file with GPU bandwidth measurements')
args = parser.parse_args()

filename = args.filename
basename = os.path.splitext(os.path.basename(filename))[0]
output_image = f"{basename}_ratios.pdf"

# === Read and parse data ===
data = []
coalesced_values = []

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or line.startswith("Total") or not line:
            continue
        parts = line.split()
        if len(parts) == 5:
            row = [float(x) for x in parts]
            coalesced_values.append(row[2])
            data.append(row)

if not data:
    print(f"No valid data found in {filename}")
    exit(1)

data = np.array(data)

# === Extract columns ===
indirection = data[:, 1]
coalesced = data[:, 2]
unco_read = data[:, 3]
unco_write = data[:, 4]

# === Compute ratios ===
read_write_ratio = unco_write / unco_read
read_coalesced_ratio = unco_read / coalesced
write_coalesced_ratio = unco_write / coalesced

# Estimate coalesced BW (mean or first value)
avg_coalesced_bw = np.mean(coalesced)
coalesced_bw_label = f", Coalesced BW: {avg_coalesced_bw:.1f} GB/s"

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(indirection, read_write_ratio, 'o-', label='Uncoalesced Write / Read')
plt.plot(indirection, read_coalesced_ratio, 's--', label='Uncoalesced Read / Coalesced')
plt.plot(indirection, write_coalesced_ratio, 'd-.', label='Uncoalesced Write / Coalesced')

plt.xlabel('Indirection size (log2)')
plt.ylabel('Bandwidth Ratio')
plt.title(f'GPU: {basename.replace("_", " ")}'+coalesced_bw_label)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(output_image, dpi=400)
plt.show()

print(f"Plot saved as: {output_image}")