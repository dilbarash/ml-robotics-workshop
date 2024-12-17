import matplotlib.pyplot as plt
ids = [1, 2, 3, 4, 5]
heights = [170, 160, 180, 150, 165]
weights = [65, 55, 75, 50, 60]
plt.figure(figsize=(8, 6))
plt.bar([i - 0.2 for i in ids], heights, width=0.4, color='coral', edgecolor='black', hatch='/', label='Height (cm)')
plt.bar([i + 0.2 for i in ids], weights, width=0.4, color='mediumpurple', edgecolor='black', hatch='\\', label='Weight (kg)')
plt.xlabel('ID', fontsize=12, fontweight='bold')
plt.ylabel('Values', fontsize=12, fontweight='bold')
plt.title('Height and Weight of Individuals', fontsize=14, fontweight='bold', color='darkblue')
plt.xticks(ids, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
