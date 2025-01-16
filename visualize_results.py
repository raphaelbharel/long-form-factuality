import matplotlib.pyplot as plt
import numpy as np

# Data from the test results
samples = ['Sample 1', 'Sample 2', 'Sample 3']
human_supported = [4, 11, 0]
human_not_supported = [14, 10, 16]
human_irrelevant = [8, 10, 2]

perplexity_supported = [3, 3, 8]
perplexity_not_supported = [8, 7, 0]
perplexity_irrelevant = [0, 0, 3]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar width
width = 0.35

# Positions for bars
x = np.arange(len(samples))

# Plot fact distribution
ax1.bar(x - width/2, human_supported, width, label='Human - Supported', color='green', alpha=0.6)
ax1.bar(x - width/2, human_not_supported, width, bottom=human_supported, label='Human - Not Supported', color='red', alpha=0.6)
ax1.bar(x - width/2, human_irrelevant, width, bottom=np.array(human_supported) + np.array(human_not_supported), label='Human - Irrelevant', color='gray', alpha=0.6)

ax1.bar(x + width/2, perplexity_supported, width, label='Perplexity - Supported', color='green', alpha=0.3)
ax1.bar(x + width/2, perplexity_not_supported, width, bottom=perplexity_supported, label='Perplexity - Not Supported', color='red', alpha=0.3)
ax1.bar(x + width/2, perplexity_irrelevant, width, bottom=np.array(perplexity_supported) + np.array(perplexity_not_supported), label='Perplexity - Irrelevant', color='gray', alpha=0.3)

ax1.set_ylabel('Number of Facts')
ax1.set_title('Distribution of Fact Classifications')
ax1.set_xticks(x)
ax1.set_xticklabels(samples)
ax1.legend()

# Plot correlations
correlations = {
    'Supported (S)': -0.778,
    'Not Supported (NS)': -0.676,
    'Irrelevant (IR)': -0.971
}

colors = ['green', 'red', 'gray']
ax2.bar(correlations.keys(), correlations.values(), color=colors, alpha=0.6)
ax2.set_ylabel('Pearson Correlation')
ax2.set_title('Correlation between Human and Perplexity Ratings')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig('fact_checker_comparison.png')
plt.close() 