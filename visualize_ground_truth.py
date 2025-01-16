import matplotlib.pyplot as plt
import numpy as np

# Ground truth comparison data from the paper
labels = ['Perplexity API', 'Human Annotators', 'Neither']
accuracy = [76, 19, 5]  # Percentage correct on disagreement cases

# Create bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracy, color=['#2ecc71', '#3498db', '#95a5a6'])

# Customize the plot
plt.title('Accuracy on Disagreement Cases\n(Based on 100 randomly sampled cases)', pad=20)
plt.ylabel('Percentage Correct (%)')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}%',
             ha='center', va='bottom')

# Adjust layout and save
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('ground_truth_comparison.png')
plt.close() 