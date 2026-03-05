import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Load data
test_df = pd.read_csv('/home/amin/PycharmProjects/Uveitis/split_2/test.csv')
probabilities_df = pd.read_csv('test_probabilities.csv')

# Merge the two dataframes on their index
data = pd.concat([test_df, probabilities_df], axis=1)

# Ignore specified categories
#ignore_set = {}
#ignore_set = {'moderate', 'severe'}
ignore_set = {'mild', 'severe'}
#ignore_set = {'moderate', 'mild'}

# Plotting histograms for Prob_Positive distribution by Label
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'orange']  # Different colors for each label
labels = data['Label'].unique()

# Ensure unique colors and labels match
for label, color in zip(labels, colors):
    subset = data[data['Label'] == label]
    plt.hist(subset['Prob_Positive'], bins=10, alpha=0.15, label=f'{label}', density=True, color=color, edgecolor='black')

plt.title('Positive Probability Distribution by Labels')
plt.xlabel('Positive Probability')
plt.ylabel('Density')
plt.xlim(left=0,right=1)
plt.ylim(bottom=0)  # Set the bottom of y-axis to zero
plt.legend(title='Label Categories')
plt.grid(True)
plt.tight_layout()
plt.savefig('prob_density_histograms.png')
#plt.show()

data = data[~data['Label'].isin(ignore_set)]

# Prepare data for ROC and PR curves
# labels = pd.Categorical(data['Label']).codes
prob_positive = data['Prob_Positive'].values
binary_labels = data['True_Label'].values
# binary_labels = label_binarize(labels, classes=range(len(data['Label'].unique())))

# Plot ROC and PR curves
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# ROC Curve
fpr, tpr, _ = roc_curve(binary_labels.ravel(), prob_positive)
roc_auc = auc(fpr, tpr)
ax[0].plot(fpr, tpr, label=f'ROC Curve (Area Under Curve = {roc_auc:.2f})', linewidth=10)
ax[0].plot([0, 1], [0, 1], 'k--', label=f'No-Skill Classifier', linewidth=10)
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Receiver Operating Characteristic Curve')
ax[0].legend(loc="lower right")
ax[0].grid()

positive_cases = data['True_Label'].sum()
total_cases = len(data['True_Label'])
baseline_precision = positive_cases / total_cases

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(binary_labels.ravel(), prob_positive)
pr_auc = auc(recall, precision)
ax[1].plot(recall, precision, label=f'PR Curve (Area Under Curve = {pr_auc:.2f})', linewidth=10)
ax[1].axhline(y=baseline_precision, xmin=0, xmax=1, color = 'k', linestyle = '--', label=f'No-Skill Classifier', linewidth=10)
# ax[1].plot([0, baseline_precision], [1, baseline_precision], 'k--', label=f'No-Skill Classifier', linewidth=10)
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].set_title('Precision-Recall Curve')
ax[1].legend(loc="upper right")
ax[1].grid()

plt.tight_layout()
#if len(
#path = 'ROC_PR_curves'
plt.savefig('ROC_PR_curves'+((', '+str(ignore_set)[1:-1]+' removed') if len(ignore_set)!=0 else '')+'.png')
#plt.show()
