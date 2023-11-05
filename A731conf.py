from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
actual_labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
predicted_labels = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
cm = confusion_matrix(actual_labels, predicted_labels)
sns.set(font_scale=1.4)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.title("Confusion Matrix", fontsize=18)
class_labels = ['Negative', 'Positive']
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks + 0.5, class_labels, fontsize=14)
plt.yticks(tick_marks + 0.5, class_labels, fontsize=14)
for i in range(2):
    for j in range(2):
        plt.text(j + 0.5, i + 0.5, str(cm[i, j]), size=16, ha='center', va='center', color='black')
plt.show()
