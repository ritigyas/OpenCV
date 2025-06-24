import pickle
import pickle
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array([sample for sample in data_dict['data'] if len(sample) == 42])
labels = np.array([label for i, label in enumerate(data_dict['labels']) if len(data_dict['data'][i]) == 42])

print(f"\n✅ Loaded {len(data)} valid samples with 42 features each")


class_data = defaultdict(list)
for i, label in enumerate(labels):
    class_data[label].append(data[i])

min_count = min(len(samples) for samples in class_data.values())

balanced_data = []
balanced_labels = []
for label, samples in class_data.items():
    resampled = resample(samples, n_samples=min_count, random_state=42)
    balanced_data.extend(resampled)
    balanced_labels.extend([label] * min_count)

balanced_data = np.asarray(balanced_data)
balanced_labels = np.asarray(balanced_labels)

print(f"\n⚖️ Balanced all classes to {min_count} samples each")


x_train, x_test, y_train, y_test = train_test_split(
    balanced_data, balanced_labels,
    test_size=0.2,
    shuffle=True,
    stratify=balanced_labels,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))


cv_scores = cross_val_score(model, balanced_data, balanced_labels, cv=5)
print(f"\n🔁 Cross-validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("\n💾 Model saved to 'model.p'")