#libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_dict = pickle.load(open('data_BSL.pickle', 'rb'))
max_length = max(len(seq) for seq in data_dict['data'])
# adding padding to dataset 
def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))  
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq  
    return padded_sequences

data = pad_sequences(data_dict['data'], max_length)
labels = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)


# Models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier()
}

results = {}
trained_models = {}


# Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }
    trained_models[name] = model

#
# Print Results
print("\nModel Evaluation Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# best model
best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
best_model = trained_models[best_model_name]
print(f"\n Best Model: {best_model_name} with Accuracy = {results[best_model_name]['Accuracy']:.4f}")


# 📊 Plot Comparison
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(metrics))  # metric positions
width = 0.2  # bar width

plt.figure(figsize=(10,6))

for i, (name, scores) in enumerate(results.items()):
    values = [scores[m] for m in metrics]
    bars = plt.bar(x + i*width, values, width, label=name)
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

plt.xticks(x + width * (len(results) - 1) / 2, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Model Comparison")
plt.legend()
plt.show()


# Save Best Model
with open("model_BSL.pkl", "wb") as f:
    pickle.dump({"model": best_model}, f)

print("\n Best model saved as 'model_BSL.pkl'")
