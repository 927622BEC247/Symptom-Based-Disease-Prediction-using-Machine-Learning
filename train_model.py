import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
df = pd.read_csv("training.csv")

# Step 2: Split features (X) and target (y)
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Find best k from k=3 to 15
best_k = 3
best_accuracy = 0

print("Evaluating k values...\n")

for k in range(3, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k = {k}, Accuracy = {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print(f"\nBest k = {best_k} with accuracy = {best_accuracy:.4f}")

# Step 5: Train final model with best k
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X, y)  # Train on full data for final model

# Step 6: Save the model
with open("knn_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print("Model saved as knn_model.pkl")
