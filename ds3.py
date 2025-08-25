import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Generate Synthetic Dataset
# -------------------------------
np.random.seed(42)
n_samples = 500

# Demographic features
age = np.random.randint(18, 60, n_samples)                # Age between 18-60
income = np.random.randint(20000, 120000, n_samples)      # Annual Income
gender = np.random.choice(["Male", "Female"], n_samples)  # Gender

# Behavioral features
browsing_time = np.random.randint(1, 60, n_samples)       # Time spent on site (mins)
clicked_ads = np.random.randint(0, 10, n_samples)         # Ads clicked

# Target: Whether purchased (based on simple rule + randomness)
purchased = (
    (income > 50000).astype(int) | 
    (browsing_time > 30).astype(int)
)  # Higher income or more browsing = more likely purchase
purchased = np.where(np.random.rand(n_samples) > 0.8, 1 - purchased, purchased)  # add noise

# Create DataFrame
data = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Gender": gender,
    "BrowsingTime": browsing_time,
    "ClickedAds": clicked_ads,
    "Purchased": purchased
})

print("Sample Data:\n", data.head())

# -------------------------------
# Step 2: Preprocess Data
# -------------------------------
data = pd.get_dummies(data, drop_first=True)  # Convert Gender to numeric

X = data.drop("Purchased", axis=1)
y = data["Purchased"]

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Decision Tree
# -------------------------------
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate Model
# -------------------------------
y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 6: Visualize Tree
# -------------------------------
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=["No Purchase", "Purchase"],
          filled=True, rounded=True)
plt.show()
