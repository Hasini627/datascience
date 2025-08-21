# You need to have matplotlib and pandas installed:
# pip install matplotlib pandas

import pandas as pd
import matplotlib.pyplot as plt

# Example data
data = {
    'age': [23, 45, 22, 36, 27, 30, 41, 29, 34, 25],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female']
}
df = pd.DataFrame(data)

# Histogram for age distribution
plt.figure(figsize=(6,4))
plt.hist(df['age'], bins=5, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar chart for gender distribution
plt.figure(figsize=(6,4))
df['gender'].value_counts().plot(kind='bar', color=['orange', 'green'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()