

import pandas as pd

df = pd.read_csv('wine quality prediction\WineQT.csv')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
df.hist(bins=30, figsize=(20,15))
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("R2 Score: ", r2_score(y_test, predictions))
# check if any prediction is greater than or equal to 6
good_quality_pred = predictions[predictions >= 6]

if good_quality_pred.shape[0] > 0:
    good_quality_ids = X_test.loc[predictions >= 6, 'Id']
    print("IDs of wines with good quality: ", good_quality_ids.tolist())
else:
    print("No wine has good quality according to the predictions.")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Load the dataset
df = pd.read_csv('wine quality prediction/WineQT.csv')

# Separate features and target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
df_pca = pd.DataFrame(data=X_pca, columns=['residual sugar', 'citric acid', 'free sulfur dioxide'])

# Concatenate the principal components DataFrame with the target variable
df_pca['quality'] = y

# Plot a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for different wine qualities
for quality in df_pca['quality'].unique():
    subset = df_pca[df_pca['quality'] == quality]
    ax.scatter(subset['residual sugar'], subset['citric acid'], subset['free sulfur dioxide'], label=f'Quality {quality}')

ax.set_xlabel('residual sugar')
ax.set_ylabel('citric acid')
ax.set_zlabel('free sulfur dioxide')
ax.set_title('3D Scatter Plot after PCA')
ax.legend()
plt.show()
