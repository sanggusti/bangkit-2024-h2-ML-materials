import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, 
    AffinityPropagation, Birch, OPTICS
)

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore")

# Loading csv
df = pd.read_csv('https://drive.google.com/uc?id=19903lXYiKFUwB6oVn8tWFL4KsraZ0y-p')

# Features to be used for clustering
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

# One hot encoding categorical features
df_with_dummies = pd.get_dummies(df[features])

X = df_with_dummies.values

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Data preprocessing - Dividing Data for training
X_train, X_test = train_test_split(X, test_size=0.25, shuffle=False, random_state=1)

model_names = [
    'KMeans', 'AgglomerativeClustering', 'DBSCAN', 'MeanShift', 'SpectralClustering',
    'AffinityPropagation', 'Birch', 'OPTICS'
]

# Useful functions for processing all models
def model_score(model_name, save_to_csv=False):
    model = eval(model_name + '()')
    model.fit(X_train)
    
    y_train_pred = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
    y_test_pred = model.fit_predict(X_test)
    
    silhouette_avg = silhouette_score(X_test, y_test_pred)
    ari_score = adjusted_rand_score(y_train_pred, y_test_pred)
    
    if save_to_csv:
        y_submission_pred = model.fit_predict(X)
        final_df = pd.DataFrame({'CustomerID': df['CustomerID'], 'Cluster': y_submission_pred})
        final_df.to_csv(f'{model_name}.csv', index=False)
    
    return [model_name, silhouette_avg, ari_score]

models_scores = [model_score(name) for name in model_names]

models_scores_df = pd.DataFrame(models_scores, columns=['Model', 'Silhouette_Score', 'ARI_Score'])
models_scores_df.sort_values(['Silhouette_Score', 'ARI_Score'], ascending=False, inplace=True)
models_scores_df.set_index('Model', inplace=True)

print(models_scores_df)
