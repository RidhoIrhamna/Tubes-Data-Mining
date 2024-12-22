
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
df = pd.read_csv('BC.csv')

# Preprocessing
duplicate = df.duplicated().sum()
null_values = df.isnull().sum()

# Split data
y = df['diagnosis']
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Logistic Regression
param_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]}
log_reg_tuned = LogisticRegression()
grid_search = GridSearchCV(estimator=log_reg_tuned, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

# Identify highly correlated features
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop highly correlated features
X_train_dropped = X_train.drop(columns=to_drop)
X_test_dropped = X_test.drop(columns=to_drop)

# Add constant to the new dataset
X_const_tuned = sm.add_constant(X_train_dropped)
logit_model = sm.Logit(y_train, X_const_tuned)
result = logit_model.fit()

conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
class_report = classification_report(y_test, y_pred_tuned, output_dict=True)
y_pred_prob = result.predict(X_const_tuned )

# K-Means Clustering
aggregated_df = df.groupby('id').agg(
    radius_mean = ('radius_mean', 'mean'),
    texture_mean = ('texture_mean', 'mean'),
    perimeter_mean = ('perimeter_mean', 'mean'),
    area_mean = ('area_mean', 'mean'),
    smoothness_mean = ('smoothness_mean', 'mean'),
    compactness_mean = ('compactness_mean', 'mean'),
    concavity_mean = ('concavity_mean', 'mean'),
    concave_points_mean = ('concave_points_mean', 'mean'),
    symmetry_mean = ('symmetry_mean', 'mean'),
    fractal_dimension_mean = ('fractal_dimension_mean', 'mean'),
    radius_worst = ('radius_worst', 'max'),
    texture_worst = ('texture_worst', 'max'),
    perimeter_worst = ('perimeter_worst', 'max'),
    area_worst = ('area_worst', 'max'),
    smoothness_worst = ('smoothness_worst', 'max'),
    compactness_worst = ('compactness_worst', 'max'),
    concavity_worst = ('concavity_worst', 'max'),
    concave_points_worst = ('concave_points_worst', 'max'),
    symmetry_worst = ('symmetry_worst', 'max'),
    fractal_dimension_worst = ('fractal_dimension_worst', 'max')
).reset_index()

max_k = min(14, len(aggregated_df))
inertia = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(aggregated_df)
    inertia.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(aggregated_df)
aggregated_df['cluster'] = kmeans.labels_

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(aggregated_df.drop(columns=['id', 'cluster']))
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = aggregated_df['cluster'].replace({0: 'Tidak Ganas', 1: 'Ganas'})
centroids = pca_df.groupby('Cluster').mean()

# Streamlit Sidebar
st.sidebar.title("Segmentasi Analisis")
option = st.sidebar.radio("Pilih Analisis:", [
    "Informasi Dataset",
    "Preprocessing Data", 
    "Distribusi Data", 
    "Logistic Regression", 
    "K-Means Clustering", 
    "Prediksi Diagnosa"])

from io import StringIO

# Streamlit Dashboard
if option == "Informasi Dataset":
    st.title("Informasi Dataset")
    st.write("Deskripsi Dataset:")
    st.write("""
    Dataset ini berisi data mengenai diagnosis kanker payudara. 
    Setiap baris mewakili satu pasien dan setiap kolom mewakili fitur yang diukur dari sel-sel kanker.
    """)
    st.write("Jumlah data:", len(df))
    st.write("Jumlah fitur:", len(df.columns))
    st.subheader("Data Understanding")
    st.write("Menampilkan statistik deskriptif dan informasi dataset.")
    st.dataframe(df)

elif option == "Preprocessing Data":
    st.title("Data Preprocessing")
    st.write(f"Jumlah data yang duplikat: {duplicate}")
    st.write("Jumlah data null: ", null_values)

elif option == "Distribusi Data":
    st.title("Distribusi Data")
    
    # Distribusi Diagnosis
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='diagnosis', palette='Set2')
    plt.title('Distribusi Diagnosis', fontsize=16)
    plt.xlabel('Diagnosis (0: Tidak Berisiko, 1: Berisiko)', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    st.pyplot(plt)
    st.write("Distribusi Diagnosis:", df['diagnosis'].value_counts())

    # Matriks Korelasi Antar Fitur
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.iloc[:, 2:].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Korelasi'})
    plt.title('Matriks Korelasi Antar Fitur', fontsize=16)
    st.pyplot(plt)

    # Distribusi Berdasarkan Diagnosis
    plt.figure(figsize=(12, 6))
    for col in ['radius_mean', 'texture_mean', 'perimeter_mean']:  # Pilih beberapa fitur utama
        sns.kdeplot(data=df, x=col, hue='diagnosis', fill=True)
    plt.title('Distribusi Berdasarkan Diagnosis', fontsize=16)
    plt.xlabel('Nilai Fitur', fontsize=12)
    plt.ylabel('Kepadatan', fontsize=12)
    st.pyplot(plt)

elif option == "Logistic Regression":
    st.title("Logistic Regression Model")
    # st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)
    st.write("Tuned Model Accuracy:", accuracy_tuned)
    st.write("Confusion Matrix:", conf_matrix_tuned)
    
    st.write("Classification Report (After Tuning):")
    st.dataframe(class_report)
    
    # Plotting the prediction probabilities
    st.write("Prediction Probabilities:")
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_prob, bins=10, kde=True)
    plt.title('Logistic Regression Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif option == "K-Means Clustering":
    st.title("K-Means Clustering")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette={'Tidak Ganas': 'blue', 'Ganas': 'red'})
    plt.scatter(centroids['PCA1'], centroids['PCA2'], s=100, c='green', label='Centroids', marker='X')
    plt.title('K-Means Clustering Breast Cancer Results')
    plt.legend()
    st.pyplot(plt)

elif option == "Prediksi Diagnosa":
    st.title("Prediksi Diagnosa")
    st.write("Masukkan Data Pasien:")

    # Input form
    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f"{column}", value=0.0)

    # Predict button
    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        prediction = best_model.predict(input_df)[0]
        result = "Ganas" if prediction == 1 else "Tidak Ganas"
        st.write(f"Hasil Prediksi: {result}")

# Save models
with open('modelLog.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('kmeans_model.pkl', 'wb') as f_kmeans:
    pickle.dump(kmeans, f_kmeans)
