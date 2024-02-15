import streamlit as st
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2

from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage.transform import resize

## Tab bar
st.set_page_config(
  page_title = "Lung Disease",
  page_icon = ":lung:"
)

df = pd.read_csv('dataset\dfClean.csv',delimiter=";")
# st.dataframe(df)
X = df.drop("Label",axis=1)
y = df['Label']

## Preprocessing
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X,y)

scaler = MinMaxScaler()
X_smote_norm = scaler.fit_transform(X_smote)

X_train_normal, X_temp_normal, y_train_normal, y_temp_normal = train_test_split(X_smote_norm, y_smote, test_size=0.2, random_state=42,stratify=y_smote)

##Load Model
# model = pickle.load(open("model/knn_model.pkl","rb"))
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_normal, y_train_normal)

# y_pred = model.predict(X_smote_norm)
# y_pred_knn = knn_model.predict(X_test_normal)
# accuracy = accuracy_score(y_test_normal,y_pred_knn)
# accuracy = accuracy_score(y_smote,y_pred)
# accuracy = round((accuracy * 100),2)

## Main Page
st.title(":red[Lung Disease] Classification")
st.write(f"Class: {len(np.unique(y))}")
st.write("Class[0] = :green[Normal]")
st.write("Class[1] = :red[Pneumonia] || Class[2] = :red[Derrame Pleural] ")
st.write("Class[3] = :red[Pneumotorax] || Class[4] = :red[Bronkitis]")
st.write("Class[5] = :red[Tuberculose] ||  Class[6] = :red[Abscesses]")
st.write("Class[7] = :red[Pericarditis] ||  Class[8] = :red[Atelectasis]")

# st.write(f'Accuracy = ',accuracy,'%')

uploaded_file = st.file_uploader("Pilih gambar X-ray dan prediksi:", type=["jpg", "jpeg", "png"])

## Fungsi GLCM
def extract_glcm_features(image):
    
    if image is None:
        print("Error: Input image is None.")
        return []

    if not isinstance(image, np.ndarray):
        print("Error: Input image should be a NumPy array.")
        return []
    
    image = (image * 255).astype(np.uint8)  # Mengonversi citra float menjadi uint8

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Derajat sudut yang diinginkan
    glcm_features = []

    for angle in angles:
        glcm = greycomatrix(image, [1], [angle], 256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        glcm_features.extend([contrast, dissimilarity, homogeneity, energy, correlation])

    return glcm_features

if uploaded_file is not None:
    # Baca gambar
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Periksa jumlah saluran warna
    num_channels = image.shape[-1] if len(image.shape) == 3 else 1

    if num_channels > 1:
        st.subheader("Error: Gambar berwarna tidak dapat diidentifikasi sebagai X-rays.")
    else:
        # Ekstraksi fitur GLCM
        glcm_features = extract_glcm_features(image)

        if len(glcm_features) > 0:
            # Menyusun DataFrame dengan nama kolom
            glcm_df = pd.DataFrame([glcm_features], columns=[
                'contrast_0', 'dissimilarity_0', 'homogeneity_0', 'energy_0', 'correlation_0',
                'contrast_45', 'dissimilarity_45', 'homogeneity_45', 'energy_45', 'correlation_45',
                'contrast_90', 'dissimilarity_90', 'homogeneity_90', 'energy_90', 'correlation_90',
                'contrast_135', 'dissimilarity_135', 'homogeneity_135', 'energy_135', 'correlation_135'
            ])

            glcm_df = np.reshape(glcm_df, (1, -1))
            glcm_df = scaler.transform(glcm_df)
            prediction = model.predict(glcm_df)[0]

            # Pemetaan prediksi ke label hasil
            if prediction == 0:
                result = "Normal"
            elif prediction == 1:
                result = "Pneumonia"
            elif prediction == 2:
                result = "Derrame Pleural"
            elif prediction == 3:
                result = 'Pneumotorax'
            elif prediction == 4:
                result = 'Bronkitis'
            elif prediction == 5:
                result = 'Tuberculose'
            elif prediction == 6:
                result = 'Abscesses'
            elif prediction == 7:
                result = 'Pericarditis'
            elif prediction == 8:
                result = 'Atelectasis'
            else:
                result = 'Bukan Penyakit Paru-paru'

            st.write("Ekstraksi GLCM")
            st.dataframe(glcm_df)
            st.subheader("Prediksi:")
            st.subheader(result)
        else:
            st.subheader("Error: Fitur GLCM tidak dapat diekstraksi dari gambar yang diunggah.")
else:
    st.subheader("Silakan unggah gambar X-rays terlebih dahulu.")
