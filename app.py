import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = "best_rice_disease_model.keras"

IMG_SIZE = (224, 224)


# ============================================================
# NAMA KELAS
# ============================================================

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]


# ============================================================
# INFORMASI PENYAKIT
# ============================================================

DISEASE_INFO = {

    "Bacterial Leaf Blight": {
        "description": "Penyakit hawar daun bakteri pada tanaman padi.",
        "recommendation": "Gunakan benih sehat, jaga sanitasi lahan, dan hindari penggunaan nitrogen berlebihan."
    },

    "Brown Spot": {
        "description": "Penyakit bercak coklat yang ditandai dengan munculnya bercak berwarna coklat pada daun.",
        "recommendation": "Gunakan benih sehat, perbaiki kondisi nutrisi tanah, dan lakukan pengelolaan tanaman dengan baik."
    },

    "Healthy Rice Leaf": {
        "description": "Daun padi terdeteksi dalam kondisi sehat.",
        "recommendation": "Pertahankan pemupukan, pengairan, dan pemantauan tanaman secara rutin."
    },

    "Leaf Blast": {
        "description": "Penyakit blas pada padi yang dapat menyebabkan kerusakan pada daun.",
        "recommendation": "Gunakan varietas tahan, atur pemupukan nitrogen, dan lakukan pengendalian penyakit sesuai kebutuhan."
    },

    "Leaf scald": {
        "description": "Penyakit leaf scald yang menyebabkan kerusakan dan perubahan warna pada permukaan daun.",
        "recommendation": "Jaga sanitasi lahan dan lakukan pemantauan tanaman secara berkala."
    },

    "Narrow Brown Leaf Spot": {
        "description": "Penyakit dengan gejala bercak coklat sempit dan memanjang pada daun.",
        "recommendation": "Gunakan benih sehat dan perhatikan keseimbangan nutrisi tanaman."
    },

    "Rice Hispa": {
        "description": "Kerusakan daun yang berkaitan dengan serangan hama rice hispa.",
        "recommendation": "Lakukan pemantauan populasi hama dan pengendalian sesuai tingkat serangan."
    },

    "Sheath Blight": {
        "description": "Penyakit hawar pelepah yang menyerang bagian pelepah tanaman padi.",
        "recommendation": "Atur jarak tanam, hindari kelembapan berlebihan, dan lakukan pengendalian sesuai kebutuhan."
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# PREDIKSI
# ============================================================

def predict_image(model, image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = predictions[
        predicted_index
    ] * 100

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RICH - Rice Check",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌾 RICH - Rice Check"
)

st.subheader(
    "Rice Leaf Disease Detection System"
)

st.write(
    "Sistem deteksi penyakit daun padi "
    "berbasis Deep Learning."
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Model tidak dapat dimuat."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header(
    "📷 Upload Gambar Daun Padi"
)

uploaded_file = st.file_uploader(
    "Pilih gambar daun padi",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    col1, col2 = st.columns(
        2
    )

    # --------------------------------------------------------
    # GAMBAR
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Gambar Input"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDIKSI
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Hasil Deteksi"
        )

        if st.button(
            "🔍 DETEKSI PENYAKIT",
            use_container_width=True
        ):

            with st.spinner(
                "Menganalisis gambar..."
            ):

                predicted_class, confidence, predictions = predict_image(
                    model,
                    image
                )


            # ================================================
            # HASIL
            # ================================================

            st.success(
                "Deteksi selesai!"
            )

            st.metric(
                "Hasil Prediksi",
                predicted_class
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # ================================================
            # INFORMASI PENYAKIT
            # ================================================

            if predicted_class in DISEASE_INFO:

                info = DISEASE_INFO[
                    predicted_class
                ]

                st.info(
                    f"**Informasi:**\n\n"
                    f"{info['description']}"
                )

                st.warning(
                    f"**Rekomendasi:**\n\n"
                    f"{info['recommendation']}"
                )


            # ================================================
            # PROBABILITAS
            # ================================================

            st.subheader(
                "📊 Probabilitas Setiap Kelas"
            )

            results = []

            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                probability = (
                    predictions[i] * 100
                )

                results.append(
                    (
                        class_name,
                        probability
                    )
                )


            results.sort(
                key=lambda x: x[1],
                reverse=True
            )


            for class_name, probability in results:

                st.write(
                    f"**{class_name}** "
                    f"— {probability:.2f}%"
                )

                st.progress(
                    float(
                        probability / 100
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RICH - Rice Check | "
    "Rice Leaf Disease Detection"
)