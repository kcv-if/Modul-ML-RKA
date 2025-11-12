# Deployment

## Daftar Isi

-   [Daftar Isi](#daftar-isi)
-   [Deployment](#deployment)
-   [Deployment Options](#deployment-options)
-   [Contoh](#contoh)
-   [Referensi](#referensi)

## Deployment

Setelah melatih model, langkah selanjutnya adalah menentukan **metode deployment** yang sesuai dengan _use case_ model agar dapat diakses dengan mudah oleh sistem atau orang lain. Perlu diperhatikan bahwa **pemilihan metode deployment sangat berpengaruh terhadap biaya (cost)** dan **kinerja jangka panjang (scalability & latency)**. Kesalahan dalam memilih metode bisa membuat biaya operasional menjadi tinggi atau performa model tidak optimal.

## Deployment Options

Berikut adalah beberapa metode deployment yang umum digunakan saat ini:

| Metode                  | Keterangan                                                                                                                                      | Kelebihan & Kekurangan                                                                                                                                                                                                            | Use Case                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Cloud-Based**         | Model dihosting dan dijalankan pada server dalam virtual network (cloud) yang dikelola oleh pihak ketiga seperti AWS, Google Cloud, atau Azure. | <ul><li>Scaling on demand (Scalable)</li><li>Low latency</li><li>Mudah diimplementasikan</li><li>Recurrent fees meskipun model jarang digunakan</li><li>Vendor lock-in pada managed service (SageMaker, Vertex AI, dll)</li></ul> | Mid-sized project dengan penggunaan model konsisten                                |
| **On-Premise**          | Model dihosting dan dijalankan pada physical server milik sendiri.                                                                              | <ul><li>Infrastruktur sangat fleksibel</li><li>Tidak ada biaya langganan, tetapi butuh investasi awal</li><li>Kompleks untuk diimplementasi</li></ul>                                                                             | Large-scale project, enterprise, atau korporasi                                    |
| **Edge Deployment**     | Model dijalankan langsung pada edge device seperti smartphone, Raspberry Pi, atau IoT device.                                                   | <ul><li>Prediksi real-time dan low latency</li><li>Data pengguna tetap lokal</li><li>Implementasi sederhana</li><li>Tidak cocok untuk model besar atau kompleks</li></ul>                                                         | Model sederhana dan lightweight seperti image classification atau sensor inference |
| **Serverless Function** | Model dihosting pada container dan hanya dijalankan saat ada request.                                                                           | <ul><li>Cost-effective, hanya membayar saat digunakan</li><li>Minim manajemen infrastruktur</li><li>Hemat sumber daya</li><li>Cold start delay jika function tidur terlalu lama</li><li>Batas waktu eksekusi</li></ul>            | Aplikasi kecil, prototype, atau model dengan traffic rendah                        |

## Contoh

Berikut adalah cara men-deploy model klasifikasi TensorFlow kita di Streamlit HuggingFace Space. Model dapat diakses di output dari [notebook ini](https://www.kaggle.com/code/danieladhitthana/hands-on-cnn-club-dev-ai-bem-fteic-tensorflow).

1. Buat Space baru.

2. Tentukan nama Space, lisensi, dan pilih SDK Docker > Streamlit.

3. Clone repository Space ke local.

    ```bash
    git clone https://huggingface.co/spaces/{username}/{name}
    ```

    > Jika diminta melakukan autentikasi, gunakan token di HuggingFace.

4. Pada root directory, tambahkan model kalian.

5. (Opsional) Setup virtual environment agar dependency yang akan digunakan tidak mempengaruhi local environment

-   Windows

    ```bash
    python -m venv .venv
    venv\Scripts\activate
    ```

    > Jika gagal, jalankan Powershell dengan Run As Administrator lalu perbolehkan eksekusi script dengan menjalankan command berikut

    ```ps1
    Set-ExecutionPolicy RemoteSigned
    ```

-   Linux / macOS
    ```bash
    source .venv/bin/activate
    ```

6. Install semua package yang dibutuhkan dan catat package yang telah terinstalasi ke `requirements.txt`

    ```bash
    pip install streamlit tensorflow pillow
    pip freeze > requirements.txt
    ```

7. Tambahkan tampilan menggunakann Streamlit pada `app.py`

    Import package dan setup halaman:

    ```python
    # Import necessary libraries
    import streamlit as st              # Streamlit for web app
    import tensorflow as tf             # TensorFlow for the model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Image processing
    import numpy as np                  # Numerical operations
    from PIL import Image              # Image handling
    import io                          # Input/output operations

    # Configure the Streamlit page
    st.set_page_config(
        page_title="Digit Classifier",  # Browser tab title
        layout="centered"                             # Center the content
    )
    ```

    Load model ke cache.

    ```python
    @st.cache_resource  # Cache the model to avoid reloading every time
    def load_model():
        return tf.keras.models.load_model('model.h5')  # Load your saved model
    ```

    Lakukan preprocessing gambar terlebih dahulu. Pastikan sesuai dengan implementasi model.

    ```python
    def preprocess_image(img: Image.Image):
        img = img.convert('L')              # Convert to grayscale
        img = img.resize((28, 28))          # Resize to 28x28
        img = img_to_array(img)             # Convert to numpy array
        img = np.expand_dims(img, axis=0)   # Expand batch dimension
        return img
    ```

    Definisikan fungsi `main`

    ```python
    def main():
        # Add title and description
        st.title("Digit Classifier")
        st.write("Upload an image and the model will predict the digit")

        # Create file uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:  # If an image was uploaded
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Create predict button
            if st.button('Predict'):
                # Load the model (cached)
                model = load_model()

                # Preprocess the image
                processed_image = preprocess_image(image)

                # Make prediction with loading spinner
                with st.spinner('Predicting...'):
                    prediction = model.predict(processed_image)              # Get model prediction
                    pred_class = np.argmax(prediction)         # Get predicted class
                    confidence = float(prediction.max()) * 100              # Calculate confidence

                # Show results
                st.success(f'Prediction: {pred_class}')            # Show predicted class
                st.info(f'Confidence: {confidence:.2f}%')                  # Show confidence

                # Show probability bars for each class
                st.write("Class Probabilities:")
                for i, prob in enumerate(prediction[0]):
                    st.progress(float(prob))                               # Show probability bar
                    st.write(f"{i}: {float(prob)*100:.2f}%") # Show probability text
    ```

    Lalu tambahkan entry point

    ```python
    if __name__ == "__main__":
        main() # Run the main function when script is executed
    ```

    Full Code

    ```python
    import streamlit as st
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    from PIL import Image
    import io

    st.set_page_config(
        page_title="Digit Classifier",
        layout="centered"
    )

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('model.h5')

    def preprocess_image(img):
        img = img.resize((32, 32))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        img = (img - 0.5) / 0.5
        return img

    def main():
        st.title("Digit Classifier")
        st.write("Upload an image and the model will predict the digit")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Predict'):
                model = load_model()

                processed_image = preprocess_image(image)

                with st.spinner('Predicting...'):
                    prediction = model.predict(processed_image)
                    pred_class = np.argmax(prediction)
                    confidence = float(prediction.max()) * 100

                st.success(f'Prediction: {pred_class.upper()}')
                st.info(f'Confidence: {confidence:.2f}%')

                st.write("Class Probabilities:")
                for i, prob in enumerate(prediction[0]):
                    st.progress(float(prob))
                    st.write(f"{i}: {float(prob)*100:.2f}%")

    if __name__ == "__main__":
        main()
    ```

8. Jalankan app

    ```bash
    streamlit run src/streamlit_app.py
    ```

9. Jika sudah puas dengan tampilannya, push perubahannya (selain .venv jika ada) ke repository HuggingFace Space.

-   CLI

    ```bash
    git add .
    git commit -m "add files"
    git push
    ```

    > Tambahkan `.gitignore` untuk menghindari `.venv` dalam commit. `.gitignore` yang umum digunakan dapat diakses [disini](https://github.com/github/gitignore/blob/main/Python.gitignore)

-   Manual

10. Setelah build selesia, HuggingFace Space yang telah dibuat dapat diakses di `https://huggingface.co/spaces/{username}/{nama}`.

    > Jika gagal untuk mengakses lewat URL tersebut, pastikan repository kalian bersifat public

## Referensi

-   [Streamlit Documentation](https://docs.streamlit.io/)
-   [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
-   [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)
