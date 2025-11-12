# Deployment

## Daftar Isi
- [Daftar Isi](#daftar-isi)
- [Deployment](#deployment)
  - [Deployment Options](#deployment-options)
  - [Contoh](#contoh)
- [Referensi](#referensi)

## Deployment
Setelah melatih model, langkah selanjutnya adalah menentukan **metode deployment** yang sesuai dengan *use case* model agar dapat diakses dengan mudah oleh sistem atau orang lain. Perlu diperhatikan bahwa **pemilihan metode deployment sangat berpengaruh terhadap biaya (cost)** dan **kinerja jangka panjang (scalability & latency)**. Kesalahan dalam memilih metode bisa membuat biaya operasional menjadi tinggi atau performa model tidak optimal.

### Deployment Options
Berikut adalah beberapa metode deployment yang umum digunakan saat ini:

| Metode | Keterangan | Kelebihan & Kekurangan | Use Case |
|--------|-------------|------------------------|-----------|
| **Cloud-Based** | Model dihosting dan dijalankan pada server dalam virtual network (cloud) yang dikelola oleh pihak ketiga seperti AWS, Google Cloud, atau Azure. | <ul><li>Scaling on demand (Scalable)</li><li>Low latency</li><li>Mudah diimplementasikan</li><li>Recurrent fees meskipun model jarang digunakan</li><li>Vendor lock-in pada managed service (SageMaker, Vertex AI, dll)</li></ul> | Mid-sized project dengan penggunaan model konsisten |
| **On-Premise** | Model dihosting dan dijalankan pada physical server milik sendiri. | <ul><li>Infrastruktur sangat fleksibel</li><li>Tidak ada biaya langganan, tetapi butuh investasi awal</li><li>Kompleks untuk diimplementasi</li></ul> | Large-scale project, enterprise, atau korporasi |
| **Edge Deployment** | Model dijalankan langsung pada edge device seperti smartphone, Raspberry Pi, atau IoT device. | <ul><li>Prediksi real-time dan low latency</li><li>Data pengguna tetap lokal</li><li>Implementasi sederhana</li><li>Tidak cocok untuk model besar atau kompleks</li></ul> | Model sederhana dan lightweight seperti image classification atau sensor inference |
| **Serverless Function** | Model dihosting pada container dan hanya dijalankan saat ada request. | <ul><li>Cost-effective, hanya membayar saat digunakan</li><li>Minim manajemen infrastruktur</li><li>Hemat sumber daya</li><li>Cold start delay jika function tidur terlalu lama</li><li>Batas waktu eksekusi</li></ul> | Aplikasi kecil, prototype, atau model dengan traffic rendah |

### Contoh


## Referensi
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
* [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)
