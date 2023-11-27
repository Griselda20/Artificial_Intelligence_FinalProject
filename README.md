# Network Traffic Adroid Malware Classification Using Linear Regression Method

Pada proyek mata kuliah Kecerdasan Buatan kami ditugaskan untuk membuat sebuah model Machine Learning. Kelompok kami menerapkan paradigma ```Supervised Learning``` dengan tipe ```Classification``` yang menggunakan ``` Linear Regression Method```.

Berikut Anggota Kelompok 4 :
1. 12S21046 - Ruth Marelisa Hutagalung
2. 12S21048 - Nessy Pentasonia Pangaribuan
3. 12S21050 - Jessica Wastry Sitorus
4. 12S21052 - Griselda
5. 12S21054 - Diah Anastasya Sihombing

## Tahap 1
Pada tahap 1, kelompok kami mempersiapkan dataset yang akan digunakan. Dataset yang digunakan diambil dari kaggle <br />
https://www.kaggle.com/datasets/xwolf12/network-traffic-android-malware 

### Penjelasan terkait dataset
Dataset berisi mengenai data traffic android. Informasi yang diberikan melibatkan jumlah paket yang dikirim dan diterima, jumlah byte yang dikirim dan diterima, serta label klasifikasi. Dataset ini berguna untuk berbagai analisis data, seperti pola traffic android.

## Tahap 2

### Penjelasan Code
#### Data Loading and Preprocessing

```import pandas as pd``` => Mengimpor Pustaka Padas <br /> 
```data = pd.read_csv('android_traffic.csv', sep=';')``` => Membaca file CSV, dan menentukan separator ';' <br />
```data.shape``` => Membaca jumlah baris dan kolom <br />
```data.fillna(0, inplace=True)``` <br />
```data.head()```

![Screenshot 2023-11-27 094519](https://github.com/Griselda20/Malware-Classification-using-Linear-Regression-Method/assets/89493421/8ab1bd02-ae6b-4e01-be83-d2f20eb1bfdb)





``` ``` <br />
``` ``` <br />
