# Network Traffic Adroid Malware Classification Using Logistic Regression Method (Binary Logistic Regression)

Pada proyek mata kuliah Kecerdasan Buatan kami ditugaskan untuk membuat sebuah model Machine Learning. Kelompok kami menerapkan paradigma ```Supervised Learning``` dengan tipe ```Classification``` yang menggunakan ``` Logistic Regression Method (Binary Logistic Regression)```.

Berikut Anggota Kelompok 4 :
1. 12S21046 - Ruth Marelisa Hutagalung
2. 12S21048 - Nessy Pentasonia Pangaribuan
3. 12S21050 - Jessica Wasty Sitorus
4. 12S21052 - Griselda
5. 12S21054 - Diah Anastasya Sihombing

## Tahap 1
Pada tahap 1, kelompok kami mempersiapkan dataset yang akan digunakan. Dataset yang digunakan diambil dari kaggle <br />
https://www.kaggle.com/datasets/xwolf12/network-traffic-android-malware 

### Penjelasan terkait dataset
Dataset berisi mengenai data traffic android. Informasi yang diberikan melibatkan jumlah paket yang dikirim dan diterima, jumlah byte yang dikirim dan diterima, serta label klasifikasi. Dataset ini berguna untuk berbagai analisis data, seperti pola traffic android.

## Tahap 2

### Penjelasan Code

#### Import libraries
```import pandas as pd```<br />
```from sklearn.preprocessing import LabelEncoder```<br />
```from sklearn.model_selection import train_test_split```<br />
```from sklearn.linear_model import LogisticRegression```<br />
```from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score```

#### Data Loading and Preprocessing
```data = pd.read_csv('android_traffic.csv', sep=';')```<br />
```data.head()```

![Screenshot 2023-11-27 094519](https://github.com/Griselda20/Malware-Classification-using-Linear-Regression-Method/assets/89493421/8ab1bd02-ae6b-4e01-be83-d2f20eb1bfdb)

Membaca file CSV 'android_traffic.csv', untuk sepator pada file CSV diindentifikasi sebagai ';'. Kemudian kami memeriksa jumlah baris dan kolom dalam dataset dan menampilkan beberapa baris dari dataset.

##### Mengganti nilai yang hilang (NaN) diganti nilai 0

```data.fillna(0, inplace=True)``` <br />
```data.head()``` <br />

![image](https://github.com/Griselda20/Malware-Classification-using-Linear-Regression-Method/assets/89493421/ad8274fc-f78a-49c5-9200-8b1f52311c81)

Karena pada dataset ada nilai yang hilang (NaN) diganti dengan nilai 0 menggunakan ```data.fillna(0, inplace=True)```. Kemudian menampilkan beberapa baris pertama.

#### Feature and Target Variable Setup
```X = data.drop('type', axis=1)```<br />
```y = data['type']```<br />

Pada matriks fitur X menghapus kolom yang berlabel 'Type' dari dataset. Kami membuat variabel target y dengan memilih kolom 'type'

#### Transformasi Data
```X = pd.get_dummies(X) ``` <br />
``` label_encoder = LabelEncoder() ``` <br />
```y = label_encoder.fit_transform(y)```

Mengubah kolom dengan format string ke format numerik

#### Train-Test Split

```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)```

#### Model Training and Prediction

```model = LogisticRegression()```<br />
```model.fit(X_train, y_train)```<br />
```y_pred = model.predict(X_test)```

#### Evaluation
```accuracy = accuracy_score(y_test, y_pred)```<br />
```precision = precision_score(y_test, y_pred)```<br />
```recall = recall_score(y_test, y_pred)```<br />
```f1 = f1_score(y_test, y_pred)```<br />
```roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])```<br />

```print(f'Accuracy: {accuracy}')```<br />
```print(f'Precision: {precision}')```<br />
```print(f'Recall: {recall}')```<br />
```print(f'F1 Score: {f1}')```<br />
```print(f'ROC AUC Score:{roc_auc}')```

## Tahap 3
### Penjelasan Visualisasi
