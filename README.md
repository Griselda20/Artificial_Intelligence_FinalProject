# Network Traffic Android Malware Classification Using Logistic Regression Method (Binary Logistic Regression)

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
```import numpy as np```<br />
```import matplotlib.pyplot as plt```<br />
```from sklearn.datasets import make_classification```<br />
```from sklearn.preprocessing import LabelEncoder```<br />
```from sklearn.model_selection import train_test_split```<br />
```from sklearn.linear_model import LogisticRegression```<br />
```from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score```<br />
```from sklearn.metrics import confusion_matrix```<br />
```from sklearn.metrics import roc_curve, roc_auc_score```<br />
```from sklearn.metrics import precision_recall_curve, average_precision_score```

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

Berikut adalah hasil yang didapatkan :<br />
![image](https://github.com/Griselda20/Artificial_Intelligence_FinalProject/assets/89493421/e2bcde14-68c9-4329-9431-0a6303f3124b)

## Tahap 3
### Penjelasan Visualisasi
#### Menampilkan Confusion Matrix

Creating confusion matrix
```cm = confusion_matrix(y_test, y_pred)```<br />

Visualizing confusion matrix
```plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)```<br />
```plt.title('Confusion Matrix')```<br />
```plt.colorbar()```<br />

Adding text to the confusion matrix
```for i in range(len(cm)):```<br />
    ```for j in range(len(cm[0])):```<br />
        ```plt.text(j, i, cm[i, j], ha='center', va='center', color='red')```<br />

```classes = np.unique(y)```<br />
```tick_marks = np.arange(len(classes))```<br />
```plt.xticks(tick_marks, classes)```<br />
```plt.yticks(tick_marks, classes)```<br />

```plt.xlabel('Predicted Label')```<br />
```plt.ylabel('True Label')```<br />
```plt.tight_layout()```<br />
```plt.show()```<br />

Hasil yang diperoleh :<br />
![output](https://github.com/Griselda20/Artificial_Intelligence_FinalProject/assets/89493491/b03d4d1f-8fa8-4808-b8a2-f002c28bce2c)




#### Receiver Operating Characteristic (ROC) Curve

Menilai kemungkinan prediksi pada data uji<br />
```y_prob = model.predict_proba(X_test)[:, 1]```<br />

Menghitung nilai TPR (True Positive Rate), FPR (False Positive Rate), dan threshold<br />
```false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)```<br />

Menghitung nilai AUC (Area Under Curve)<br />
```area_under_curve = roc_auc_score(y_test, y_prob)```<br />

Plot kurva ROC (Receiver Operating Characteristic)<br />
```plt.figure(figsize=(8, 6))```<br />
```plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {area_under_curve:.2f}')```<br />
```plt.plot([0, 1], [0, 1], linestyle='--', color='grey')```<br />
```plt.xlabel('False Positive Rate')```<br />
```plt.ylabel('True Positive Rate')```<br />
```plt.title('Receiver Operating Characteristic (ROC) Curve')```<br />
```plt.legend()```<br />
```plt.show()```<br />
Hasil yang diperoleh : <br />
![image](https://github.com/Griselda20/Artificial_Intelligence_FinalProject/assets/89493421/ef94b679-7101-44ca-9a33-236a85b3fc10)

#### Precision-Recall Curve
Menghitung nilai precision, recall, dan threshold<br />
```precision, recall, thresholds = precision_recall_curve(y_test, y_prob)```<br />

Menghitung nilai Average Precision (AP)<br />
```ap = average_precision_score(y_test, y_prob)```<br />
```print(f"Average Precision (AP): {ap:.4f}")```

Hasil yang diperoleh : <br />
![image](https://github.com/Griselda20/Artificial_Intelligence_FinalProject/assets/89493421/6a955ff0-61cc-4dd7-915f-236d10c99045)

Menghitung nilai precision, recall, dan threshold<br />
```precision, recall, thresholds = precision_recall_curve(y_test, y_prob)```<br />

Menghitung nilai Average Precision (AP)<br />
```plt.figure(figsize=(8, 6))```<br />
```plt.plot(recall, precision, label=f'AP = {ap:.2f}')```<br />
```plt.xlabel('Recall')```<br />
```plt.ylabel('Precision')```<br />
```plt.title('Precision-Recall Curve')```<br />
```plt.legend()```<br />
```plt.show()```<br />

Hasil yang diperoleh : <br />
![output1](https://github.com/Griselda20/Artificial_Intelligence_FinalProject/assets/89493491/8e075772-dc46-42e8-af11-de60062a8b43)

### Tabel Efektivitas
import pandas as pd<br />
```from IPython.display import display, HTML```<br />

Create the DataFrame<br />
```data = pd.DataFrame({```<br />
```"Status": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],```<br />
```"Value": [0.6832377310388783, 0.7768595041322314, 0.2979397781299525, 0.4306987399770905, 0.6742757798059735]```<br />
```})```<br />
```df = pd.DataFrame(data, columns=['Status', 'Value'])```<br />
```display(df)```<br />
Hasil yang diperoleh : <br />
<img width="173" alt="output2" src="https://github.com/Griselda20/Artificial_Intelligence_FinalProject/assets/89493491/8a242388-5869-493f-87f3-d1679b2b39f3">






