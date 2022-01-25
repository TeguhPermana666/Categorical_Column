import pandas as pd
from sklearn.model_selection import train_test_split
#read data
file_path="Intermediate_ml\melb_data.csv"
model_data=pd.read_csv(file_path)
#sperate target prediction dengan fitur
y=model_data.Price
X=model_data.drop(['Price'],axis=1)
# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
## Jatuhkan kolom dengan nilai yang hilang (pendekatan paling sederhana)
cols_with_missing=[col for col in X_train_full.columns
                   if X_train_full[col].isnull().any()]
print(cols_with_missing)
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)
#inplace default = false, inplace digunakan untuk overwrithing
print(X_train_full)
print(X_valid_full)
"""
# "Kardinalitas" berarti jumlah nilai unik dalam kolom
# Pilih kolom kategoris dengan kardinalitas yang relatif rendah (nyaman tetapi sewenang-wenang)
"""
low_cardinality_cols=[cname for cname in X_train_full.columns
                      if X_train_full[cname].nunique()<10 and X_train_full[cname].dtypes=='object']
print(low_cardinality_cols)
#select numerical columns
numerical_columns=[cname for cname in X_train_full.columns
                   if X_train_full[cname].dtypes in ['int64','float64']]
print(numerical_columns)
# Simpan hanya kolom yang dipilih,->kardinalitas terendah berdasarkan unique variabel pada column yang berjenis object dan ->columnsyg bertipekan int dan float
my_cols=low_cardinality_cols+numerical_columns
X_train=X_train_full[my_cols]
X_valid=X_valid_full[my_cols]
print(X_train)
print(X_valid)
X_train=X_train_full[my_cols].copy()
X_valid=X_valid_full[my_cols].copy()
print(X_train)
print(X_valid)
#ternyata ga ada bedanya anjir
"""
Selanjutnya, kami memperoleh daftar semua variabel kategori dalam data pelatihan.
Kami melakukan ini dengan memeriksa tipe data (atau dtype) dari setiap kolom. Objek dtype menunjukkan kolom memiliki teks 
(ada hal lain yang secara teoritis bisa, tapi itu tidak penting untuk tujuan kita). Untuk kumpulan data ini, kolom dengan teks menunjukkan variabel kategori.
"""
# Get list of categorical variables
s=(X_train.dtypes=='object')
print(s)
object_cols=list(s[s].index)
print(object_cols)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train,X_valid,y_train,y_valid):
    model=RandomForestRegressor(n_estimators=10,random_state=0)
    model=model.fit(X_train,y_train)
    predicts=model.predict(X_valid)
    return mean_absolute_error(y_valid, predicts)
#1)methode drop categorical
drop_X_train=X_train.select_dtypes(exclude=['object'])
drop_X_valid=X_valid.select_dtypes(exclude=['object'])
mae=score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
print(mae)#183550.22137772635

#2)methode ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
#->membuat copyan data
label_X_train=X_train.copy()
label_X_valid=X_valid.copy()

#->apply ordinal encoder to each column with categorical data
ordinal_encoder=OrdinalEncoder()
label_X_train[object_cols]=ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols]=ordinal_encoder.transform(X_valid[object_cols])
print(label_X_train)
print(label_X_valid)
print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))#175062.2967599411

#3)methode one-hot encoding
"""
Kami menggunakan kelas OneHotEncoder 
dari scikit-learn untuk mendapatkan penyandian satu-panas. yang dapat digunakan untuk menyesuaikan perilakunya.
=>Kami menyetel handle_unknown='ignore' untuk menghindari kesalahan saat data validasi berisi kelas yang tidak direpresentasikan dalam data pelatihan, dan
=>setting sparse=False memastikan bahwa kolom yang dikodekan dikembalikan sebagai array numpy (bukan matriks sparse)
"""
from sklearn.preprocessing import OneHotEncoder
oh_encoding=OneHotEncoder(handle_unknown='ignore',sparse=False)
oh_X_train=pd.DataFrame(oh_encoding.fit_transform(X_train[object_cols]))
oh_X_valid=pd.DataFrame(oh_encoding.transform(X_valid[object_cols]))
print(oh_X_train)
print(oh_X_valid)
# One-hot encoding removed index; put it back
oh_X_train.index=X_train.index
oh_X_valid.index=X_valid.index

print(oh_X_train)
print(oh_X_valid)
# Remove categorical columns (will replace with one-hot encoding)
num_X_train=X_train.drop(object_cols,axis=1)
num_X_valid=X_valid.drop(object_cols,axis=1)
# Add one-hot encoded columns to numerical features
oh_X_train=pd.concat([num_X_train,oh_X_train],axis=1)
# oh_X_train=oh_X_train.rename(columns={0:"One-hot encoding"})

oh_X_valid=pd.concat([num_X_valid,oh_X_valid],axis=1)
# oh_X_valid=oh_X_train.rename(columns={0:"One-hot encoding"})
print(oh_X_train)
print(oh_X_valid)
print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(oh_X_train, oh_X_valid, y_train, y_valid))