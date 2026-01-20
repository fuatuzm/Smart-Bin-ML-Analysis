import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns 

# veri okunmasi 
df = pd.read_csv('Smart_Bin.csv')

# pivot tablosu
pivot_tablo = df.pivot_table(index='Container Type', 
                               columns='Recyclable fraction', 
                               values='FL_B', 
                               aggfunc='mean')
print("Ortalama Doluluk Oranları (FL_B):")
print(pivot_tablo)
print("-" * 30)

sectigim_sutunlar = ['FL_B', 'VS', 'Container Type', 'Recyclable fraction', 'Class']
df_model = df[sectigim_sutunlar].dropna().copy()

# verileri sayiya cevirioruz
le = LabelEncoder()
df_model['Container Type'] = le.fit_transform(df_model['Container Type'])
df_model['Recyclable fraction'] = le.fit_transform(df_model['Recyclable fraction'])
df_model['Class'] = le.fit_transform(df_model['Class'])

X = df_model.drop('Class', axis=1)
y = df_model['Class']

# Eğitim ve testt kısmı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model eğitimi

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))

# sonucları yazdırma
print(f"Decision Tree       : {dt_acc:.4f}")
print(f"Logistic Regression : {lr_acc:.4f}")
print(f"Random Forest       : {rf_acc:.4f}")
print(f"KNN                 : {knn_acc:.4f}")

model_isimleri = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'KNN']
basari_oranlari = [dt_acc, lr_acc, rf_acc, knn_acc]
#grafik
plt.figure(figsize=(10, 6))
barlar = plt.bar(model_isimleri, basari_oranlari, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])

plt.ylabel('Doğruluk Oranı (Accuracy)')
plt.title('Modellerin Başarı Karşılaştırması')
plt.ylim(0, 1.0) 

# Barların üzerine değerleri yazma
for bar in barlar:
    yukseklik = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yukseklik + 0.01, f'{yukseklik:.3f}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')


onem_degerleri = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=onem_degerleri, y=onem_degerleri.index, palette='viridis')

plt.title('Hangi Özellik Kararda Daha Etkili')
plt.xlabel('Önem Puanı')
plt.ylabel('Özellikler')

for i, v in enumerate(onem_degerleri):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')


plt.tight_layout()
plt.show()