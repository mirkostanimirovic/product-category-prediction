import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# 1. Učitavanje sirovih podataka
df = pd.read_csv("data/products.csv")

df.columns = df.columns.str.strip()

print("Podaci su uspešno učitani.")
print("Broj redova:", len(df))


# 2. Čišćenje podataka
df_clean = df.dropna(subset=["Product Title", "Category Label"]).copy()

print("Broj redova nakon čišćenja:", len(df_clean))


# 3. Standardizacija naziva kategorija
category_mapping = {
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones"
}

df_clean["Category Label"] = df_clean["Category Label"].replace(category_mapping)


# 4. Ulazni podaci i ciljna promenljiva
X = df_clean["Product Title"]
y = df_clean["Category Label"]


# 5. Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. TF-IDF vektorizacija
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)


# 7. Treniranje finalnog modela
model_svc = LinearSVC()

model_svc.fit(X_train_tfidf, y_train)


# 8. Čuvanje modela i vektorizatora
joblib.dump(model_svc, "data/product_category_model.pkl")
joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

print("Finalni model je uspešno istreniran.")
print("Model i TF-IDF vektorizator su sačuvani.")