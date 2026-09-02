import joblib


# Učitavanje modela i TF-IDF vektorizatora
model = joblib.load("data/product_category_model.pkl")
vectorizer = joblib.load("data/tfidf_vectorizer.pkl")


print("Model je uspešno učitan.")
print("Unesite naziv proizvoda za predikciju.")
print("Za izlazak unesite: exit")


while True:
    proizvod = input("\nNaziv proizvoda: ")

    if proizvod.lower() == "exit":
        print("Kraj programa.")
        break

    proizvod_tfidf = vectorizer.transform([proizvod])

    kategorija = model.predict(proizvod_tfidf)[0]

    print("Predviđena kategorija:", kategorija)