# Product Category Prediction

## Opis projekta

Cilj projekta je automatska klasifikacija proizvoda u odgovarajuće kategorije na osnovu naziva proizvoda.

U projektu su korišćeni tekstualni podaci iz skupa `products.csv`. Podaci su analizirani i očišćeni, nakon čega su testirana dva modela mašinskog učenja.

Kao konačni model izabran je LinearSVC, koji je ostvario tačnost od približno 96.67% na test skupu.

## Struktura projekta

- `data/products.csv` – originalni skup podataka
- `data/products_cleaned.csv` – očišćeni podaci
- `data/product_category_model.pkl` – sačuvani LinearSVC model
- `data/tfidf_vectorizer.pkl` – sačuvani TF-IDF vektorizator
- `notebooks/product_category_prediction.ipynb` – analiza podataka, čišćenje, treniranje i evaluacija modela
- `train_model.py` – treniranje i čuvanje modela
- `predict_category.py` – interaktivno predviđanje kategorije proizvoda

## Korišćene tehnologije

- Python
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook

## Pokretanje projekta

### 1. Instalacija biblioteka

U terminalu pokrenuti:

```bash
pip install pandas scikit-learn joblib jupyter
