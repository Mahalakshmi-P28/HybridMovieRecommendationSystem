# 🎬 Hybrid Movie Recommendation System

**Built professionally by Mahalakshmi ❤️**  
[Try the live demo here](https://hybridmovierecommendationsystem-s3syq25zeyzgbpunhdthbx.streamlit.app/)  

An **interactive hybrid movie recommender** built with **Python, Streamlit, pandas, and scikit-learn**, combining **Collaborative Filtering (average ratings)** with **Content-Based Filtering (movie genres)**.

---

## Features

- **Hybrid Recommendations:** Combines content similarity and collaborative ratings.
- **Interactive Interface:** Users can input a movie, select number of recommendations, and adjust weighting (`alpha`).
- **Fast & Lightweight:** Optimized for MovieLens 100K+ dataset without freezing.
- **Professional UI:** Footer credit: “Built professionally by Mahalakshmi ❤️.”

---

## Installation

1. **Install Python 3.11+**  

2. **Clone or download this repository**  

3. **Create a virtual environment**  

```bash
python -m venv movie_env
```

4. **Activate the virtual environment**

  Windows: 
  ```bash
  movie_env\Scripts\activate
  ```
  
  Mac/Linux: 
  ```bash
  source movie_env/bin/activate
  ```

5. **Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn
```
---

## Data
Place MovieLens dataset files in a data/ folder:
```bash
  data/
  ├── u.item   # Movie information
  ├── u.data   # User ratings
```

---

## Running the App
```bash
  streamlit run app.py
```

- Enter a Movie Name (ignore year).
- Select number of recommendations and alpha.
- Click Get Recommendations.

---

## How It Works

- **Content-Based Filtering:** Computes similarity between movies using genres (cosine similarity).
- **Collaborative Filtering:** Uses average movie ratings.
- **Hybrid Recommendation:** Weighted combination using alpha.

---

## Project Structure
```bash
  movie_recommender/
  │── app.py              # Streamlit app
  │── data/               # Dataset folder
  │    ├── u.item         # Movies info
  │    ├── u.data         # Ratings info
  │── README.md           # Documentation
```

---

## Technologies Used
- Python 3.11+
- Streamlit
- pandas, numpy, scikit-learn

---

## Contact
Built professionally by Mahalakshmi
[pillamahalakshmi2004@gmail.com]
