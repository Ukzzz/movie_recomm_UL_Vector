# 🎬 Movie Recommendation System

A **Content-Based Movie Recommendation System** built using Natural Language Processing (NLP) and unsupervised machine learning techniques. The system analyzes movie metadata to find similar movies based on plot, genre, cast, and director information.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Algorithm & Methodology](#algorithm--methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Deep Dive](#technical-deep-dive)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **content-based filtering** recommendation engine that suggests movies similar to a given movie. Unlike collaborative filtering (which relies on user behavior), content-based filtering analyzes the inherent features of items themselves.

### How It Works

1. **Feature Extraction**: Extracts metadata (genres, keywords, cast, director, overview) from each movie
2. **Text Processing**: Applies NLP techniques (tokenization, stemming) to normalize text
3. **Vectorization**: Converts text features into numerical vectors using Bag-of-Words (CountVectorizer)
4. **Similarity Calculation**: Computes cosine similarity between all movie vectors
5. **Recommendation**: Returns top-N movies with highest similarity scores

---

## ✨ Features

| Feature                        | Description                                                     |
| ------------------------------ | --------------------------------------------------------------- |
| 🎭 **Content-Based Filtering** | Recommends movies based on content similarity, not user ratings |
| 📝 **NLP Processing**          | Uses Porter Stemmer for text normalization                      |
| 🔢 **Vectorization**           | Bag-of-Words with 5000 max features                             |
| 📊 **Cosine Similarity**       | Measures angular distance between movie vectors                 |
| 🎯 **Identity Preservation**   | Handles multi-word names ("Chris Pratt" → "ChrisPratt")         |

---

## 📁 Project Structure

```
Movie Prediction/
│
├── 📓 notebook/                     # Jupyter notebooks
│   └── Movie_Recomm_UL_Vector.ipynb # Main implementation notebook
│
├── 📊 data/                         # Dataset files
│   └── TMBD Movie Dataset.csv       # Source dataset (1,287 movies)
│
├── 📚 docs/                         # Documentation
│   └── METHODOLOGY.md               # Detailed algorithm explanation
│
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Python dependencies
├── 📄 LICENSE                       # MIT License
└── 📄 .gitignore                    # Git ignore rules
```

---

## 📊 Dataset

### TMDB Movie Dataset

| Property           | Value                                                          |
| ------------------ | -------------------------------------------------------------- |
| **Source**         | [Kaggle - TMDB Movie Dataset](https://www.kaggle.com/datasets) |
| **Total Records**  | 1,287 movies                                                   |
| **Total Features** | 24 columns                                                     |
| **File Size**      | ~976 KB                                                        |

### Dataset Schema

| Column                 | Type    | Description                  |
| ---------------------- | ------- | ---------------------------- |
| `id`                   | int64   | Unique movie identifier      |
| `imdb_id`              | object  | IMDB identifier              |
| `popularity`           | float64 | Popularity score             |
| `budget`               | float64 | Production budget (USD)      |
| `revenue`              | float64 | Box office revenue (USD)     |
| `original_title`       | object  | Movie title                  |
| `cast`                 | object  | Pipe-separated cast list     |
| `director`             | object  | Director name                |
| `tagline`              | object  | Movie tagline                |
| `keywords`             | object  | Pipe-separated keywords      |
| `overview`             | object  | Plot summary                 |
| `runtime`              | int64   | Duration in minutes          |
| `genres`               | object  | Pipe-separated genres        |
| `production_companies` | object  | Production companies         |
| `release_date`         | object  | Release date                 |
| `vote_count`           | int64   | Number of votes              |
| `vote_average`         | float64 | Average rating (0-10)        |
| `release_year`         | int64   | Year of release              |
| `popularity_level`     | object  | Categorical: Low/Medium/High |

### Features Used for Recommendation

The model specifically uses these 7 features:

```python
selected_columns = ['id', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'director']
```

---

## 🧠 Algorithm & Methodology

### Step-by-Step Pipeline

```mermaid
flowchart LR
    A[Raw Data] --> B[Feature Selection]
    B --> C[Text Preprocessing]
    C --> D[Vectorization]
    D --> E[Similarity Matrix]
    E --> F[Recommendations]
```

### 1. Data Loading & Feature Selection

```python
movies = pd.read_csv('data/TMBD Movie Dataset.csv')
selected_columns = ['id', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'director']
movies = movies[selected_columns]
```

### 2. Text Preprocessing

#### a) Pipe-to-List Conversion

Converts `"Action|Adventure|Thriller"` → `['Action', 'Adventure', 'Thriller']`

#### b) Identity Preservation (Critical Fix)

```python
def collapse_spaces(obj):
    """
    Converts 'Chris Pratt' -> 'ChrisPratt'
    Prevents 'Chris Pratt' matching with 'Chris Evans' solely on 'Chris'
    """
    if isinstance(obj, list):
        return [str(i).replace(" ", "") for i in obj]
    return []
```

#### c) Cast Limiting

Limits cast to top 3 actors to reduce noise from uncredited extras.

#### d) Tag Creation

Concatenates all features into a single "tags" string:

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']
```

### 3. Natural Language Processing

#### Porter Stemmer

Reduces words to their root form:

- "running" → "run"
- "adventure" → "adventur"
- "directed" → "direct"

```python
ps = PorterStemmer()
def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])
```

### 4. Vectorization (Bag-of-Words)

```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

| Parameter      | Value     | Rationale                               |
| -------------- | --------- | --------------------------------------- |
| `max_features` | 5000      | Statistical sweet spot for dataset size |
| `stop_words`   | 'english' | Removes common words (the, a, is)       |

### 5. Cosine Similarity

Measures the cosine of the angle between two vectors:

$$
\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
$$

- **1.0** = Identical movies
- **0.0** = Completely different

```python
similarity = cosine_similarity(vectors)
```

This creates a 1287 × 1287 matrix where `similarity[i][j]` represents the similarity between movie `i` and movie `j`.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install numpy pandas nltk scikit-learn
```

### Dependencies

| Library        | Version | Purpose                     |
| -------------- | ------- | --------------------------- |
| `numpy`        | ≥1.21   | Numerical operations        |
| `pandas`       | ≥1.3    | Data manipulation           |
| `nltk`         | ≥3.6    | Natural Language Processing |
| `scikit-learn` | ≥0.24   | Machine Learning algorithms |

### NLTK Data Download

```python
import nltk
nltk.download('punkt')
```

---

## 🚀 Usage

### Basic Usage

```python
# 1. Load the data and compute similarity matrix (run all notebook cells)

# 2. Define your query movie
query = 'The Dark Knight'.lower().strip()
titles_series = new_df['original_title'].str.lower()

# 3. Get recommendations
if query in titles_series.values:
    movie_index = titles_series[titles_series == query].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].original_title)
```

### Sample Output

```
✅ Recommendations for 'The Dark Knight':
========================================
• The Dark Knight Rises
• Batman Begins
• Harsh Times
• Sherlock Holmes: A Game of Shadows
• The Usual Suspects
```

### Creating a Reusable Function

```python
def recommend(movie_title, n_recommendations=5):
    """
    Returns top N movie recommendations for a given movie title.

    Args:
        movie_title (str): The title of the movie to find recommendations for
        n_recommendations (int): Number of recommendations to return

    Returns:
        list: List of recommended movie titles
    """
    query = movie_title.lower().strip()
    titles_series = new_df['original_title'].str.lower()

    if query not in titles_series.values:
        return f"Movie '{movie_title}' not found in database."

    movie_index = titles_series[titles_series == query].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n_recommendations+1]

    return [new_df.iloc[i[0]].original_title for i in movies_list]

# Usage
print(recommend('Jurassic World'))
# Output: ['Jurassic Park', 'The Lost World: Jurassic Park', ...]
```

---

## 🔬 Technical Deep Dive

### Why Content-Based Filtering?

| Approach          | Pros                                                     | Cons                                               |
| ----------------- | -------------------------------------------------------- | -------------------------------------------------- |
| **Content-Based** | No cold-start problem for items, Works without user data | Requires feature engineering, Limited diversity    |
| **Collaborative** | Discovers unexpected recommendations                     | Cold-start problem, Requires user interaction data |
| **Hybrid**        | Best of both worlds                                      | More complex to implement                          |

### Why Cosine Similarity Over Euclidean Distance?

- **Scale Invariant**: Not affected by document length
- **Bounded**: Always between -1 and 1
- **Efficient**: Fast computation for sparse vectors

### Vector Space Visualization

```
                    Movie A (Action, Adventure)
                   /
                  /  θ = small angle = high similarity
                 /
Origin ─────────●─────────────────→ Movie B (Action, Thriller)
                 \
                  \  θ = large angle = low similarity
                   \
                    Movie C (Romance, Drama)
```

### Performance Considerations

| Metric                  | Value                   |
| ----------------------- | ----------------------- |
| **Vocabulary Size**     | 5,000 features          |
| **Matrix Dimensions**   | 1,287 × 5,000 (vectors) |
| **Similarity Matrix**   | 1,287 × 1,287           |
| **Memory Usage**        | ~25 MB                  |
| **Recommendation Time** | O(n) per query          |

---

## 📈 Results

### Sample Recommendations

| Query Movie                      | Top 5 Recommendations                                                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **The Dark Knight**              | The Dark Knight Rises, Batman Begins, Harsh Times, Sherlock Holmes: A Game of Shadows, The Usual Suspects |
| **Jurassic World**               | (Action, Adventure, Science Fiction movies)                                                               |
| **Star Wars: The Force Awakens** | (Other Star Wars & Sci-Fi movies)                                                                         |

### Why These Results Make Sense

For "The Dark Knight":

1. **The Dark Knight Rises** - Same franchise, same director (Nolan)
2. **Batman Begins** - Same franchise, same director
3. **Harsh Times** - Shares cast member (Christian Bale)
4. **Sherlock Holmes** - Crime/Mystery themes
5. **The Usual Suspects** - Crime/Thriller themes

---

## 🔮 Future Improvements

### Short-Term

- [ ] Add TF-IDF vectorization (better than raw counts)
- [ ] Implement Word2Vec/Doc2Vec embeddings
- [ ] Create a Flask/FastAPI web interface
- [ ] Export model using pickle for production

### Medium-Term

- [ ] Expand dataset (5,000+ movies)
- [ ] Add user ratings for hybrid approach
- [ ] Include poster images using CNNs
- [ ] Deploy as a microservice

### Long-Term

- [ ] Real-time learning from user feedback
- [ ] Multi-language support
- [ ] Integration with streaming platforms (Netflix, Prime)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where possible

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) for the movie database
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [NLTK](https://www.nltk.org/) for NLP tools

---

## 📬 Contact

**Project Maintainer**: [Your Name]

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

<p align="center">
  Made with ❤️ for movie lovers
</p>
