# 🔬 Technical Methodology

A comprehensive guide to the algorithms, techniques, and design decisions used in this Content-Based Movie Recommendation System.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [NLP Techniques](#nlp-techniques)
5. [Vectorization Methods](#vectorization-methods)
6. [Similarity Metrics](#similarity-metrics)
7. [Recommendation Algorithm](#recommendation-algorithm)
8. [Optimization Decisions](#optimization-decisions)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MOVIE RECOMMENDATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐ │
│  │   Data   │──▶│    Feature   │──▶│    Text    │──▶│   Vector   │ │
│  │  Loading │   │  Extraction  │   │ Processing │   │   Space    │ │
│  └──────────┘   └──────────────┘   └────────────┘   └────────────┘ │
│                                                            │        │
│                                                            ▼        │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────────┐ │
│  │  Output  │◀──│  Similarity  │◀──│     Cosine Similarity      │ │
│  │  Top-N   │   │   Ranking    │   │         Matrix             │ │
│  └──────────┘   └──────────────┘   └────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Preprocessing Pipeline

### 2.1 Raw Data Structure

The TMDB dataset contains 24 columns with various data types:

| Category           | Columns                                               | Preprocessing Needed   |
| ------------------ | ----------------------------------------------------- | ---------------------- |
| **Identifiers**    | id, imdb_id                                           | None                   |
| **Numeric**        | popularity, budget, revenue, vote_count, vote_average | Not used in this model |
| **Text**           | original_title, overview, tagline                     | Tokenization, Stemming |
| **Pipe-separated** | cast, genres, keywords, production_companies          | Split on '\|'          |
| **Date**           | release_date                                          | Not used               |

### 2.2 Feature Selection Rationale

Selected features that describe **content characteristics**:

```python
selected_columns = ['id', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'director']
```

| Feature    | Weight in Similarity | Justification                 |
| ---------- | -------------------- | ----------------------------- |
| `overview` | High                 | Contains plot details, themes |
| `genres`   | High                 | Primary categorization        |
| `keywords` | Medium               | Specific themes and topics    |
| `cast`     | Medium               | Actor preferences matter      |
| `director` | Medium               | Directorial style influences  |

**Excluded features:**

- `popularity`, `vote_average` - Popularity ≠ Similarity
- `budget`, `revenue` - Financial metrics not content-related
- `production_companies` - Not a strong content indicator

---

## 3. Feature Engineering

### 3.1 Pipe-to-List Conversion

**Problem**: Data stored as `"Action|Adventure|Thriller"`

**Solution**:

```python
def convert_pipe_to_list(text):
    if isinstance(text, str):
        return text.split('|')
    return []
```

### 3.2 Identity Preservation (Critical Algorithm)

**Problem**: "Chris Pratt" would match "Chris Evans" on the word "Chris"

**Solution**: Remove spaces in multi-word entities

```python
def collapse_spaces(obj):
    """
    'Chris Pratt' → 'ChrisPratt'
    'Science Fiction' → 'ScienceFiction'
    """
    if isinstance(obj, list):
        return [str(i).replace(" ", "") for i in obj]
    return []
```

**Before Identity Preservation:**

```
"Jurassic World" features: ['Chris', 'Pratt', 'Bryce', 'Dallas', 'Howard']
"Guardians of the Galaxy" features: ['Chris', 'Pratt', 'Zoe', 'Saldana']
# "Chris" matches even if different actors!
```

**After Identity Preservation:**

```
"Jurassic World" features: ['ChrisPratt', 'BryceDallasHoward']
"Guardians of the Galaxy" features: ['ChrisPratt', 'ZoeSaldana']
# Only exact name matches
```

### 3.3 Cast Limiting (Top 3)

**Rationale**:

- Lead actors define a movie's appeal
- Supporting actors add noise
- Reduces feature dimensionality

```python
movies['cast'] = movies['cast'].apply(lambda x: x[:3])
```

---

## 4. NLP Techniques

### 4.1 Tokenization

**Definition**: Splitting text into individual words (tokens)

```python
def split_overview(text):
    if isinstance(text, str):
        return text.split()
    return []
```

**Example**:

```
Input:  "A theme park built on dinosaurs"
Output: ['A', 'theme', 'park', 'built', 'on', 'dinosaurs']
```

### 4.2 Stemming (Porter Stemmer)

**Definition**: Reducing words to their root/base form

**Algorithm**: Porter Stemmer (1980) - rule-based suffix stripping

```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])
```

**Examples**:

| Original    | Stemmed  |
| ----------- | -------- |
| running     | run      |
| runner      | runner   |
| adventure   | adventur |
| adventurous | adventur |
| directed    | direct   |
| directing   | direct   |
| dinosaurs   | dinosaur |

**Why Stemming over Lemmatization?**

| Aspect       | Stemming              | Lemmatization       |
| ------------ | --------------------- | ------------------- |
| Speed        | Faster                | Slower              |
| Accuracy     | Less accurate         | More accurate       |
| Dependencies | Rule-based            | Requires dictionary |
| Use case     | Information retrieval | Text understanding  |

For recommendation systems, speed and coverage matter more than linguistic accuracy.

### 4.3 Stop Word Removal

Handled by `CountVectorizer(stop_words='english')`

**Stop words removed**:

```
a, an, the, is, are, was, were, be, been, being, have, has, had,
do, does, did, will, would, could, should, may, might, must, shall,
of, at, by, for, with, about, against, between, into, through, etc.
```

---

## 5. Vectorization Methods

### 5.1 Bag-of-Words (CountVectorizer)

**Concept**: Represent text as a vector of word counts

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

**Example**:

```
Vocabulary: [action, adventur, dinosaur, knight, dark, thriller]

"The Dark Knight" tags: "dark knight gotham batman action thriller"
Vector: [1, 0, 0, 1, 1, 1]  → action=1, dark=1, knight=1, thriller=1

"Jurassic World" tags: "dinosaur theme park action adventure thriller"
Vector: [1, 1, 1, 0, 0, 1]  → action=1, adventur=1, dinosaur=1, thriller=1
```

### 5.2 Parameter Choices

| Parameter      | Value          | Reasoning                              |
| -------------- | -------------- | -------------------------------------- |
| `max_features` | 5000           | Balances vocabulary coverage vs. noise |
| `stop_words`   | 'english'      | Removes non-informative words          |
| `lowercase`    | True (default) | Case-insensitive matching              |

**Why 5000 features?**

- Dataset has 1,287 movies
- ~50-100 unique words per movie
- 5000 captures most meaningful terms
- Higher values add noise (misspellings, rare words)
- Lower values miss important distinctions

---

## 6. Similarity Metrics

### 6.1 Cosine Similarity

**Mathematical Definition**:

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Properties**:

- Range: [-1, 1] (for non-negative vectors: [0, 1])
- 1 = Identical direction (same content)
- 0 = Orthogonal (no common features)

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

### 6.2 Why Cosine over Euclidean?

| Metric             | Cosine Similarity                    | Euclidean Distance       |
| ------------------ | ------------------------------------ | ------------------------ |
| **Scale**          | Invariant (normalizes length)        | Sensitive to magnitude   |
| **For text**       | Preferred (documents vary in length) | Penalizes long documents |
| **Interpretation** | Angle between vectors                | Straight-line distance   |

**Example**:

```
Movie A tags (short): "action thriller"      → [1, 0, 1]
Movie B tags (long):  "action action thriller thriller" → [2, 0, 2]

Cosine similarity: cos(θ) = 1.0 (identical direction)
Euclidean distance: sqrt((1-2)² + (1-2)²) = sqrt(2) ≠ 0
```

---

## 7. Recommendation Algorithm

### 7.1 Similarity Matrix

```python
similarity = cosine_similarity(vectors)
# Shape: (1287, 1287)
# similarity[i][j] = similarity between movie i and movie j
```

### 7.2 Recommendation Function

```python
def recommend(movie_title, n=5):
    # 1. Find movie index
    query = movie_title.lower().strip()
    titles_series = new_df['original_title'].str.lower()
    movie_index = titles_series[titles_series == query].index[0]

    # 2. Get similarity scores for this movie
    distances = similarity[movie_index]

    # 3. Sort by similarity (descending)
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]  # Skip index 0 (the movie itself)

    # 4. Return titles
    return [new_df.iloc[i[0]].original_title for i in movies_list]
```

### 7.3 Time Complexity

| Operation                  | Complexity | Notes                  |
| -------------------------- | ---------- | ---------------------- |
| Building similarity matrix | O(n² × d)  | n=movies, d=features   |
| Single recommendation      | O(n log n) | Sorting n similarities |
| Total preprocessing        | O(n × d)   | Vectorization          |

---

## 8. Optimization Decisions

### 8.1 Trade-offs Made

| Decision                       | Pro             | Con                                 |
| ------------------------------ | --------------- | ----------------------------------- |
| Bag-of-Words over TF-IDF       | Simpler, faster | Doesn't weight rare terms           |
| Porter Stemmer over Lemmatizer | Faster          | Less linguistically accurate        |
| Top 3 cast                     | Reduces noise   | May miss important supporting roles |
| 5000 max features              | Good coverage   | May miss some rare terms            |

### 8.2 Potential Improvements

1. **TF-IDF Vectorization**: Weight rare terms higher
2. **Word Embeddings (Word2Vec)**: Capture semantic similarity
3. **Weighted Features**: Give more weight to genres than cast
4. **Hybrid Approach**: Combine with collaborative filtering

### 8.3 Memory Optimization

```python
# Current: Dense matrix
vectors = cv.fit_transform(new_df['tags']).toarray()

# Better for large datasets: Sparse matrix
vectors = cv.fit_transform(new_df['tags'])  # Returns sparse matrix
```

---

## References

1. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
2. Porter, M. F. (1980). An algorithm for suffix stripping.
3. Lops, P., De Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems.
