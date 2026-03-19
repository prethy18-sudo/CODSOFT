import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 1. Create a small dataset of movies
data = {
    'Movie': ['Toy Story', 'The Dark Knight', 'Shrek', 'Iron Man', 'Frozen'],
    'Genre': ['Animation Kids', 'Action Crime', 'Animation Kids', 'Action Sci-Fi', 'Animation Kids']
}
df = pd.DataFrame(data)

# 2. Convert text genres into numbers (Vectors)
cv = CountVectorizer()
genre_matrix = cv.fit_transform(df['Genre'])

# 3. Calculate how similar movies are to each other
similarity = cosine_similarity(genre_matrix)

# 4. Function to recommend
def recommend(movie_name):
    index = df[df['Movie'] == movie_name].index[0]
    distances = similarity[index]
    # Get top 2 similar movies (excluding itself)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:3]
    
    print(f"Because you watched {movie_name}, you might like:")
    for i in movie_list:
        print("-", df.iloc[i[0]].Movie)

recommend('Toy Story')
