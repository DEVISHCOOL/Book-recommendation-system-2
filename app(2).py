
# -------------------------------
# app.py — Simple Book Recommender (Final)
# -------------------------------

# 1. Import libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. App title
st.title("📚 Simple Book Recommendation App")

# 3. Load dataset safely
try:
    df = pd.read_csv("Book_rec_clean.csv")   # use your cleaned file here
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# 4. Check for the right column
if 'Book-Title' not in df.columns:
    df.columns = ['Book-Title']

# 5. Remove missing or empty titles
df['Book-Title'] = df['Book-Title'].fillna('')
df = df[df['Book-Title'].str.strip() != '']

# 6. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Book-Title'])

# 7. Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# 8. Define recommend() function
def recommend(book_title):
    # Case-insensitive matching
    matches = df[df['Book-Title'].str.lower() == book_title.lower()]
    if matches.empty:
        return ["❌ Book not found. Please check the spelling or try another title."]
    
    idx = matches.index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    
    recommended_books = [df.iloc[i[0]]['Book-Title'] for i in distances[1:6]]
    return recommended_books

# 9. Text input for user
book_input = st.text_input("Enter the title of a book you liked:")

# 10. Show recommendations
if book_input:
    st.subheader("📖 Recommended Books:")
    results = recommend(book_input)
    for i, title in enumerate(results, 1):
        st.write(f"{i}. {title}")

# 11. Footer
st.write("---")
st.caption("Made with ❤️ using Streamlit")
