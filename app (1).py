
# 1. Import required libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. Page title
st.title("📚 Simple Book Recommendation App")

# 3. Load dataset
# Make sure Book_rec.csv is in the same folder as this file
df = pd.read_csv("Book_rec_small.csv")
# 4. Fill missing values to avoid errors
df['Book-Title'] = df['Book-Title'].fillna('')

# 5. Create TF-IDF matrix (based on book titles)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Book-Title'])

# 6. Compute similarity scores
similarity = cosine_similarity(tfidf_matrix)

# 7. Define recommendation function
def recommend(book_title):
    if book_title not in df['Book-Title'].values:
        return ["❌ Book not found. Please check the spelling or try another title."]
    idx = df[df['Book-Title'] == book_title].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recommended_books = [df.iloc[i[0]]['Book-Title'] for i in distances[1:6]]
    return recommended_books

# 8. Create a text input for the user
book_input = st.text_input("Enter the title of a book you liked:")

# 9. If user enters a book name, show recommendations
if book_input:
    st.subheader("📖 Recommended Books:")
    results = recommend(book_input)
    for i, title in enumerate(results, 1):
        st.write(f"{i}. {title}")
      
# 10. Done!
st.write("---")
st.caption("Made with ❤️ using Streamlit")
