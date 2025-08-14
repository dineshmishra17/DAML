from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input
text1 = input("Enter something: ")
text2 = input("Enter something: ")

# Convert the texts into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Calculate cosine similarity
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
percentage = similarity_score * 100
print(f"Similarity: {percentage:.2f}%")

# Find matching words (case-insensitive)
words1 = set(text1.lower().split())
words2 = set(text2.lower().split())
matching_words = words1.intersection(words2)

# Show matching words
if matching_words:
    print("Matching words:", ", ".join(matching_words))
else:
    print("No exact word matches found.")
