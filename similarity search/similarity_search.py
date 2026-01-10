import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from numpy._core.numeric import vecdot
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
)
vector = embeddings.embed_documents(
    [
        "David Keith Lynch (January 20, 1946 – January 16, 2025) was an American filmmaker, painter, visual artist, film editor, musician, and actor."
    ]
)
query = "who is david lynch?"
q_vector = embeddings.embed_query(query)

# COSINE SIMILARITY CAN ONLY BE CALCULATED BETWEEN TWO 2D VECTORS
# THE QUERY VECTOR WAS 1D SO WE NEEDED TO RESHAPE IT
print(cosine_similarity(vector, [q_vector]))
