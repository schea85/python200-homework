from dotenv import load_dotenv
import os
from pathlib import Path
import string
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from pypdf import PdfReader



# === Step 1 - Setup ===

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("successfully connected to api")
else:
    print("could not connect")
    
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
docs_dir = Path("assignments_06/groundwork_docs/")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# === Step 2 - Load the Documents ===

docs = SimpleDirectoryReader(docs_dir).load_data()

print("\nNumber of Documents Loaded:", len(docs))

print("\nFile names:")
for doc in docs:
    print(doc.metadata["file_name"])
    
# === Step 3 - Build the Index and Query Engine ===

# build index
index = VectorStoreIndex.from_documents(docs)

if index:
    print("Index built successfully. Ready to answer questions.")
else:
    print("Please try again.")

# build query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# === Step 4 - Query the Assistant ===

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")
    
    for node in response.source_nodes:
        print(f"Similarity Score: {node.score:.4f}")
        print("-" * 30)