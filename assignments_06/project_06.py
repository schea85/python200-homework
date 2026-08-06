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
        print(f"File name: {node.node.metadata['file_name']}")
        print(f"Similarity Score: {node.score:.4f}")
        print(f"Text Snippet: {node.node.get_content()[:200]}...")
        print("-" * 30)
        
# Comment:
# The assistant provided confident and mostly accurate answers based on the
# Groundwork documents.  Most responses matched the retrieved context, especially
# the company story and catering questions.  It could have answered about the dairy-free
# milk options better; like providing a list of the alternatives.
# No, none of the final answers did not surprise me; it were mostly accurate.

# --- Step 5 - Find a Failure ---

question = "Who is the CEO of Groundwork Coffee?"

response2 = query_engine.query(question)

print(f"Q: {question}")
print(f"A: {response2}")

for node in response2.source_nodes:
    print(f"File name: {node.node.metadata['file_name']}")
    print(f"Similarity Score: {node.score:.4f}")
    print(f"Text Snippet: {node.node.get_content()[:200]}...")
    print("-" * 30)

# Comment:
# I asked the assistant, "Who is the CEO of Groundwork Coffee?"
# I expected this question to be difficult because the documents mention
# the founders but does not provide information about a CEO.
# The model's tone was confident. It retrieved the company story and convincingly answered with the
# founders instead of stating that the CEO information was unavailable. 
# In real life, users should always fact check instead of always trusting the AI-generated responses.
# To improve the system, I would add a prompt instruction to avoid guessing and state when 
# information is not found in the documents.

# === Step 6 - Reflection ===

# 1.) The LlamaIndex implementation only took a few lines, maybe around 10-15 lines of code to
# build this whole project. Using a framework can simplify development and reduce boilerplate code.
# Makes for a more efficient pipeline.

# 2.) An IT department could use a RAG system to answer employee questions using
# internal documentations, troubleshooting guides, and software manuals.

# 3.) One failure mode that RAG cannot fully prevent is answering confidently
# when the retrieved documents are related but do not actually contain the correct information.
# The model may still give incorrect answers instead of recognizing that the
# answer is unavailable.