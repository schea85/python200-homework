from dotenv import load_dotenv
import os
import string
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pypdf import PdfReader

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# --- RAG Concepts ---

# Concepts Q1:

# Scenario A: 
# I would use the RAG approach because the legal team has a lot of PDFs that get updated every quarter.
# RAG can pull information from the new documents without having to retrain the model.

# Scenario B:
# I would use fine-tuning because they already have 3,000 examples of the writing style they want.
# This will help the model learn to write in that specific brand voice more consistently.

# Scenario C:
# I would use prompt engineering because it's only one short report.
# The report can just be included in the prompt, so there is no need for RAG or fine-tuning.

# Concepts Q2:

# A confidently wrong answer is more harmful because people are more likely to believe it and act on it.
# Majority of the time, people rely on their AI for answers without fact-checking.
# Instead, if the model says "I'm not sure," then the user knows to do their own research.
# An example of a real situation where a confident hallucination could cause harm,
# is if an AI gives incorrect medical advice with confidence/certainty, some people could take the wrong medicine
# or delay getting the proper treatment.
# The tone also matters because a confident response sounds trustworthy, even when the information 
# is incorrect.

# Concepts Q3:
steps = [
    "Extract text from source documents",
    # The text is pulled/extracted from the documents that will be used as the main source.
    
    "Split text into chunks",
    # The documents are broken into smaller chunks/pieces so they are easier to search.
    
    "Convert text chunks into embeddings",
    # Each chunk is turned into a numerical representation that captures its meaning.
    
    "Receive the user's query",
    # The user asks question for the AI to answer.
    
    "Embed the user's query",
    # The user's question is also converted into an embedding.
    
    "Retrieve the most relevant chunks",
    # The system finds the chunks that are most similar to the user's question.
    
    "Inject retrieved chunks into the prompt",
    # The relevant chunks are added to the prompt that will be sent to the LLM.
    
    "Generate a response from the LLM"
    # The LLM uses the retrieved information to answer the user's question.
]

# --- Keyword Rag ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# Keyword Q1:

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print(result)

# Comment:
# The selected document was loyalty.txt. It was selected because the keyword
# retriever found a tie between multiple documents with one matching keyword.
# Since the function sorts the tied results in reverse alphabetical order,
# loyalty.txt was chosen first. This shows a limitation of keyword retrieval because
# it does not understand the meaning of the question. An embedded-based retriever
# would do better.

# Keyword Q2:

query = "Do you have anything without caffeine?"

result_2 = simple_keyword_retrieval(query, documents, verbose=True)
print(result_2)

# Comment:
# The selected document was 'None found' because the keyword retriever could not find exact
# keyword matches between the query and the documents.
# The Keyword-retrieval RAG did not get this right because it only looks for matching words
# and does not understand espresso, lattes, cappuccinos, and cold brew are related to 
# caffeine.
# An embedding-based retrieval system would do better because it understands the semantic
# and relationships between words, even when the exact keywords are not the same.  Based on
# cosine similarity.

# Keyword Q3:

# Comment/Prediction:
# My prediction is that it will get 'None found' because none of the words in the query matches
# anything written in the loyalty.txt.

query = "How do I sign up for rewards?"

result_3 = simple_keyword_retrieval(query, documents, verbose=True)
print(result_3)

# Comment/Result:
# My prediction was correct. The retriever return 'None found' because it could not match
# the exact keywords in the query.  Again, I think an embedding-based retrieval system will
# do a better job of this.

# --- Semantic RAG Concept ---

# Semantic Q1:

# a.) Vector embedding are numerical representation of a piece of text.  They help the model
# understand the meaning of the text so it can compare ideas instead of only looking for the
# exact keyword matches.

# b.) The text chunk with a cosine similarity score of 0.85 is more relevant because
# it is much closer in meaning to the query than the chunk with a score of 0.30.
# The higher the cosine similarity score, the more semantically related the two pieces
# of text are.

# c.) Semantic search can find a relevant chunk even when none of the exact words
# appear because the text is converted into numerical representations called embedding.
# The embedding captures its meaning which allows the model to compare meanings instead
# of only matching exact keywords.

# Semantic Q2:

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | vector embeddings          |
# | What is retrieved?         | Full document                     | text chunks                |
# | Can it handle synonyms?    | No                                | yes                        |
# | Storage format             | Plain text dictionary             | vector store/index         |
# | Relevance score            | Number of overlapping keywords    | cosine similarity score    |

# --- LlamaIndex ---

#set models
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("/Users/schea85/python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs/").load_data()

# build a vector index - handles chunks + embedding
index = VectorStoreIndex.from_documents(docs)

# LlamaIndex Q1:

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Comment:



# LlamaIndex Q2:
# LlamaIndex Q3:
# LlamaIndex Q4: