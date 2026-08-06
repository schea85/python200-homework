from dotenv import load_dotenv
import os
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

# Query 1: "What employee benefits does BrightLeaf offer"
# a.) Of the three chunks, the first one is the most relevant and accurate answer. 
# The other chunks were not directly related to the employee benefits. They were about the company's
# mission and partnerships.

# b.) The AI's response/tone did sound confident and specific.

# c.) An unexpected result was that some unrelated documents were retrieved (the last two).
# This shows that similarity search can sometimes return chunks that are related to the company
# overall but not directly related to the question.
# -------------------------------------------------- #
# Query 2: "What are BrightLeaf's security policies"
# a.) The answer given was relevant because it came from the security policy document.
# The other two retrieved chunks were less relevant because they discussed employee benefits
# and partnership.

# b.) The model's response sounded confident and specific.  It gave detailed explanation of 
# BrightLeaf's security practices without using uncertain/unsure language.

# c.) Same as the previous query, the last two chunks were not relevant to the question.
# This probably happened because those two documents mentions BrightLeaf and share some
# similar language.

# LlamaIndex Q2:

question = "What employee benefits does BrightLeaf offer?"

# top 1 
query_engine_top1 = index.as_query_engine(similarity_top_k=1)

print("\nTop 1 Response:")
print(f"Q: {question}")
response_top1 = query_engine_top1.query(question)
print(f"A: {response_top1}")

for node in response_top1.source_nodes:
    print(f"Similarity Score: {node.score:.4f}")

# top 5
query_engine_top5 = index.as_query_engine(similarity_top_k=5)

print("\nTop 5 Response:")
print(f"Q: {question}")
response_top5 = query_engine_top5.query(question)
print(f"A: {response_top5}")

for node in response_top5.source_nodes:
    print(f"Similarity Score: {node.score:.4f}")

# Comment:
# Both the top1 and top5 retrievals gave the same/correct answers.
# More retrieved context is not always better. Additional context can help when the
# retrieved chunks contain useful information, but lower-quality or unrelated chunks can
# add noise and may not improve the response.
# The most relevant context is often more important than simply retrieving more chunks.

# LlamaIndex Q3:

new_question = "What is BrightLeaf's expected profit by 2030?"

new_query = index.as_query_engine(similarity_top_k=3)

print(f"\nQ: {new_question}")
new_response = new_query.query(new_question)
print(f"A: {new_response}")

for node in response.source_nodes:
    print(f"Node ID: {node.node.node_id}")
    print(f"Similarity Score: {node.score:.4f}")
    print(f"Text Snippet: {node.node.get_content()[:150]}...")
    print("-" * 30)

# Comment:
# I expected the pipeline to struggle because the documents do not contain information
# about BrightLeaf's expected profit in 2030.
# The model responded that the information was not explicitly mentioned in the provided context
# instead of making up an answer.  
# The retrieved chunks were not relevant to the question because they discussed network security,
# employee benefits, and company partnerships rather than financial projections.
# To improve the system, I would increase the quality of the document collection and adjust
# the retrieval settings so the model only uses highly relevant chunks. This would
# reduce the chance of unrelated information being retrieved.

# LlamaIndex Q4:

print("\nFaithfulness and Relevancy Evaluators:")

# create judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# get response to query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

print(f"Q: {q}")
print(f"A: {response}")

# evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print(f"Faithfulness Evaluation: {str(faithfulness_result.score)}")

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print(f"Relevancy Result: {str(relevancy_result.score)}")

# query 2
q2 = "Who is the CEO of BrightLeaf?"
q2_response = query_engine.query(q2)

print(f"\nQ: {q2}")
print(f"A: {q2_response}")

faithfulness_result2 = faithfulness_evaluator.evaluate_response(query=q2, response=q2_response)
print(f"Faithfulness Evaluation: {str(faithfulness_result2.score)}")

relevancy_result2 = relevancy_evaluator.evaluate_response(query=q2, response=q2_response)
print(f"Relevancy Result: {str(relevancy_result2.score)}")

# Comment:
# a.) Faithfulness measures whether the model's response was faithful to the retrieved contexts;
# whether it contains hallucination or lying.  It can be used to assess the generation step.
# A faithfulness score of 1.0 means that it has passed the evaluation; it does not contain
# unsupported claims.
# A score of 0.0 would indicate that the response contains information that is not supported by the
# retrieved documents; could be hallucinating and/or lying.

# b.) Relevancy measures whether the model's response is relevant to the query using the 
# retrieved contexts; whether the response is off-topic or rambling.  This metric can be used
# to assess the retrieval step.
# It is different from faithfulness because faithfulness checks whether the answer is supported by 
# the context, while relevancy checks whether the answer addresses what the user actually asked.

# c.) Yes, the scores changed between the two queries. 
# The employee benefits question received a faithfulness and relevancy scores of 1.0 because the information
# were available in the BrightLeaf documents and the response directly answered the question.
# The CEO question received a faithfulness score of 1.0 because the model correctly stated
# that the information was not available (did not hallucinate), but it received a relevancy score of 0.0 
# because it did not provide the requested CEO information.

# d.) The LLM-as-a-judge approach uses a separate LLM to evaluate the quality of RAG responses.
# This is useful because natural language answers are difficult to measure with simple
# accuracy metrics.  The judge LLM can evaluate whether a response is supported by the 
# retrieved context and whether it answers the user's question.