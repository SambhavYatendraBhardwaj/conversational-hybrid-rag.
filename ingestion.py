from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
loader=PyPDFLoader("Gurjara_Pratihara_Empire.pdf")
docs=loader.load()
print(docs)
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n"]
)
chunks=splitter.split_documents(docs)
print(chunks)
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='chroma_db',
    collection_name='Gurjara_Pratihara_Empire.pdf'
)
query="who was mihir bhoja?"
results=vector_store.similarity_search(query,k=2)
print(results[0].page_content)
dense_retriever=vector_store.as_retriever(search_kwargs={"k":20})
tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
bm25 = BM25Okapi(tokenized_corpus)

def get_bm25_top_k(query: str, k: int = 20):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]
print("Loading Cross-Encoder model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search_and_rerank(query: str, top_k_rerank: int = 5):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = get_bm25_top_k(query, k=20)
    unique_docs = {}
    for doc in dense_docs + sparse_docs:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
            
    combined_docs = list(unique_docs.values())
    print(f"Total Unique Retrieved Docs before Reranking: {len(combined_docs)}")

    pairs = [[query, doc.page_content] for doc in combined_docs]
    scores = reranker.predict(pairs)
    
    # Sort docs by reranker score descending
    scored_docs = list(zip(combined_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    top_docs = [doc for doc, score in scored_docs[:top_k_rerank]]
    return top_docs, scored_docs[:top_k_rerank]

if __name__ == "__main__":
    test_query = "Who was Mihira Bhoja and what was his title?"
    print(f"\nRunning Hybrid Search for Query: '{test_query}'")
    final_docs, scored = hybrid_search_and_rerank(test_query, top_k_rerank=5)
    
    print("\n--- Top Reranked Results ---")
    for rank, (doc, score) in enumerate(scored, 1):
        print(f"\n[Rank {rank}] (Score: {score:.4f})")
        print(doc.page_content[:200] + "...")
llm=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.0
)
parser = StrOutputParser()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = PromptTemplate.from_template(
    """You are an expert historical research assistant specializing in medieval Indian history.
Answer the user's question using ONLY the provided reranked context.
If the context does not contain enough information to answer truthfully, state clearly that the document lacks the necessary details.
Keep the response factual, concise, and highlight key names, titles, and dates.

Context:
{context}

Question:
{query}

Answer:"""
)

parser = StrOutputParser()
chain = prompt | llm | parser

def run_hybrid_rag(query: str, top_k_rerank: int = 5):
    final_docs, scored = hybrid_search_and_rerank(query, top_k_rerank=top_k_rerank)
    
    # Context string join karo
    context = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
    
    # Dono variables match: "context" aur "query"
    final_result = chain.invoke({
        "context": context,
        "query": query
    })
    
    return final_result, scored

# Test run direct execution
if __name__ == "__main__":
    test_query = "Who was Nagabhata2 ?"
    print(f"\nRunning Query: {test_query}")
    answer, sources = run_hybrid_rag(test_query, top_k_rerank=5)
    
    print("\n--- FINAL ANSWER ---")
    print(answer)
