import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🏛️", layout="wide")

st.title("🏛️ Conversational Hybrid RAG: Gurjara-Pratihara Assistant")
st.caption("Dense (ChromaDB) + Sparse (BM25) + Cross-Encoder Reranker + Conversational Memory")

# Cache RAG pipeline
@st.cache_resource(show_spinner="Loading Hybrid Engine & Models...")
def get_pipeline():
    from ingestion import run_conversational_hybrid_rag
    return run_conversational_hybrid_rag

run_rag = get_pipeline()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Chat Input Box
user_query = st.chat_input("Ask a question about Gurjara-Pratihara history...")

if user_query:
    # 1. User message show karo
    with st.chat_message("user"):
        st.write(user_query)

    # 2. Response generate karo
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Retrieving relevant passages..."):
            answer, sources, rephrased_query = run_rag(
                user_input=user_query, 
                chat_history=st.session_state.messages, 
                top_k_rerank=5
            )
            
            # Show rephrased query if memory kicked in
            if len(st.session_state.messages) > 0 and rephrased_query != user_query:
                st.caption(f"🔍 *Search query resolved to:* `{rephrased_query}`")
                
            st.write(answer)
            
            # Reranked Sources in Expander
            with st.expander("🎯 View Top-5 Reranked Passages & Scores"):
                for rank, (doc, score) in enumerate(sources, 1):
                    st.markdown(f"**Rank {rank}** (Score: `{score:.4f}`)")
                    st.write(doc.page_content)
                    st.divider()

    # 3. Save to history
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.session_state.messages.append(AIMessage(content=answer))