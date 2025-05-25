try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully overridden sqlite3 with pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary not found, using system sqlite3. This might lead to version issues with ChromaDB.")
    pass # Or handle the error more explicitly if pysqlite3-binary is critical

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- Style Configuration ---

# --- THIS MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Brother Genitivi's Dragon Age Archives",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to load local CSS file ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}. Please ensure it's in the correct path.")
    except Exception as e:
        st.error(f"An error occurred while loading CSS file {file_name}: {e}")

# Load the custom CSS for styling the sidebar and other elements
local_css("style.css")


# --- Disclaimer Text ---
DISCLAIMER_TEXT = (
    "This is a non-commercial student project created for educational purposes. "
    "The chatbot's knowledge is based on Dragon Age game guides, the copyrights for which "
    "belong to their respective owners (BioWare/EA). No copyright infringement is intended."
)
PROJECT_ABOUT_TEXT = (
    "This chatbot is a project for the DS24 Deep Learning course, demonstrating a "
    "Retrieval Augmented Generation (RAG) system with Google's Gemini models and LangChain. "
    "It answers questions based on Dragon Age game guides, in the persona of Brother Genitivi."
)

# --- Sidebar ---
with st.sidebar:
    # Using st.markdown to apply the custom class
    st.markdown(f'<div class="sidebar-themed-box">{PROJECT_ABOUT_TEXT}</div>', unsafe_allow_html=True)
    st.markdown("---") # A visual divider (optional)
    st.markdown(f'<div class="sidebar-themed-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)
    
    # You can still add other elements to the sidebar if needed
    st.caption(f"Powered by LangChain & Google Gemini.") # This will use default caption styling unless you target it

# --- URLs for avatar icons ---
bot_avatar = "icon1.png"
user_avatar = "icon2.png"


# --- Main Chat Interface ---
st.title("Brother Genitivi's Dragon Age Archives")
st.caption("Greetings, seeker of knowledge! I am Brother Genitivi. Pose your queries about the Dragon Age, and I shall consult my records.")


# --- Configuration ---
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError: # For local testing if secrets.toml doesn't exist
         print("Warning: GOOGLE_API_KEY not found in Streamlit secrets or environment variables.")

CHOSEN_EMBEDDING_MODEL = "models/text-embedding-004" 
GENERATION_MODEL_NAME = "gemini-2.0-flash"
CHROMA_PERSIST_DIRECTORY = "./chatbot_db"


@st.cache_resource # Cache the RAG chain to avoid re-initialising on every interaction
def get_rag_chain():
    """
    Initializes and returns the RAG chain.
    This function will be cached by Streamlit.
    """
    print("Attempting to initialize RAG chain components...")
    try:
        # 1. Initialize Embedding Model
        # Check for API key presence before initializing components that need it
        if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
            st.error("GOOGLE_API_KEY is not set. Please configure it in Streamlit secrets or environment variables.")
            return None
            
        embeddings_model = GoogleGenerativeAIEmbeddings(model=CHOSEN_EMBEDDING_MODEL)
        print(f"Embedding model '{CHOSEN_EMBEDDING_MODEL}' initialized.")
        print(os.path.abspath(CHROMA_PERSIST_DIRECTORY))

        # 2. Load existing ChromaDB Vector Store
        if not os.path.exists(CHROMA_PERSIST_DIRECTORY) or not os.path.isdir(CHROMA_PERSIST_DIRECTORY):
            st.error(f"ChromaDB persist directory not found or is not a directory: '{CHROMA_PERSIST_DIRECTORY}'. "
                     "Please ensure this directory exists in your GitHub repository and contains the ChromaDB files.")
            return None
        
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings_model,
            collection_name="chatbot_project"
        )
        collection_count = vector_store._collection.count() if hasattr(vector_store, '_collection') and hasattr(vector_store._collection, 'count') else "N/A (unable to get count)"
        print(f"ChromaDB loaded from {CHROMA_PERSIST_DIRECTORY}. Collection count: {collection_count}")
        if collection_count == 0 or collection_count == "N/A (unable to get count)":
             st.warning(f"Warning: ChromaDB at '{CHROMA_PERSIST_DIRECTORY}' might be empty or not loaded correctly. The chatbot may not have any knowledge.")


        # 3. Initialize LLM for Generation
        llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL_NAME, temperature=0.3)
        print(f"LLM '{GENERATION_MODEL_NAME}' initialised.")

        # 4. Create a Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
        print(f"Retriever created. Will retrieve {retriever.search_kwargs['k']} chunks.")

        # 5. Design the Prompt Template (Brother Genitivi persona based on his writing)
        template = """
### ROLE & PERSONA ###
You are Brother Ferdinand Genitivi, a renowned Chantry scholar, historian, and author from the world of Thedas.

### TONE & STYLE ###
- **Core Style:** Your tone is academic and formal, but filled with an eccentric and obsessive passion for history. 
You frame your answers as if documenting your findings for the historical record, blending the principles of a "man of science and of God".
For mundane or irrelevant questions, you may be slightly dismissive.
- **Vocabulary:** Use a rich, scholarly vocabulary (e.g., "postulate," "empirical," "fallacious," "persevere") 
and naturally incorporate Chantry-specific terms like "the Maker" and "the Chant of Light."
- **Sentence Structure:** Employ complex sentences with multiple clauses, reflecting a thoughtful and detailed writing process.
- **Rhetorical Approach:** Emphasize your role as a seeker of "truth" over superstition and dogma.
- **First-Person Narrative:** Frame your knowledge through the lens of your personal experiences and scholarly struggles.


### TASK ###
Answer the user's question based *only* on the provided context below.
- When the answer is found within the provided context, synthesize the information and present it clearly in your persona as Brother Genitivi. Speak with the authority of your research, as if recalling from your extensive studies.
- If the answer is not present in the provided context, you must clearly state this. You might say, for example:
    - "A compelling query, yet the archives I have before me are silent on this particular matter. My work in pursuit of the truth is never truly done!"
    - "Alas, the specific details you seek are not illuminated by the records currently available to me."
    - "Hmm, a point that seems to elude the present documentation. A path for future inquiry, perhaps!"
- Do not, under any circumstances, make up an answer or use knowledge from outside the provided context.

### CONTEXT ###
{context}

### QUESTION ###
{question}

### YOUR ANSWER ###
"""
        prompt_template = ChatPromptTemplate.from_template(template)
        print("Prompt template created.")

        # 6. Construct the RAG Chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt_template
            | llm
            | StrOutputParser()
        )

        chain = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)
        
        print("-*- RAG Chain constructed successfully! -*-")
        return chain

    except Exception as e:
        # Log detailed error to console for debugging when deploying
        print(f"CRITICAL ERROR initialising RAG chain: {type(e).__name__} - {e}")
        # Display a user-friendly error in the Streamlit app
        st.error(f"Failed to initialise the chatbot's knowledge core. Error: {type(e).__name__}. Please check logs or contact support.")
        return None



# Attempt to load the RAG chain
# This will use the cached version after the first successful load
rag_chain_instance = get_rag_chain()

if rag_chain_instance:
    # Initialise chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Greetings! I am Brother Genitivi. How may I assist you with your inquiries into the annals of Thedas today?"}]

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_query := st.chat_input("Ask Brother Genitivi your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(user_query)

        # Display Brother Genitivi's thinking message and get response
        with st.chat_message("assistant", avatar=bot_avatar):
            with st.spinner("Brother Genitivi is diligently consulting his extensive records and codices..."):
                try:
                    response = rag_chain_instance.invoke(user_query)
                    ai_response = response.get('answer', "My apologies, I seem to have misplaced my notes on that particular matter.")
                except Exception as e_invoke:
                    ai_response = f"Regrettably, a momentary lapse in my scholarly focus (an error occurred): {type(e_invoke).__name__}."
                    st.error(ai_response) # Display error in chat too
                    print(f"Error during RAG chain invocation: {e_invoke}") # Log to console
                
                st.markdown(ai_response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    # This message is shown if get_rag_chain() returned None (i.e., failed to initialize)
    st.error("The archives seem to be in disarray (chatbot could not be initialized). Please ensure the GOOGLE_API_KEY is correctly set in Streamlit secrets and the 'chatbot_db' directory is present and correctly populated.")

