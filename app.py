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
dragon_age_theme_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=IM+Fell+English&display=swap');

    /* Base App Styling - Targeting Streamlit's main block container */
    .stApp {
        background-color: #2a201b; /* Very dark, slightly warm brown */
        background-image: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.5)),
                          url("https://www.transparenttextures.com/patterns/old-wall.png"); /* Subtle texture */
        color: #e0dacd; /* Aged parchment text */
        font-family: 'IM Fell English', serif;
    }

    /* Sidebar Styling */
    .st-emotion-cache-16txtl3 { /* This selector targets the sidebar; may change with Streamlit versions */
        background-color: #1e1a16 !important; /* Even darker brown for sidebar */
        background-image: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.6)),
                          url("https://www.transparenttextures.com/patterns/worn-dots.png"); /* Different subtle texture */
    }
    .st-emotion-cache-16txtl3 h1, 
    .st-emotion-cache-16txtl3 h2, 
    .st-emotion-cache-16txtl3 h3,
    .st-emotion-cache-16txtl3 .stMarkdown p { /* Targeting text within sidebar */
        color: #c0a062 !important; /* Muted gold for sidebar text/headers */
        font-family: 'Cinzel', serif;
    }
    .st-emotion-cache-16txtl3 .stAlert, .st-emotion-cache-16txtl3 .stInfo { /* Sidebar info/warning boxes */
        background-color: rgba(60, 50, 40, 0.7) !important;
        border: 1px solid #c0a062;
        color: #e0dacd !important;
    }


    /* Main Content Headers */
    h1, h2, h3 {
        color: #8c1c1c; /* Deep Chantry red for main headers */
        font-family: 'Cinzel', serif;
        border-bottom: 1px solid #c0a062;
        padding-bottom: 0.3em;
    }
    .stCaption {
        color: #b0a090 !important; /* Lighter, muted color for captions */
    }

    /* Chat Input Area */
    .stChatInputContainer { /* Target the container of the chat input */
        background-color: #1e1a16; /* Darker base for input area */
        border-top: 2px solid #c0a062 !important;
    }
    .st-emotion-cache-13y9j5f /* Actual input field, may change */ {
        background-color: #3b3129 !important;
        color: #e0dacd !important;
        border: 1px solid #c0a062 !important;
        border-radius: 5px !important;
    }
    .st-emotion-cache-13y9j5f::placeholder {
        color: #b0a090 !important;
    }
    .stButton>button { /* Send button */
        background-color: #8c1c1c !important;
        color: #e0dacd !important;
        border: 1px solid #c0a062 !important;
        border-radius: 5px !important;
    }
    .stButton>button:hover {
        background-color: #a82b2b !important;
        border-color: #d4af37 !important;
    }


    /* Chat Messages */
    div[data-testid="chat-message-container"] {
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-width: 1px;
        border-style: solid;
    }

    /* User chat messages */
    div[data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent"][aria-label="User message"]) {
        background-color: #4a3f35; /* User chat BG */
        border-color: #c0a062;
    }

    /* Assistant chat messages */
    div[data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent"][aria-label="Assistant message"]) {
        background-color: #5a4a3a; /* Assistant chat BG - slightly lighter/warmer */
        border-color: #8c1c1c;
    }
    
    /* Styling for the avatar icons in chat messages */
    .stChatMessage .st-emotion-cache- NNNN { /* You'll need to inspect to get the exact class for the avatar circle */
        /* background-color: #c0a062 !important; */ /* Example: Gold background for avatar circle */
        /* color: #1e1a16 !important; */ /* Dark text for emoji in avatar */
    }

    /* Spinner text color */
    .stSpinner > div > div {
        color: #c0a062 !important; /* Gold for spinner text */
    }

    /* Markdown links */
    a, a:visited {
        color: #d4af37 !important; /* Brighter gold for links */
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #e7c888 !important;
    }

</style>
"""
# --- Streamlit App UI ---
st.set_page_config(page_title="Brother Genitivi's Dragon Age Archives", layout="wide", initial_sidebar_state="expanded")
st.markdown(dragon_age_theme_css, unsafe_allow_html=True)
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
    st.header("About this Archive")
    st.info(PROJECT_ABOUT_TEXT)
    st.markdown("---")
    st.header("Disclaimer")
    st.warning(DISCLAIMER_TEXT)
    st.markdown("---")
    st.caption(f"Powered by LangChain & Google Gemini.")

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
        with st.chat_message("user"):
            st.markdown(user_query)

        # Display Brother Genitivi's thinking message and get response
        with st.chat_message("assistant"):
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

