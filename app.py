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
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700&family=Sorts+Mill+Goudy&display=swap');

    /* Base App Styling */
    .stApp {
        background-color: #1A1D24; /* Very dark blue/grey */
        /* For a subtle texture, you might try a repeating very dark pattern if you host it, or a CSS gradient */
        /* background-image: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url("your_subtle_dark_texture.png"); */
        color: #D1C7B8; /* Aged parchment text */
        font-family: 'Sorts Mill Goudy', serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #121418 !important; /* Almost black */
        border-right: 2px solid #4A4F58; /* Metallic grey border */
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #D4AF37 !important; /* Gold for sidebar headers */
        font-family: 'Cinzel Decorative', cursive;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #A39B8B !important; /* Secondary text color for sidebar content */
        font-family: 'Sorts Mill Goudy', serif;
    }

    /* Custom Themed Boxes for Sidebar (Disclaimer, About) */
    .sidebar-themed-box {
        background-color: rgba(30, 33, 40, 0.7); /* Darker, slightly transparent */
        border: 1px solid #D4AF37; /* Gold border */
        color: #D1C7B8 !important; 
        padding: 1em;
        border-radius: 4px; /* Sharper edges */
        margin-bottom: 1em;
    }
    .sidebar-themed-box h3 {
        color: #D4AF37 !important;
        font-family: 'Cinzel Decorative', cursive !important;
        margin-top: 0;
        border-bottom: 1px solid #4A4F58;
        padding-bottom: 0.3em;
    }
    .sidebar-themed-box p {
        color: #D1C7B8 !important;
        font-family: 'Sorts Mill Goudy', serif !important;
        line-height: 1.6;
    }


    /* Main Content Headers */
    .main h1, .main h2, .main h3 {
        color: #D4AF37; /* Gold for main headers */
        font-family: 'Cinzel Decorative', cursive;
        border-bottom: 2px solid #4A4F58;
        padding-bottom: 0.3em;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5); /* Subtle text shadow */
    }
    .stCaption {
        color: #A39B8B !important;
    }

    /* Chat Input Area */
    .stChatInputContainer { 
        background-color: #121418; /* Match sidebar for contrast */
        border-top: 2px solid #D4AF37 !important;
    }
    div[data-testid="stChatInputTextArea"] textarea {
        background-color: #23272F !important;
        color: #D1C7B8 !important;
        border: 1px solid #4A4F58 !important;
        border-radius: 4px !important;
        font-family: 'Sorts Mill Goudy', serif !important;
    }
    div[data-testid="stChatInputTextArea"] textarea::placeholder {
        color: #A39B8B !important;
    }
    button[data-testid="stChatInputSubmitButton"] { 
        background-color: #D4AF37 !important; /* Gold button */
        color: #121418 !important; /* Dark text on gold */
        border: 1px solid #4A4F58 !important;
        border-radius: 4px !important;
    }
    button[data-testid="stChatInputSubmitButton"]:hover {
        background-color: #EACD80 !important; /* Lighter gold on hover */
        border-color: #D4AF37 !important;
    }
    button[data-testid="stChatInputSubmitButton"] svg {
        fill: #121418 !important; /* Dark icon on gold button */
    }


    /* Chat Messages */
    div[data-testid="chat-message-container"] {
        border-radius: 6px; /* Slightly sharper */
        padding: 0.85rem;
        margin-bottom: 1rem;
        border-width: 1px;
        border-style: solid;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.3); /* Add some depth */
    }

    /* User chat messages */
    div[data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent"][aria-label="User message"]) {
        background-color: #2C313A; /* Darker blue-grey */
        border-color: #4A4F58; /* Metallic grey border */
    }

    /* Assistant chat messages (Brother Genitivi) */
    div[data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent"][aria-label="Assistant message"]) {
        background-color: #23272F; /* Slightly lighter than user, but still dark */
        /* For a parchment texture within the bubble, you'd ideally use a background-image.
           This is harder to do just with color, but a slightly warmer off-black can suggest it.
           Or background-image: url('your_parchment_texture_tile.png'); */
        border-color: #D4AF37; /* Gold border for assistant */
    }
    
    div[data-testid="chat-avatar-container"] {
        background-color: #D4AF37 !important; /* Gold background for avatar circle */
        color: #1A1D24 !important; /* Dark text/emoji color for avatar */
    }

    /* Spinner text color */
    .stSpinner > div > div {
        color: #D4AF37 !important; /* Gold for spinner text */
        font-family: 'Cinzel Decorative', cursive !important;
    }

    /* Markdown links */
    a, a:visited {
        color: #EACD80 !important; /* Brighter gold for links */
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #FFF0C0 !important; /* Even lighter gold on hover */
    }
    
    /* General button styling (if you add other st.buttons) */
    .stButton>button:not([data-testid="stChatInputSubmitButton"]) { /* Exclude chat send button */
        border: 1px solid #D4AF37;
        background-color: transparent;
        color: #D4AF37;
        border-radius: 4px;
    }
    .stButton>button:not([data-testid="stChatInputSubmitButton"]):hover {
        border-color: #EACD80;
        background-color: rgba(212, 175, 55, 0.1);
        color: #EACD80;
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
    # Using st.markdown to apply the custom class
    st.markdown(f'<div class="sidebar-themed-box">{PROJECT_ABOUT_TEXT}</div>', unsafe_allow_html=True)
    st.markdown("---") # A visual divider (optional)
    st.markdown(f'<div class="sidebar-themed-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)
    
    # You can still add other elements to the sidebar if needed
    st.caption(f"Powered by LangChain & Google Gemini.") # This will use default caption styling unless you target it

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

