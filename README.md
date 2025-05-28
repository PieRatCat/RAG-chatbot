# Dragon Age Lore RAG Chatbot ("Brother Genitivi's Dragon Age Archives")

This project is a Retrieval Augmented Generation (RAG) chatbot built to answer questions about the Dragon Age game series, based on content from PDF game guides. It uses LangChain, Google Gemini models, ChromaDB, and is presented as a Streamlit web application with a "Brother Genitivi" persona.  

The project was created for the DS24 Deep Learning course.  

### Main Files & Directories

* app.py: The main Streamlit application script that runs the chatbot interface and the RAG chain. Deployed in Streamlit Cloud at <https://da-chatbot.streamlit.app/>.

* chatbot.ipynb: Jupyter Notebook containing the Python code for data preprocessing: loading PDFs, chunking text, generating embeddings, and creating the ChromaDB vector store. [View the notebook in nbviewer](https://nbviewer.org/github/PieRatCat/RAG-chatbot/blob/main/chatbot.ipynb).

* style.css: Contains the custom CSS rules for theming the Streamlit application.

* requirements.txt: Lists the Python dependencies required to run the project.

* chatbot_db/: The directory where the persisted ChromaDB vector store is saved (contains the embedded knowledge base). Large files within are tracked by Git LFS.

* .gitattributes: Configures Git LFS to track large files (like the ChromaDB database).

### Running the App

Ensure Python 3.11 and dependencies from requirements.txt are installed.  

Set up your GOOGLE_API_KEY (e.g., in .streamlit/secrets.toml locally or as an environment variable).  

The chatbot_db/ directory (with LFS files pulled) must be present.  

Run: streamlit run app.py  

#### Disclaimer: This is a non-commercial student project for educational purposes. All Dragon Age copyrights belong to BioWare/EA.
