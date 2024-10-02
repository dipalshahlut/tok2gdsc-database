import os
import io
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import streamlit as st

# Load environment variables
load_dotenv()


# Function to display a molecule from a SMILES string
def display_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        img = Draw.MolToImage(molecule, size=(300, 300))
        return img
    else:
        return None


def init_database(db_path: str) -> SQLDatabase:
    db_uri = f"sqlite:///{db_path}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Your turn:

    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=openai_api_key)

    def get_schema(_):
        return db.get_table_info()

    return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    print(user_query)
    if "SMILES:" in user_query:
        smiles = user_query.split("SMILES:")[1].strip()
        img = display_molecule(smiles)
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return buf
        else:
            return "Invalid SMILES string. Please enter a correct SMILES notation."
    else:
        sql_chain = get_sql_chain(db)
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        """
        prompt = ChatPromptTemplate.from_template(template)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        chain = (
                RunnablePassthrough.assign(query=sql_chain.invoke).assign(
                    schema=lambda _: db.get_table_info(),
                    response=lambda vars: db.run(vars["query"]),
                )
                | prompt
                | llm
                | StrOutputParser()
        )

        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Hello! I'm a SQL and --------. Ask me anything about your database or enter a SMILES string prefixed with 'SMILES:' to visualize a molecule."),
    ]

st.set_page_config(page_title="Chat with ----", page_icon=":speech_balloon:")

st.title("Chat with SMILES!")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using SQLite. Connect to the database and start chatting.")

    db_path = st.text_input("SQLite DB Path", value="/Users/rahulsharma/Dropbox/LLMchatBotGDSC/gdsc_data.db",
                            key="db_path")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(db_path)
            st.session_state.db = db
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)

    with st.chat_message("AI"):
        if isinstance(response, io.BytesIO):  # If response is an image
            st.image(response)
            response = "Displayed the molecule structure above."  # Convert image response to a meaningful string message
        elif isinstance(response, str):  # If response is already a string
            st.markdown(response)
        else:
            response = str(response)  # Convert other types of responses to string
            st.markdown(response)

    # Now ensure that response is a string and log it
    if not isinstance(response, str):
        response = str(response)  # Safety check to convert any type to string

    st.session_state.chat_history.append(AIMessage(content=response))
