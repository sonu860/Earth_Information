
import os
import streamlit as st

from dotenv import load_dotenv
from typing import Annotated
from pydantic import BaseModel, Field, StrictStr, field_validator

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Earth Science RAG",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 Earth Science RAG Chatbot")
st.caption("Ask questions about Earth using the provided document.")


# ============================================================
# 2. DOCUMENT
# ============================================================

EARTH_TEXT = """
Earth, our home planet, is the only planet in our solar system
known to harbor life. Earth is the third planet from the Sun
and the fifth largest in the solar system.

Earth's axis is tilted 23.45 degrees with respect to the plane
of its orbit around the Sun. This tilt produces the seasons.

Earth's global ocean covers nearly 70 percent of the planet's
surface and has an average depth of about 4 kilometers.

Near the surface, Earth's atmosphere consists of 78 percent
nitrogen, 21 percent oxygen, and 1 percent other ingredients.

The atmosphere affects Earth's long-term climate and short-term
weather. It shields us from much of the harmful radiation from
the Sun and protects us from many meteors.

Earth's rotation and molten nickel-iron core give rise to a
magnetic field. Charged particles from the solar wind can
interact with air molecules near the magnetic poles, producing
aurorae, also known as the northern and southern lights.

Earth's lithosphere includes the crust and upper mantle. It is
divided into large plates that constantly move. Their movement
is explained by the theory of plate tectonics.
"""

document = Document(page_content=EARTH_TEXT)


# ============================================================
# 3. PYDANTIC OUTPUT SCHEMA
# ============================================================

class ParserOutput(BaseModel):
    answer: Annotated[
        StrictStr,
        Field(description="Answer based only on the retrieved context"),
    ]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Answer cannot be empty")

        return value


parser = PydanticOutputParser(
    pydantic_object=ParserOutput
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# 5. CREATE RAG CHAIN
# ============================================================

@st.cache_resource
def create_chain():

    google_api_key = os.getenv("GOOGLE_API_KEY")
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    if not endpoint or not token:
        raise ValueError(
            "ASTRA_DB_API_ENDPOINT or ASTRA_DB_APPLICATION_TOKEN is missing"
        )

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents([document])

    if not chunks:
        raise ValueError("No document chunks were created")

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key,
    )

    # Chat model
    # Verify that this model ID is supported by your installed package/API.
    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=google_api_key,
        temperature=0,
    )

    # AstraDB vector store
    vector_store = AstraDBVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="earth_docs",
        token=token,
        api_endpoint=endpoint,
    )

    # MMR retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    # Multi-query retriever
    multi_retriever = MultiQueryRetriever.from_llm(
        llm=model,
        retriever=retriever,
    )

    # Contextual compression
    compressor = LLMChainExtractor.from_llm(model)

    contextual_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever,
    )

    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # Prompt with chat history
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an Earth science expert.

            Answer the user's question using only the provided context.
            If the answer is not present in the context, say you don't
            know based on the provided document.

            Context:
            {context}

            {format_instruction}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]).partial(
        format_instruction=parser.get_format_instructions()
    )

    # RAG chain
    chain = (
        {
            "context": (
                RunnableLambda(lambda x: x["question"])
                | contextual_retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | prompt
        | model
        | parser
    )

    return chain


# ============================================================
# 6. DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


# ============================================================
# 7. USER INPUT
# ============================================================

question = st.chat_input("Ask about Earth...")


if question:

    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Searching documents and generating answer..."):

            chain = create_chain()

            response = chain.invoke({
                "question": question.strip(),
                "chat_history": st.session_state.chat_history,
            })

        # Save user and assistant messages
        st.session_state.chat_history.extend([
            HumanMessage(content=question.strip()),
            AIMessage(content=response.answer),
        ])

        with st.chat_message("assistant"):
            st.write(response.answer)

    except Exception as error:
        st.error(f"Something went wrong: {error}")
