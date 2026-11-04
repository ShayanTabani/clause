from os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env") 
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Indexing
#1.Load
text=Path("data/shop_handbook.md").read_text(encoding="utf-8")
doc=Document(
    page_content=text,
    metadata={"source": "shop_handbook.md"}
    )

#2.Chunks
splitter= RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks=splitter.split_documents([doc])

#3.Embed
embeddings=init_embeddings(
     model="os.getenv('EMBEDDINGS_MODEL')",
     provider="os.getenv('EMBEDDINGS_PROVIDER')" 
     )
#4.STORE
vector_store=InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

#RAG
##LLM DEFINING
llm=init_chat_model(
    model=os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("CHAT_PROVIDER"),
    api_key=os.getenv("CHAT_API_KEY") or None,
    temperature=0.1
)
query=input("You:").strip()

#5-7.RETRIEVING
retriver=vector_store.similarity_search(query,k=3)
texts=[]
for doc in retriver:
    texts.append(doc.page_content)
context="\n\n".join(texts)#string

messages=[
    {"role":"system", "content":"You are a helpful Clause shop assistant"},
    {"role":"system", "content":f"Context:{context}\n\nQuestion:{query}"}
]

print(llm.invoke(input=messages).text)

