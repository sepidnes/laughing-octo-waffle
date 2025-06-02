from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.prompts import PromptTemplate
from langserve import add_routes
from langchain_community.llms import VLLMOpenAI
#from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field
from typing import Any, List, Union, TypedDict
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough

#docker run -p 6333:6333 -p 6334:6334     -v "$(pwd)/qdrant_storage:/qdrant/storage:z"     qdrant/qdrant
# rm -rf .venv
# uv venv --python=python3.11
# source .venv/bin/activate
# uv pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu && uv pip install vllm==0.8.3 --no-build-isolation

load_dotenv()


url = "http://127.0.0.1:6334"

app = FastAPI()
# pip install vllm --no-deps --break-system-packages && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
#pip install transformers --break-system-packages
#pip install msgspec --break-system-packages

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
#pip install msgspec --break-system-packages
llm = VLLMOpenAI(
    openai_api_key="sepid-001",
    openai_api_base="http://192.168.2.42:8001/v1",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Smaller model that's more CPU-friendly#HouseResearch/Meta-Llama-3-8B-Instruct
    temperature=0.7,
    max_tokens=512,
    gpu_memory_utilization=0.0,  # Force CPU usage
    quantization="q4_0"  # Use 4-bit quantization
)

RAG_PROMPT_TEMPLATE = """
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. The questions will be about Dungeons and Dragons.

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# embeddings = OllamaEmbeddings(
#     model="mxbai-embed-large",
# )





embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="DnD_Documents",
    prefer_grpc=True,
    url=url
)

retriever = qdrant_vectorstore.as_retriever()

lcel_rag_chain = (
    {"context": itemgetter("query") | retriever, "query": itemgetter("query")} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

# Initialize Qdrant client separately
# qdrant_client = QdrantClient(url=url, prefer_grpc=True)

# qdrant_vectorstore = QdrantVectorStore(
#     client=qdrant_client,
#     collection_name="DnD_Documents",
#     embedding=embeddings
# )

# retriever = qdrant_vectorstore.as_retriever()

# from langchain_core.runnables import RunnablePassthrough


# lcel_rag_chain = (
#     {"context": retriever, "query": RunnablePassthrough()}
#     | rag_prompt
#     | llm
# )

class Input(BaseModel):
    query: str = Field(description="The query to search for in the DnD documents")

class Output(BaseModel):
    output: str = Field(description="The response from the RAG chain")

Input.model_rebuild()
Output.model_rebuild()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/dnd/playground")

add_routes(
    app,
    lcel_rag_chain.with_types(
    input_type=Input,output_type=Output).with_config(
        {"run_name": "DnD_RAG"}
        ),
    path="/dnd"
)

#--------------------------------------------
#-------------------------------------------
# configured_chain = lcel_rag_chain.with_config({"run_name": "DnD_RAG"})

# from fastapi import Request

# @app.post("/dnd", response_model=Output)
# async def dnd_rag_endpoint(input: Input, request: Request):
#     result = await configured_chain.ainvoke({"query": input.query})
#     return {"output": result}

#--------------------------------------------
# FastAPI endpoint
#-------------------------------------------
# configured_chain = lcel_rag_chain.with_config({"run_name": "DnD_RAG"})
# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

# # Configure the chain with proper types
# configured_chain = lcel_rag_chain.with_types(
#     input_type=Input,
#     output_type=Output
# ).with_config({"run_name": "DnD_RAG"})

# from fastapi import Request

# @app.post("/dnd", response_model=Output)
# async def dnd_rag_endpoint(input: Input, request: Request):
#     result = await configured_chain.ainvoke({"query": input.query})
#     return {"output": result}

#--------------------------------------------
#-------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)