import asyncio
import json
import os

from datetime import date
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
from pathlib import Path
from source_metadata import SOURCE_METADATA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server/weather_server.py"],
)

# -----------------------------
# RAG setup
# -----------------------------

knowledge_base_path = Path("knowledge_base")

documents = []

for file_path in knowledge_base_path.glob("*.md"):
    loader = TextLoader(file_path, encoding="utf-8")

    loaded_docs = loader.load()

    file_sources = SOURCE_METADATA.get(file_path.name, [])

    for doc in loaded_docs:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["sources"] = file_sources

    documents.extend(loaded_docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

def convert_mcp_tools_to_gemini_tools(mcp_tools):
    declarations = []

    for tool in mcp_tools:
        declarations.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        })

    return declarations

def route_question(question):
    routing_prompt = f"""
        You are a router for a Singapore travel assistant.

        Decide what information is required to answer the user's question.

        Available information sources:

        Current date will be provided with the user question.
        Use it to correctly interpret relative dates such as:
        - today
        - tomorrow
        - next week
        - this weekend

        RAG:
        Use RAG for stable Singapore travel knowledge such as:
        - attractions
        - neighbourhoods
        - transportation
        - cultural and practical tips
        - food and local experiences
        - sample itineraries
        - indoor/outdoor activities

        MCP:
        Use MCP for current or changing information such as:
        - weather forecasts
        - currency conversion

        Return ONLY one of these values:

        RAG
        MCP
        BOTH

        User question:
        {question}
        """
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=routing_prompt
    )
    return response.text.strip().upper()

def format_docs_with_sources(docs):
    formatted_docs = []

    for doc in docs:
        source_file = doc.metadata.get("source_file", "Unknown")
        sources = doc.metadata.get("sources", [])
    
        source_lines = []

        for source in sources:
            source_lines.append(
                f"- {source['title']} ({source['url']})"
            )

        formatted_docs.append(
            f"""
            [Source file: {source_file}]
            [Sources:] 
            {chr(10).join(source_lines)}

            {doc.page_content}
            """
        )
    return "\n\n".join(formatted_docs)

def format_conversation_history(history):
    if not history:
        return "No previous conversation."

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

async def process_question(question, conversation_history, session, gemini_tools):

    # --------------------------------
    # Build question with conversation history
    # --------------------------------
    conversation_context = format_conversation_history(
        conversation_history
    )

    question_with_context = f"""
    Current date: {date.today().isoformat()}

    Conversation history:
    {conversation_context}

    Current user question:
    {question}
    """

    # --------------------------------
    # Route the question
    # --------------------------------
    route = route_question(question_with_context)

    # --------------------------------
    # RAG: Retrieve knowledge-base information
    # --------------------------------
    rag_context = ""

    if route in ["RAG", "BOTH"]:
        retrieved_docs = retriever.invoke(question_with_context)
        rag_context = format_docs_with_sources(retrieved_docs)

    # --------------------------------
    # MCP: Retrieve current information
    # --------------------------------
    mcp_result = {}

    if route in ["MCP", "BOTH"]:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=question_with_context,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        function_declarations=gemini_tools
                    )
                ]
            )
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts

        for part in parts:

            if part.function_call:

                function_call = part.function_call
                tool_name = function_call.name
                tool_args = dict(function_call.args)

                try:

                    tool_result = await session.call_tool(
                        tool_name,
                        arguments=tool_args
                    )

                    if tool_result.is_error:
                        mcp_result = {
                            "error": f"MCP tool '{tool_name}' failed",
                            "details": tool_result.content[0].text
                        }

                    else:
                        tool_text = tool_result.content[0].text
                        try:
                            mcp_result = json.loads(tool_text)

                        except json.JSONDecodeError:

                            mcp_result = {
                                "error": "MCP tool returned invalid JSON",
                                "raw_result": tool_text
                            }

                except Exception as e:

                    mcp_result = {
                        "error": f"MCP tool '{tool_name}' failed: {str(e)}"
                    }

    # --------------------------------
    # Generate final answer
    # --------------------------------
    combined_prompt = f"""
    You are a helpful Singapore travel assistant.

    You must answer using the information provided below.

    IMPORTANT GROUNDING RULES:
    1. MCP results are the source of truth for current information such as
    weather forecasts and currency exchange rates.
    2. The Knowledge Base is the source of truth for stable Singapore
    travel information.
    3. Do not invent facts that are not present in the Knowledge Base
    or MCP results.
    4. You may make reasonable travel-planning recommendations by
    combining the provided Knowledge Base information with MCP data.
    5. Clearly distinguish:
    - Information from the Knowledge Base
    - Current information from MCP tools
    - Your own travel-planning recommendations
    6. Do not describe weather conditions beyond what the MCP result
    supports.
    7. If the Knowledge Base does not contain enough information to answer
    a destination-related question, explicitly say that the Knowledge
    Base does not contain enough information.
    8. If an MCP tool fails or does not provide the requested information,
    explicitly say that current information could not be retrieved.
    Do not guess or invent the missing information.
    9. Preserve relevant preferences or constraints from the conversation
    history.

    CURRENT DATE:
    {date.today().isoformat()}

    CONVERSATION HISTORY:
    {conversation_context}

    KNOWLEDGE BASE CONTEXT:
    {rag_context if rag_context else "No Knowledge Base information was retrieved."}

    CURRENT MCP INFORMATION:
    {mcp_result if mcp_result else "No MCP information was retrieved."}

    CURRENT USER QUESTION:
    {question}

    Provide a clear, structured answer.
    """
    try:
        final_response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=combined_prompt
        )
        answer = final_response.text

    except Exception as e:
        print("\nFinal Gemini response failed:")
        print(str(e))

        answer = (
            "I could not generate the final AI response "
            "because the language model is temporarily unavailable."
        )

    # --------------------------------
    # Update conversation history
    # --------------------------------

    conversation_history.append({
        "role": "User",
        "content": question
    })

    conversation_history.append({
        "role": "Assistant",
        "content": answer
    })

    return answer

async def main():
    conversation_history = []

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover the tools exposed by our MCP server.
            tools_result = await session.list_tools()

            # Convert MCP tool definitions into the format
            # expected by Gemini function calling.
            gemini_tools = convert_mcp_tools_to_gemini_tools(
                tools_result.tools
            )

            # For set of questions asked
            while True:
                question = input("\nYou: ")

                if question.lower() == "exit":
                    break

                try:
                    answer = await process_question(
                        question,
                        conversation_history,
                        session,
                        gemini_tools
                    )
                    print("\nAssistant:")
                    print(answer)

                except Exception as e:
                    print("\nAssistant:")
                    print(
                        "I could not generate a response because "
                        "the language model is temporarily unavailable."
                    )

if __name__ == "__main__":
    asyncio.run(main())