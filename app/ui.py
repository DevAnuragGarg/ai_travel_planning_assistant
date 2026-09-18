import asyncio

import streamlit as st

from assistant import process_question
from mcp_client import (
    convert_mcp_tools_to_gemini_tools,
    mcp_session,
)


# ------------------------------------------------------------
# Run the AI Assistant
# ------------------------------------------------------------
async def run_assistant(question, conversation_history):
    """
    Create an MCP session, discover the available MCP tools,
    and pass everything to the main assistant orchestration.

    The UI layer does not decide whether RAG or MCP should be used.
    That responsibility belongs to assistant.py and llm.py.

    Flow:
        Streamlit UI
             ↓
        run_assistant()
             ↓
        MCP session + tool discovery
             ↓
        process_question()
             ↓
        RAG / MCP / BOTH
             ↓
        Final answer
    """

    # Start an MCP server connection and keep the session alive
    # while this request is being processed.
    async with mcp_session() as session:

        # Discover the tools currently exposed by the MCP server.
        #
        # This avoids hardcoding the MCP tool list in the UI.
        tools_result = await session.list_tools()

        # MCP provides its own tool definitions. Convert those
        # definitions into the format expected by Gemini so that
        # the LLM can select an appropriate tool.
        gemini_tools = convert_mcp_tools_to_gemini_tools(tools_result.tools)

        # Delegate the actual assistant workflow to assistant.py.
        #
        # process_question() handles:
        #   - conversation context
        #   - RAG/MCP/BOTH routing
        #   - RAG retrieval
        #   - MCP tool selection and execution
        #   - final LLM response
        answer = await process_question(
            question,
            conversation_history,
            session,
            gemini_tools,
            tools_result.tools,
        )

        print("\nAssistant:")
        print(answer)

        return answer


# ------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------

st.title("✈️ Singapore AI Travel Assistant")


# ------------------------------------------------------------
# Conversation State
# ------------------------------------------------------------
# Streamlit reruns the Python script whenever the user submits
# a new message.
#
# st.session_state allows the conversation history to survive
# these reruns.
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ------------------------------------------------------------
# Display Previous Conversation
# ------------------------------------------------------------
# Re-render all previous messages whenever Streamlit reruns.
# The history itself is stored in session_state.
for message in st.session_state.conversation_history:

    if message["role"] == "User":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.write(message["content"])


# ------------------------------------------------------------
# Get New User Question
# ------------------------------------------------------------
# chat_input waits for the user to submit a new question.
question = st.chat_input("Ask something about Singapore...")


# ------------------------------------------------------------
# Process New Question
# ------------------------------------------------------------
if question:

    # Display the user's message immediately.
    with st.chat_message("user"):
        st.write(question)

    # Show a progress indicator while the assistant performs:
    #
    #   Routing → RAG/MCP → Final LLM generation
    #
    # The actual processing is delegated to assistant.py.
    with st.spinner("Thinking..."):

        answer = asyncio.run(
            run_assistant(
                question,
                st.session_state.conversation_history,
            )
        )

    # Display the final assistant response.
    with st.chat_message("assistant"):
        st.write(answer)
