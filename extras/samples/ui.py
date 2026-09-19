import asyncio
import streamlit as st

from mcp_llm_test import (
    process_question,
    convert_mcp_tools_to_gemini_tools,
    server_params,
)
from mcp import ClientSession
from mcp.client.stdio import stdio_client


async def run_assistant(question, conversation_history):

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools_result = await session.list_tools()

            gemini_tools = convert_mcp_tools_to_gemini_tools(
                tools_result.tools
            )

            answer = await process_question(
                question,
                conversation_history,
                session,
                gemini_tools
            )

            return answer


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("✈️ Singapore AI Travel Assistant")

# Create conversation history once
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Display previous conversation
for message in st.session_state.conversation_history:

    if message["role"] == "User":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.write(message["content"])


# Chat input
question = st.chat_input("Ask something about Singapore...")


# Process new question
if question:

    # Show user's question immediately
    with st.chat_message("user"):
        st.write(question)

    # Call our existing AI assistant
    with st.spinner("Thinking..."):

        asyncio.run(
            run_assistant(
                question,
                st.session_state.conversation_history
            )
        )

    # process_question() already added the answer to history
    answer = st.session_state.conversation_history[-1]["content"]

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)