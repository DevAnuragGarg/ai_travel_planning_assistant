from datetime import date

from llm import (
    generate_final_answer,
    get_tool_call,
    route_question,
    select_mcp_tool,
)
from mcp_client import call_mcp_tool, is_valid_mcp_tool
from rag import format_docs_with_sources, retriever


def format_conversation_history(history):
    """
    Convert the stored conversation history into plain text that can
    be included in the LLM prompts.

    The history is maintained by the application as a list of messages:
        [
            {"role": "User", "content": "..."},
            {"role": "Assistant", "content": "..."}
        ]
    """
    if not history:
        return "No previous conversation."

    return "\n".join(f"{message['role']}: {message['content']}" for message in history)


async def process_question(
    question,
    conversation_history,
    session,
    gemini_tools,
    mcp_tools,
):
    """
    Main orchestration function for the travel assistant.

    Flow:
        1. Add current date and conversation history to the question.
        2. Use the LLM to decide whether RAG, MCP, or BOTH are required.
        3. Retrieve stable destination knowledge through RAG when required.
        4. Select and call an appropriate MCP tool when current information
           is required.
        5. Send the retrieved information to the final LLM.
        6. Store the user question and assistant response in conversation
           history for multi-turn conversations.

    MCP and RAG are deliberately handled separately because they serve
    different purposes:
        - RAG -> stable Singapore travel knowledge
        - MCP -> current/changing information such as weather and currency
    """
    print(f"Question: {question}")

    try:
        # ------------------------------------------------------------
        # 1. Build the contextual question
        # ------------------------------------------------------------
        # The current date allows the LLM to correctly interpret relative
        # dates such as "today", "tomorrow", and "next week".
        #
        # Conversation history allows follow-up questions to retain context.
        conversation_context = format_conversation_history(conversation_history)

        current_date = date.today().isoformat()

        contextual_question = f"""
            Current date: {current_date}

            Conversation history:
            {conversation_context}

            Current user question:
            {question}
        """

        # ------------------------------------------------------------
        # 2. Route the question
        # ------------------------------------------------------------
        # The router decides which information source is required:
        #
        # RAG  -> stable destination knowledge
        # MCP  -> current/changing information
        # BOTH -> combination of RAG and MCP
        route = route_question(contextual_question)

        print(f"Route selected: {route}")

        # ------------------------------------------------------------
        # 3. Retrieve information from the Knowledge Base using RAG
        # ------------------------------------------------------------
        rag_context = ""

        if route in ["RAG", "BOTH"]:
            # Semantic retrieval finds the most relevant knowledge-base
            # chunks for the user's question.
            retrieved_docs = retriever.invoke(contextual_question)

            # Include source metadata with the retrieved content so that
            # the final LLM can provide source references in its answer.
            rag_context = format_docs_with_sources(retrieved_docs)

        # ------------------------------------------------------------
        # 4. Retrieve current information through MCP
        # ------------------------------------------------------------
        mcp_result = {}

        if route in ["MCP", "BOTH"]:
            # The LLM selects an MCP tool based on the user's requirement.
            # The available tool schemas are passed to the LLM so that it
            # can generate a valid tool call.
            tool_selection_response = select_mcp_tool(
                contextual_question,
                gemini_tools,
            )

            tool_name, tool_args = get_tool_call(tool_selection_response)

            print(f"MCP Tool name selected: {tool_name}")

            # If the LLM did not select a tool, do not attempt to guess one.
            if not tool_name:
                mcp_result = {"error": "No MCP tool was selected."}

            # Validate the LLM-generated tool name against the tools
            # actually exposed by the MCP server.
            elif not is_valid_mcp_tool(tool_name, mcp_tools):
                mcp_result = {"error": f"MCP tool '{tool_name}' is not available."}

            else:
                # Execute the selected MCP tool using the active MCP session.
                # call_mcp_tool() also handles MCP-level errors and converts
                # the result into a Python dictionary.
                mcp_result = await call_mcp_tool(
                    session,
                    tool_name,
                    tool_args,
                )

                print(f"MCP Result: {mcp_result}")

        # ------------------------------------------------------------
        # 5. Generate the final grounded answer
        # ------------------------------------------------------------
        # At this point the LLM receives:
        #
        #   - conversation history
        #   - retrieved RAG context
        #   - current MCP result
        #   - current user question
        #
        # The prompt explicitly prevents the model from treating its
        # general knowledge as a source of unsupported travel facts.
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

            4. You may make travel-planning recommendations by combining the
            provided Knowledge Base information with MCP data. Clearly label these
            as recommendations. Do not introduce new factual claims that are not
            supported by the Knowledge Base or MCP results.

            5. Clearly distinguish:
            - Information from the Knowledge Base
            - Current information from MCP tools
            - Your own travel-planning recommendations

            6. For information taken from the Knowledge Base, you MUST include
            a "Sources" section at the end of your answer.

            In the Sources section, list the relevant source title and URL exactly
            as provided in the Knowledge Base context.

            Do not invent, modify, or omit the source URLs provided in the
            Knowledge Base context.

            7. Do not describe weather conditions beyond what the MCP result supports.

            8. If the Knowledge Base does not contain enough information to answer
            a destination-related question, explicitly say that the Knowledge
            Base does not contain enough information.

            9. If an MCP tool fails or does not provide the requested information,
            explicitly say that current information could not be retrieved.

            Do not guess or invent the missing information.

            10. Preserve relevant preferences or constraints from the conversation
                history.

            CURRENT DATE:
            {current_date}

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
        answer = generate_final_answer(combined_prompt)

    except Exception as e:
        # ------------------------------------------------------------
        # 6. Graceful failure handling
        # ------------------------------------------------------------
        # The application should remain usable even when an external
        # dependency such as the LLM API is unavailable or quota-limited.
        #
        # Importantly, the failed turn is still stored in conversation
        # history below so that the UI and conversation state remain
        # consistent.
        print("\nAssistant processing failed:")
        print(type(e).__name__)
        print(str(e))

        answer = (
            "I could not generate a response because "
            "the language model is temporarily unavailable."
        )

    # ------------------------------------------------------------
    # 7. Preserve multi-turn conversation context
    # ------------------------------------------------------------
    # Streamlit reruns the application after user interaction, so the
    # conversation history is stored in st.session_state by the UI.
    # Adding both sides of the conversation here makes the history
    # available to subsequent questions.
    conversation_history.append(
        {
            "role": "User",
            "content": question,
        }
    )

    conversation_history.append(
        {
            "role": "Assistant",
            "content": answer,
        }
    )

    print("\n============= Conversation History =============")
    print(conversation_history)

    return answer
