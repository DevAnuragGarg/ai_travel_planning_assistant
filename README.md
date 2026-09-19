# ✈️ Singapore AI Travel Planning Assistant

An AI-powered travel planning assistant for Singapore that combines
**Retrieval-Augmented Generation (RAG)** with **Model Context Protocol
(MCP)**.

The application uses a local Singapore travel knowledge base for stable
destination information and MCP tools for dynamic information such as
weather forecasts and currency conversion.

## GIT Repo Link: https://github.com/DevAnuragGarg/ai_travel_planning_assistant

------------------------------------------------------------------------

## 1. Project Overview

The objective is to build a travel assistant that can distinguish
between:

1.  **Stable destination knowledge** --- retrieved from the local RAG
    knowledge base.
2.  **Current/dynamic information** --- retrieved through MCP tools.
3.  **Combined requests** --- RAG provides destination knowledge while
    MCP provides current information, and the LLM combines both.

### Main capabilities

-   Singapore attractions and neighbourhoods
-   Transportation guidance
-   Cultural, food and practical travel information
-   Sample itineraries
-   Indoor and outdoor activities
-   Current weather through MCP
-   Currency conversion through MCP
-   Combined RAG + MCP weather-aware itinerary planning
-   Multi-turn conversation context
-   Knowledge-base source titles and URLs
-   Missing-knowledge and MCP/API failure handling
-   Streamlit chat interface

------------------------------------------------------------------------

## 2. Assignment Requirements Mapping

  ------------------------------------------------------------------------
  Requirement                         Implementation
  ----------------------------------- ------------------------------------
  Destination knowledge using RAG     Local Singapore Markdown knowledge
                                      base

  Major attractions and               `knowledge_base/attractions.md`
  neighbourhoods                      

  Local transportation guidance       `knowledge_base/transportation.md`

  Cultural/practical travel tips      `knowledge_base/cultural_food.md`

  Food/local experiences              `knowledge_base/cultural_food.md`

  Sample itineraries                  `knowledge_base/itineraries.md`

  Indoor/outdoor activities           `attractions.md` and
                                      `itineraries.md`

  Meaningful document chunking        `RecursiveCharacterTextSplitter`

  Embeddings                          Gemini `gemini-embedding-001`

  Vector store                        FAISS

  Semantic retrieval                  LangChain FAISS retriever

  Source title/link                   Source metadata attached to
                                      retrieved documents

  Current weather                     MCP `get_weather`

  Currency conversion                 MCP `convert_currency`

  At least two MCP tools              Two custom tools in
                                      `travel_server.py`

  AI tool selection                   Gemini tool-selection step

  Tool input passing                  MCP calls using model-generated
                                      arguments

  RAG + MCP combined scenario         Three-day weather-aware Singapore
                                      itinerary

  Multi-turn context                  Streamlit `session_state`
                                      conversation history

  Missing KB knowledge                Grounding prompt requires explicit
                                      acknowledgement

  MCP failure handling                Exceptions are propagated and
                                      surfaced

  Simple usable interface             Streamlit chat UI

  LangChain                           Document loading, splitting,
                                      embeddings and retrieval

  Documentation                       This README
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. High-Level Architecture

``` text
                              ┌─────────────────────────┐
                              │       User / Browser    │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │      Streamlit UI       │
                              │        app/ui.py        │
                              │                         │
                              │  Chat input             │
                              │  Conversation display   │
                              │  Session history        │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │    Assistant Layer      │
                              │    app/assistant.py     │
                              │                         │
                              │  1. Route question      │
                              │  2. Retrieve RAG data   │
                              │  3. Select MCP tool     │
                              │  4. Execute MCP tool    │
                              │  5. Build final prompt  │
                              │  6. Generate answer     │
                              └───────┬─────────┬───────┘
                                      │         │
                         ┌────────────┘         └────────────┐
                         │                                   │
                         ▼                                   ▼
                ┌──────────────────┐                ┌──────────────────┐
                │       RAG        │                │   MCP Client     │
                │    app/rag.py    │                │ app/mcp_client.py│
                │                  │                │                  │
                │   Markdown KB    │                │ Discover tools   │
                │        ↓         │                │ Convert schemas  │
                │     Chunking     │                │ Validate tool    │
                │        ↓         │                │ Execute tool     │
                │    Embeddings    │                │                  │
                │        ↓         │                └────────┬─────────┘
                │      FAISS       │                         │
                │        ↓         │                    stdio│
                │    Retrieval     │                         │
                └────────┬─────────┘                         ▼
                         │                         ┌─────────────────────┐
                         │                         │   MCP Travel Server │
                         │                         │     mcp_server/     │
                         │                         │ travel_server.py    │
                         │                         │                     │
                         │                         │ get_weather()       │
                         │                         │ convert_currency()  │
                         │                         └──────────┬──────────┘
                         │                                    │
                         │                     ┌──────────────┴─────────────┐
                         │                     │                            │
                         │                     ▼                            ▼
                         │              ┌──────────────┐             ┌──────────────┐
                         │              │  Open-Meteo  │             │  Frankfurter │
                         │              │ Weather API  │             │ Currency API │
                         │              └──────────────┘             └──────────────┘
                         │
                         └─────────────────────┬─────────────────────────────┐
                                               │                             │
                                               ▼                             ▼
                                      ┌────────────────────────────────────────┐
                                      │              Gemini LLM                │
                                      │              app/llm.py                │
                                      │                                        │
                                      │           Question Routing             │
                                      │           RAG / MCP / BOTH             │
                                      │                                        │
                                      │          MCP Tool Selection            │
                                      │                                        │
                                      │        Final Answer Generation         │
                                      └───────────────────┬────────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │ Final Response   │
                                               │ to Streamlit UI  │
                                               └──────────────────┘
```

## Architecture Flow
```text
User Question
      ↓
Streamlit UI
      ↓
Assistant Orchestrator
      ↓
Question Routing
      │
      ├────────────── RAG ──────────────┐
      │                                 │
      │                         Retrieve relevant
      │                        Singapore knowledge
      │                                 │
      │                                 ▼
      │                            RAG Context
      │
      ├────────────── MCP ──────────────┐
      │                                 │
      │                         Select appropriate
      │                              MCP tool
      │                                 │
      │                                 ▼
      │                          Current API Data
      │
      └────────────── BOTH ─────────────┘
                       │
                       ▼
               Combine RAG + MCP
                       │
                       ▼
                 Final Prompt
                       │
                       ▼
                  Gemini LLM
                       │
                       ▼
               Grounded Response
                       │
                       ▼
                  Streamlit UI
```

------------------------------------------------------------------------

## 4. Project Structure

``` text
ai-travel-planning-assistant/
│
├── app/
│   ├── assistant.py
│   ├── llm.py
│   ├── mcp_client.py
│   ├── rag.py
│   ├── source_metadata.py
│   └── ui.py
│
├── knowledge_base/
│   ├── attractions.md
│   ├── cultural_food.md
│   ├── itineraries.md
│   └── transportation.md
│
├── mcp_server/
│   └── travel_server.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### File responsibilities

**`app/ui.py`** --- Streamlit chat interface and session history.

**`app/assistant.py`** --- Main orchestration: routing, retrieval, MCP
selection/execution, final prompt, answer and conversation history.

**`app/llm.py`** --- Gemini integration for routing, MCP tool selection
and final answer generation.

**`app/rag.py`** --- Knowledge-base loading, source metadata, chunking,
embeddings, FAISS and retrieval.

**`app/mcp_client.py`** --- MCP stdio session, tool discovery, Gemini
schema conversion, validation and tool calls.

**`app/source_metadata.py`** --- Maps knowledge-base files to source
titles and URLs.

**`mcp_server/travel_server.py`** --- Custom MCP server exposing weather
and currency tools.

------------------------------------------------------------------------

# 5. Knowledge Base

The local knowledge base contains four Markdown resources.

### `attractions.md`

Contains major attractions and neighbourhoods including Gardens by the
Bay, Marina Bay, Sentosa, Singapore Botanic Gardens, Mandai, Chinatown,
Little India, Kampong Glam and Orchard Road. It also contains indoor,
outdoor and family activity suggestions.

### `cultural_food.md`

Contains languages, climate, drinking water, payments, tipping, food
culture, hawker experiences, local foods, etiquette and cultural
neighbourhoods.

### `itineraries.md`

Contains sample three-day itinerary ideas, cultural and family
itineraries, indoor/outdoor planning and a seven-day itinerary
reference.

### `transportation.md`

Contains general guidance about MRT, buses, payment, taxis/private hire,
walking and transport planning.

The knowledge base is intentionally focused on relatively stable
information. Live transport status, live fares and live route
information are not stored as static facts.

------------------------------------------------------------------------

# 6. Knowledge-Base Sources

The KB records source titles and URLs.

### Attractions

-   Wikivoyage --- Singapore Travel Guide\
    https://en.wikivoyage.org/wiki/Singapore
-   Visit Singapore --- Things to Do\
    https://www.visitsingapore.com/

### Culture/Food

-   Visit Singapore --- Essential Travel Information\
    https://www.visitsingapore.com/travel-tips/essential-travel-information/
-   Wikivoyage --- Singapore Travel Guide\
    https://en.wikivoyage.org/wiki/Singapore

### Itineraries

-   Visit Singapore --- Enjoy Singapore in 7 Days\
    https://www.visitsingapore.com/content/visitsingapore/en/travel-tips/travelling-to-singapore/itineraries/7-days-in-singapore
-   Visit Singapore --- Singapore Itineraries\
    https://www.visitsingapore.com/singapore-itineraries/
-   Visit Singapore --- Travel Itineraries\
    https://www.visitsingapore.com/

### Transportation

-   Visit Singapore --- Essential Travel Information\
    https://www.visitsingapore.com/travel-tips/essential-travel-information/
-   Wikivoyage --- Singapore Travel Guide\
    https://en.wikivoyage.org/wiki/Singapore

------------------------------------------------------------------------

# 7. RAG Implementation

The RAG pipeline in `app/rag.py` follows:

``` text
Markdown documents
       ↓
TextLoader
       ↓
RecursiveCharacterTextSplitter
       ↓
Gemini embeddings
       ↓
FAISS vector store
       ↓
Semantic retriever
       ↓
Relevant chunks + source metadata
       ↓
Final grounded prompt
       ↓
LLM response
```

## RAG Workflow
The RAG pipeline converts the static Singapore travel knowledge base into searchable vector representations and retrieves the most relevant information for each user question.

The complete workflow is:
```text
                 ┌─────────────────────────┐
                 │      Singapore KB       │
                 │   Markdown Documents    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │        TextLoader       │
                 │    Load .md documents   |
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Add Metadata         │
                 │                         │
                 │ source_file             │
                 │ source title            │
                 │ source URL              │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Text Splitter       │
                 │                         │
                 │ chunk_size = 800        │
                 │ overlap = 100           │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Gemini Embeddings    │
                 │                         │
                 │   gemini-embedding-001  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      FAISS Index        │
                 │                         │
                 │ Vector representations  │
                 │ of KB chunks            │
                 └────────────┬────────────┘
                              │
                              │
                        USER QUESTION
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Query Embedding     │
                 │                         │
                 │ Convert question into   │
                 │ vector representation   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     FAISS Retriever     │
                 │                         │
                 │ Semantic similarity     │
                 │ search                  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Top 5 Relevant Chunks  │
                 │                         │
                 │ Content + source        │
                 │ metadata                │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Final Prompt       │
                 │                         │
                 │ Question + retrieved    │
                 │ context + instructions  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       Gemini LLM        │
                 │                         │
                 │ Grounded answer         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Final Response      │
                 │                         │
                 │ Answer + source refs    │
                 └─────────────────────────┘
```

## 7.1 Document Loading

All Markdown files under:

``` text
knowledge_base/*.md
```

are loaded using LangChain's `TextLoader`.

Each document receives metadata containing its source filename and
associated source references.

## 7.2 Chunking

The application uses:

``` python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
```

The 800-character target keeps chunks focused, while the 100-character
overlap helps preserve context across boundaries.

## 7.3 Embeddings

The embedding model is:

``` text
models/gemini-embedding-001
```

Text is converted into vectors so that semantically related questions
and documents can be matched.

## 7.4 FAISS

FAISS stores the embedding vectors and supports similarity retrieval.

The retriever returns:

``` text
k = 5
```

relevant chunks.

## 7.5 Semantic Retrieval

Retrieval is based on semantic similarity rather than simple keyword
matching.

For example:

``` text
"What are some ways to get around Singapore?"
```

can retrieve transportation content even when the wording differs from
the source.

## 7.6 Source Grounding

Retrieved chunks are formatted with:

-   source filename
-   source title
-   source URL
-   retrieved text

The final prompt instructs the LLM to use the retrieved content for
stable destination facts and include the supplied source references.

------------------------------------------------------------------------

# 8. MCP Implementation

The project contains a custom MCP server:

``` text
mcp_server/travel_server.py
```

It exposes two tools:

``` text
Singapore Travel Server
├── get_weather()
└── convert_currency()
```

MCP is deliberately used for dynamic information rather than static
destination knowledge.

------------------------------------------------------------------------

## 8.1 Weather Tool

``` text
get_weather(
    latitude,
    longitude,
    start_date,
    end_date
)
```

The tool calls the Open-Meteo Forecast API and retrieves:

-   current temperature
-   current weather code
-   current wind speed
-   daily weather code
-   daily maximum temperature
-   daily minimum temperature
-   maximum precipitation probability

Weather is dynamic, so it is retrieved at runtime through MCP.

------------------------------------------------------------------------

## 8.2 Currency Tool

``` text
convert_currency(
    amount,
    from_currency,
    to_currency
)
```

The tool calls Frankfurter for the latest available exchange-rate
conversion.

Currency is dynamic, so it is retrieved at runtime through MCP rather
than being stored in the static KB.

------------------------------------------------------------------------

# 9. MCP Client Flow

`app/mcp_client.py` handles the client side:

``` text
MCP client
   ↓
start travel_server.py
   ↓
stdio communication
   ↓
initialize ClientSession
   ↓
list available tools
   ↓
convert MCP schemas to Gemini declarations
   ↓
Gemini selects tool
   ↓
validate tool
   ↓
call MCP tool
   ↓
receive structured result
```

The adapter uses the MCP tool's input schema to create the corresponding
Gemini function declaration.

This allows the LLM to understand the available MCP tools without the
application hard-coding a separate tool definition for each call.

------------------------------------------------------------------------

# 10. Question Routing

The Gemini router returns exactly one of:

``` text
RAG
MCP
BOTH
```

### RAG

Used for stable destination questions:

``` text
What are the major attractions in Singapore?
What food experiences should I try?
How can I travel around Singapore?
```

### MCP

Used for current information:

``` text
What is the current weather?
Convert 10000 INR to SGD.
```

### BOTH

Used when stable destination knowledge and current information are both
required:

``` text
Create a three-day Singapore itinerary for next week
and adjust it according to the weather forecast.
```

This prevents unnecessary use of MCP for questions that can be answered
from the static knowledge base.

------------------------------------------------------------------------

# 11. MCP Tool Selection

For MCP/BOTH requests, the application exposes discovered MCP tools to
Gemini.

Gemini selects the appropriate tool and supplies its arguments.

Examples:

``` text
Weather question
      ↓
get_weather
```

``` text
Currency question
      ↓
convert_currency
```

The application extracts the function name and arguments from Gemini's
function-call response and validates that the function exists among the
discovered MCP tools before execution.

------------------------------------------------------------------------

# 12. Combined RAG + MCP Scenario

The mandatory scenario is:

> **Create a three-day Singapore itinerary for next week and adjust it
> according to the weather forecast.**

The complete flow is:

``` text
User question
     ↓
Router
     ↓
BOTH
     ├───────────────┐
     ▼               ▼
RAG retrieval      MCP tool selection
     │               │
     │               ▼
     │          get_weather
     │               │
     │               ▼
     │          Open-Meteo
     │               │
     └───────┬───────┘
             ▼
       Final LLM prompt
             ↓
    Weather-aware itinerary
```

### RAG provides

-   attraction information
-   sample itinerary ideas
-   indoor/outdoor activities
-   transportation guidance
-   cultural and food experiences

### MCP provides

-   current/forecast weather data

### LLM provides

-   the final structured itinerary
-   recommendations based on the supplied information
-   indoor alternatives when weather makes outdoor activities less
    suitable

The final prompt explicitly distinguishes KB facts, MCP current
information and LLM recommendations.
-----------------------------------------------------------------------

# Overall Context Flow

```text
                         User Question
                               │
                               ▼
                      ┌─────────────────┐
                      │ Question Router │
                      └────────┬────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
               RAG            MCP            BOTH
                │              │              │
                ▼              ▼              ▼
           Retrieve KB      Select MCP      RAG + MCP
             Context          Tool          Context
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │     Context Assembly      │
                 │                           │
                 │ Conversation History      │
                 │ RAG Context               │
                 │ MCP Result                │
                 │ Current Question          │
                 │ Grounding Instructions    │
                 └──────────────┬────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │    Gemini LLM      │
                     │ Final Generation   │
                     └─────────┬──────────┘
                               │
                               ▼
                       Grounded Response
```

------------------------------------------------------------------------

# 13. Prompt Strategy

The final prompt in `assistant.py` contains grounding rules.

The important rules are:

1.  **MCP is the source of truth for current information.**
2.  **The KB is the source of truth for stable destination
    information.**
3.  **Do not invent unsupported facts.**
4.  **Recommendations may be generated by combining supplied KB and MCP
    information.**
5.  **Clearly distinguish facts, MCP information and recommendations.**
6.  **Use the supplied KB source titles and URLs.**
7.  **Do not overstate weather beyond the MCP result.**
8.  **If the KB is insufficient, say so.**
9.  **If an MCP tool fails, say current information could not be
    retrieved; do not guess.**
10. **Preserve relevant preferences from conversation history.**

This prompt strategy is designed to keep the final response grounded
while still allowing the LLM to produce useful travel recommendations.

------------------------------------------------------------------------

# 14. Multi-Turn Context

The Streamlit application stores conversation history in:

``` python
st.session_state.conversation_history
```

The history is passed to the assistant for subsequent questions.

Example:

``` text
User:
Plan a three-day Singapore trip for my family.

Assistant:
[three-day plan]

User:
Make Day 2 more suitable for rainy weather.

Assistant:
[updated Day 2 using the previous context]
```

The context mechanism allows follow-up questions to refer to previous
turns.

------------------------------------------------------------------------

# 15. Information Provenance

The application distinguishes three categories.

### Knowledge Base Facts

Retrieved from RAG:

-   attractions
-   neighbourhoods
-   general transportation
-   food/culture
-   itinerary ideas

### MCP Current Information

Retrieved dynamically:

-   weather
-   exchange-rate conversion

### LLM Recommendations

Generated from the supplied information.

For example:

``` text
Weather forecast
      +
Indoor attraction from KB
      ↓
LLM recommendation
```

The recommendation is not represented as though it were directly
retrieved from a source.

------------------------------------------------------------------------

# 16. Why RAG and MCP Are Separate

RAG and MCP solve different problems:

| RAG | MCP |
|---|---|
| Stable knowledge | Dynamic information |
| Local knowledge base | Runtime tools |
| Attractions | Weather |
| Culture | Currency conversion |
| Food | External API data |
| General transportation | Current values |
| Sample itineraries | Current forecast |

The separation makes the architecture easier to reason about and avoids treating changing information as permanent static knowledge.

---

# 17. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application language |
| LangChain | RAG, document processing and orchestration |
| Gemini | Question routing, tool selection and final response generation |
| Gemini Embeddings | Semantic document and query embeddings |
| FAISS | Vector similarity search |
| MCP | Dynamic tool integration |
| Open-Meteo | Weather data |
| Frankfurter | Currency conversion |
| Streamlit | User interface |
| python-dotenv | Environment configuration |
| Requests | External HTTP requests |

------------------------------------------------------------------------

# 18. Installation and Setup

## Prerequisites

-   Python installed
-   Internet access for Gemini, Open-Meteo and Frankfurter
-   Google Gemini API key

## Create virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
```

Activate:

``` powershell
.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation:

``` powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## Install dependencies

``` powershell
pip install -r requirements.txt
```

## Configure environment

Create `.env` in the project root:

``` env
GOOGLE_API_KEY=your_google_api_key
```

`.env.example` contains the placeholder configuration.

Do not commit the real `.env` file.

------------------------------------------------------------------------

# 19. Run the Application

From the project root:

``` powershell
streamlit run app/ui.py
```

Open the local URL displayed by Streamlit.

------------------------------------------------------------------------

# 20. Example Questions

### RAG

``` text
What are the major attractions in Singapore?
```

``` text
What food experiences should I try in Singapore?
```

``` text
How can I travel around Singapore?
```

Expected: RAG retrieval, grounded answer and source references.

### MCP Weather

``` text
What is the current weather forecast for Singapore?
```

Expected: weather tool selection, Open-Meteo data and explicit
indication that the current information came through MCP.

### MCP Currency

``` text
Convert 50000 INR to SGD.
```

Expected: currency tool selection and current conversion result from the
MCP tool.

### Combined

``` text
Create a three-day Singapore itinerary for next week
and adjust it according to the weather forecast.
```

Expected: BOTH routing, RAG retrieval, weather MCP call, weather-aware
itinerary, indoor/outdoor alternatives and source references.

------------------------------------------------------------------------

# 21. Failure Handling

## Insufficient knowledge

If the retrieved KB does not contain enough information, the assistant
is instructed to say so instead of inventing a destination fact.

## MCP failure

The weather/currency tools use request timeouts and HTTP status
validation.

If an external API fails:

``` text
External API
    ↓
request failure
    ↓
MCP tool exception
    ↓
MCP client/application
    ↓
user-facing failure message
```

The assistant is instructed not to guess current information.

## LLM/API quota failure

The LLM itself is an external dependency. If the configured Gemini API
reaches a quota limit or becomes temporarily unavailable, the
application surfaces the processing failure instead of fabricating an
answer.

------------------------------------------------------------------------

# 22. Testing Performed

The following flows were tested during development:

### RAG retrieval

Transportation-specific questions retrieve transportation-related KB
content.

Attraction and itinerary questions retrieve relevant KB chunks.

### Source references

Retrieved KB content can be returned with associated source title and
URL.

### Weather MCP

`get_weather` successfully communicates with Open-Meteo when the
required external services and LLM availability are present.

### Currency MCP

`convert_currency` successfully communicates with Frankfurter when the
external service and LLM availability are present.

### Combined RAG + MCP

The mandatory scenario has been exercised through the:

``` text
BOTH → RAG retrieval + get_weather → final answer
```

flow.

### Error handling

LLM/API failures are surfaced rather than replaced with invented current
information.

Conversation history is updated by the assistant layer so that the UI
remains usable after processing failures.

------------------------------------------------------------------------

# 23. Demo Scenarios

A short demonstration can be organized into five parts.

## Demo 1 --- RAG

Ask:

``` text
What are the main attractions and neighbourhoods I can visit in Singapore?
```

Demonstrate:

-   RAG route
-   semantic retrieval
-   grounded answer
-   source references

## Demo 2 --- Weather MCP

Ask:

``` text
What is the current weather forecast for Singapore?
```

Demonstrate:

-   MCP route
-   `get_weather`
-   dynamic external data
-   MCP source indication

## Demo 3 --- Currency MCP

Ask:

``` text
Convert 50000 INR to SGD.
```

Demonstrate:

-   MCP route
-   `convert_currency`
-   tool arguments
-   structured result

## Demo 4 --- RAG + MCP

Ask:

``` text
Create a three-day Singapore itinerary for next week
and adjust it according to the weather forecast.
```

Demonstrate:

-   BOTH route
-   RAG retrieval
-   weather MCP call
-   weather-aware itinerary
-   indoor/outdoor alternatives
-   sources

## Demo 5 --- Multi-turn context

Ask:

``` text
Plan a three-day Singapore trip for my family.
```

Then:

``` text
Make Day 2 more suitable for rainy weather.
```

Demonstrate retained conversation context.

------------------------------------------------------------------------

# 24. Security Considerations

-   API keys are stored in `.env`.
-   `.env` is excluded from Git.
-   `.env.example` contains only a placeholder.
-   `.venv` is excluded from Git.
-   `__pycache__` is excluded from Git.
-   External HTTP requests use timeouts.
-   MCP tools are selected from discovered tool definitions.
-   Unknown tool names are not executed.
-   Current information is not fabricated when an external service
    fails.

------------------------------------------------------------------------

# 25. Limitations

This is an assignment-scale travel assistant with a deliberately focused
scope.

### Destination scope

The static knowledge base focuses on Singapore.

### Static knowledge

The RAG system uses the local KB and is not a general live web-search
system.

### Weather

Weather depends on the availability of the Open-Meteo service and the
requested forecast period.

### Currency

Currency conversion depends on the Frankfurter service and its available
exchange-rate data.

### Transportation

The KB provides general transportation guidance rather than live routes,
live fares or live service status.

### Vector-store persistence

FAISS is created from the knowledge-base documents when the RAG module
initializes. The implementation prioritizes simplicity for this
assignment rather than introducing a separate persistent vector-database
service.

------------------------------------------------------------------------

# 26. Repository Hygiene

The repository should contain source code and required project
resources, but not generated/runtime files.

Do not commit:

``` text
.venv/
__pycache__/
*.pyc
.env
```

These are covered by `.gitignore`.

Commit:

``` text
app/
knowledge_base/
mcp_server/
.env.example
.gitignore
requirements.txt
README.md
```

------------------------------------------------------------------------

# 27. Submission Checklist

Before submitting:

-   [ ] Repository contains application source code.
-   [ ] Knowledge base contains at least three resources.
-   [ ] KB covers attractions/neighbourhoods.
-   [ ] KB covers transportation.
-   [ ] KB covers culture/practical tips.
-   [ ] KB covers food/local experiences.
-   [ ] KB contains sample itineraries.
-   [ ] KB contains indoor/outdoor activities.
-   [ ] Each KB resource has source metadata.
-   [ ] Documents are meaningfully chunked.
-   [ ] Embeddings are generated.
-   [ ] FAISS vector store is used.
-   [ ] Semantic retrieval works.
-   [ ] Source title/link is included in grounded responses.
-   [ ] Custom MCP server is included.
-   [ ] MCP exposes at least two tools.
-   [ ] Weather tool is implemented.
-   [ ] Currency tool is implemented.
-   [ ] AI selects the appropriate MCP tool.
-   [ ] Tool arguments are passed correctly.
-   [ ] RAG + MCP combined scenario is implemented.
-   [ ] Multi-turn context is maintained.
-   [ ] Missing KB information is handled without fabrication.
-   [ ] MCP failures are handled without fabrication.
-   [ ] Streamlit UI works.
-   [ ] `requirements.txt` is present.
-   [ ] `.env.example` is present.
-   [ ] `.env` is not committed.
-   [ ] `.venv` is not committed.
-   [ ] `__pycache__` is not committed.
-   [ ] README contains setup and run instructions.
-   [ ] Demo scenarios are prepared.

------------------------------------------------------------------------

# 28. Quick Start

``` powershell
git clone <repository-url>
cd ai-travel-planning-assistant

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create `.env`:

``` env
GOOGLE_API_KEY=your_google_api_key
```

Run:

``` powershell
streamlit run app/ui.py
```

Then try:

``` text
What are the major attractions in Singapore?
```

``` text
What is the current weather in Singapore?
```

``` text
Convert 50000 INR to SGD.
```

Finally run the mandatory scenario:

``` text
Create a three-day Singapore itinerary for next week
and adjust it according to the weather forecast.
```

------------------------------------------------------------------------

# 29. Conclusion

This project demonstrates an AI travel assistant architecture combining:

``` text
RAG
↓
Stable Singapore destination knowledge

MCP
↓
Current weather and currency information

Gemini
↓
Question routing + MCP tool selection + final generation

Streamlit
↓
Conversational interface
```

The central design principle is:

> **Use RAG for stable destination knowledge, MCP for dynamic
> information, and both when a request requires both types of
> information.**

The implementation demonstrates the required concepts of semantic
retrieval, grounded generation, source references, MCP tool discovery
and selection, dynamic API access, combined RAG + MCP workflows,
conversation context, and graceful failure handling.
