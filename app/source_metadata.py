"""
Source metadata for the Singapore Knowledge Base.

Each Knowledge Base Markdown file is mapped to the public sources
used to create its content.

This metadata is attached to LangChain documents during the RAG
ingestion process so that retrieved information can be traced back
to its source and displayed in the final answer.
"""


SOURCE_METADATA = {
    # ------------------------------------------------------------
    # Attractions and neighbourhoods
    # ------------------------------------------------------------
    "attractions.md": [
        {
            "title": "Wikivoyage — Singapore Travel Guide",
            "url": "https://en.wikivoyage.org/wiki/Singapore",
        },
        {
            "title": "Visit Singapore — Things to Do",
            "url": "https://www.visitsingapore.com/",
        },
    ],

    # ------------------------------------------------------------
    # Culture, food and practical travel information
    # ------------------------------------------------------------
    "cultural_food.md": [
        {
            "title": "Visit Singapore — Essential Travel Information",
            "url": "https://www.visitsingapore.com/travel-tips/essential-travel-information/",
        },
        {
            "title": "Wikivoyage — Singapore Travel Guide",
            "url": "https://en.wikivoyage.org/wiki/Singapore",
        },
    ],

    # ------------------------------------------------------------
    # Sample itineraries
    # ------------------------------------------------------------
    "itineraries.md": [
        {
            "title": "Visit Singapore — Enjoy Singapore in 7 Days",
            "url": "https://www.visitsingapore.com/content/visitsingapore/en/travel-tips/travelling-to-singapore/itineraries/7-days-in-singapore",
        },
        {
            "title": "Visit Singapore — Singapore Itineraries",
            "url": "https://www.visitsingapore.com/singapore-itineraries/",
        },
        {
            "title": "Visit Singapore — Travel Itineraries",
            "url": "https://www.visitsingapore.com/",
        },
    ],

    # ------------------------------------------------------------
    # Transportation
    # ------------------------------------------------------------
    "transportation.md": [
        {
            "title": "Visit Singapore — Essential Travel Information",
            "url": "https://www.visitsingapore.com/travel-tips/essential-travel-information/",
        },
        {
            "title": "Wikivoyage — Singapore Travel Guide",
            "url": "https://en.wikivoyage.org/wiki/Singapore",
        },
    ],
}