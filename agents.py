import os

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from tools import web_search, scrape_url

load_dotenv()

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

try:
    from langchain_xai import ChatXAI
except ImportError:  # pragma: no cover
    ChatXAI = None


def _build_llm():
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if groq_key:
        if ChatGroq is None:
            raise RuntimeError("Groq support is not installed. Install langchain-groq in the environment.")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0,
        )

    xai_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("xai_api_key")
    if xai_key:
        if ChatXAI is None:
            raise RuntimeError("xAI support is not installed. Install langchain-xai in the environment.")
        return ChatXAI(
            model="grok-4.5",
            api_key=xai_key,
            temperature=0,
        )

    raise RuntimeError(
        "Missing API key. Add GROQ_API_KEY or XAI_API_KEY to your .env file before running the pipeline."
    )


#model_setup
llm = _build_llm()

#list agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )

#2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


#write_chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()