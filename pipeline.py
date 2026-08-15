from langchain_core.messages import HumanMessage

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def _last_message_content(result: object) -> str:
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]
    if hasattr(last_message, "content"):
        content = last_message.content
    elif isinstance(last_message, dict):
        content = last_message.get("content", "")
    else:
        content = str(last_message)

    return content if isinstance(content, str) else str(content)


def run_research_pipeline(topic: str) -> dict:
    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    state = {}

    #search agent working 
    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_prompt = HumanMessage(
        content=f"Find recent, reliable and detailed information about: {topic}"
    )
    search_result = search_agent.invoke({"messages": [search_prompt]})
    state['search_results'] = _last_message_content(search_result)

    print("\n search result", state['search_results'])

    #step 2 - reader agent working
    print("\n" + " =" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_prompt = HumanMessage(
        content=(
            f"Based on the following search result about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content. \n\n"
            f"Search results: \n{state['search_results'][:800]}"
        )
    )
    reader_result = reader_agent.invoke({"messages": [reader_prompt]})
    state['scraped_content'] = _last_message_content(reader_result)

    print("\nScraped_content\n", state['scraped_content'])

    #step -3 Writer Chain
    print("\n" + " =" * 50)
    print("step 3 - Writer is Drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })

    print("\n Final Report\n", state['report'])

    #critic_report
    print("\n" + " =" * 50)
    print("step 4 - Critic is reviewing the report ...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state['report'],
    })

    print("\n critic report \n", state['feedback'])

    return state


if __name__ == "__main__":
    try:
        topic = input("\n Enter a research topic : ").strip()
        run_research_pipeline(topic)
    except ValueError as exc:
        print(f"\n Invalid input: {exc}")
    except Exception as exc:
        print(f"\n Research pipeline failed: {exc}")
