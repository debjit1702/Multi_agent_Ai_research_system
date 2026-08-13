from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

def run_research_piprline(topic : str) -> dict:
    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    state['search_results'] = search_result ['messages'] [-1] ['content']

    print("\n search result", state['search_results'])

    #step 2 - reader agent working
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    resder_agent = build_reader_agent()
    reader_agent.invoke()