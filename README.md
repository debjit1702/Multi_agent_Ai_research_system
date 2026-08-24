# Multi-Agent AI Research System

A powerful, modular research automation platform that leverages multiple specialized AI agents to conduct end-to-end research workflows. The system orchestrates a coordinated pipeline of agents to search, extract, synthesize, and validate information, delivering high-quality research reports with minimal human intervention.

## ✨ Overview

The Multi-Agent AI Research System automates the entire research lifecycle by delegating tasks to specialized agents that work in sequence:

1. **Search Agent** — Discovers relevant sources using web search
2. **Reader Agent** — Scrapes and extracts content from top-ranked sources
3. **Writer Agent** — Synthesizes findings into a coherent report
4. **Critic Agent** — Reviews and validates the report for accuracy and quality

The Streamlit UI provides an intuitive interface for running research queries and viewing results, with full transparency into each pipeline stage.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API key for one of: **Groq** (recommended), **xAI/Grok**, or another LLM provider
- Virtual environment (recommended)

### Installation

1. **Clone and navigate to the project directory:**
   ```bash
   cd "multi agent system"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   # OR
   XAI_API_KEY=your_xai_api_key_here
   ```

### Run the Application

#### Standard Execution

Start the Streamlit web interface:
```bash
python -m streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

#### Using the Python Module (Windows)

If the above doesn't work, try the module-based approach:
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

#### Custom Port

To run on a different port:
```bash
python -m streamlit run app.py --server.port 8000
```

#### Headless Mode (No Browser Auto-Open)

```bash
python -m streamlit run app.py --server.headless true
```

#### Running via Direct Python

```bash
python -m streamlit run app.py
```

### Using the Application

1. **Enter a research topic** — Type your research query in the input field (e.g., "Latest developments in AI safety")
2. **Click "Run pipeline"** — The system will execute all 4 stages sequentially
3. **View progress** — A status box shows real-time progress through each agent
4. **Review results** — Once complete, view:
   - **Report** — The final synthesized research report
   - **Critic Feedback** — Quality review and validation notes
   - **Search Results** — Raw search engine output
   - **Scraped Content** — Full text from the top source
5. **Download report** — Click the download button to save as Markdown
6. **Access history** — Previous research runs appear in the left sidebar for quick access

### Troubleshooting Startup Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install streamlit` |
| Port 8501 already in use | Use `--server.port 8000` or kill the existing process |
| `.env` file not found | Create `.env` with `GROQ_API_KEY=your_key` |
| "API key not configured" | Verify your API key in `.env` and reload the app |
| Slow startup | First run may take longer; subsequent runs are faster |

## 🏗️ Detailed Architecture

### System Overview

The system is built on a **modular, agent-based architecture** using LangChain for orchestration and specialized agents for each research phase. Each agent is equipped with specific tools and prompts to complete its task effectively.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                        │
│  • Research form, progress tracking, result tabs, history       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               Pipeline Orchestrator (pipeline.py)                │
│  • State management, sequential workflow, message passing        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬───────────────┐
        ▼             ▼             ▼               ▼
   ┌────────┐   ┌────────┐   ┌──────────┐   ┌───────────┐
   │ Search │   │ Reader │   │ Writer   │   │ Critic    │
   │ Agent  │   │ Agent  │   │ Chain    │   │ Chain     │
   └───┬────┘   └───┬────┘   └──────────┘   └───────────┘
       │            │
   [Tools Layer]    │
   ┌───┴────────────┴──────────────────────────────────┐
   │  • web_search (Tavily API)                        │
   │  • scrape_url (BeautifulSoup)                     │
   └─────────────────────────────────────────────────────┘
       │
   [LLM Layer]
   ┌─────────────────────────────────────────────────────┐
   │  • Groq (llama-3.3-70b-versatile) [Preferred]       │
   │  • xAI Grok (grok-4.5) [Fallback]                   │
   └─────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **LLM Layer** (`agents.py` - `_build_llm()`)

Automatic LLM selection with fallback:
- **Primary**: Groq LLaMA 3.3 (70B) — Fast, free tier, optimized for reasoning
- **Fallback**: xAI Grok 4.5 — Advanced multi-modal reasoning
- **Configuration**: Temperature = 0 (deterministic output)
- **API Key Detection**: Checks environment for `GROQ_API_KEY` or `XAI_API_KEY`

```python
# LLM is instantiated once and reused across all agents
llm = _build_llm()  # Raises RuntimeError if no valid API key found
```

#### 2. **Agent Layer** (Agents in `agents.py`)

Four specialized agents, each with distinct roles:

##### **Search Agent** (`build_search_agent()`)
- **Tool**: `web_search` (Tavily API)
- **Purpose**: Discover relevant sources and initial information
- **Input**: Research topic from user
- **Output**: 5 ranked search results with titles, URLs, and snippets (300 chars each)
- **LangChain Type**: ReAct agent (Reasoning + Acting)
- **Prompt**: "Find recent, reliable and detailed information about: {topic}"

##### **Reader Agent** (`build_reader_agent()`)
- **Tool**: `scrape_url` (BeautifulSoup)
- **Purpose**: Extract deep content from top-ranked source
- **Input**: Search results + topic context
- **Output**: Clean text content from best URL (3000 chars max)
- **LangChain Type**: ReAct agent
- **Prompt**: "Pick the most relevant URL and scrape it for deeper content"
- **Processing**: Removes scripts, styles, nav, footer tags; normalizes whitespace

##### **Writer Chain** (`writer_chain`)
- **Type**: LCEL (LangChain Expression Language) pipeline
- **Purpose**: Synthesize information into structured report
- **Input**: Combined search results + scraped content + topic
- **Output**: Formatted report with Introduction, Key Findings (min 3), Conclusion, Sources
- **System Prompt**: "You are an expert research writer. Write clear, structured and insightful reports."
- **Template**: ChatPromptTemplate with structured report format

##### **Critic Chain** (`critic_chain`)
- **Type**: LCEL pipeline
- **Purpose**: Review and validate report quality
- **Input**: Generated report
- **Output**: Structured feedback (Score/10, Strengths, Areas to Improve, Verdict)
- **System Prompt**: "You are a sharp and constructive research critic. Be honest and specific."
- **Scoring**: 1-10 numeric score + detailed feedback

#### 3. **Tools Layer** (`tools.py`)

##### **web_search(query: str) → str**
- **Provider**: Tavily (research API)
- **Functionality**: Searches web for relevant, up-to-date information
- **Returns**: JSON with 5 results containing: title, URL, content snippet
- **Error Handling**: Validates non-empty query, catches API failures
- **Output Format**: Formatted text with title, URL, snippet for each result

##### **scrape_url(url: str) → str**
- **Provider**: BeautifulSoup + requests
- **Functionality**: Extracts clean text from HTML pages
- **Processing Steps**:
  1. HTTP GET request with Mozilla User-Agent
  2. HTML parsing with BeautifulSoup
  3. Remove noise: `<script>`, `<style>`, `<nav>`, `<footer>` tags
  4. Extract normalized text (3000 char limit)
- **Timeout**: 8 seconds per request
- **Error Handling**: Returns error message if scraping fails

#### 4. **Pipeline Orchestrator** (`pipeline.py` - `run_research_pipeline()`)

Sequential workflow with state management:

```python
def run_research_pipeline(topic: str) -> dict:
    state = {}
    
    # Step 1: Search
    search_agent = build_search_agent()
    search_result = search_agent.invoke({"messages": [prompt]})
    state['search_results'] = extract_content(search_result)
    
    # Step 2: Read
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({"messages": [prompt]})
    state['scraped_content'] = extract_content(reader_result)
    
    # Step 3: Write
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": search_results + scraped_content
    })
    
    # Step 4: Critique
    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })
    
    return state
```

**Key Features**:
- **Sequential Execution**: Each step waits for the previous to complete
- **State Management**: Dictionary tracks outputs from each stage
- **Message Passing**: Uses LangChain's `HumanMessage` format
- **Content Extraction**: `_last_message_content()` helper extracts AI responses
- **Error Propagation**: Exceptions bubble up to UI for user feedback
- **Logging**: Print statements for debugging (console output visible in terminal)

#### 5. **UI Layer** (`app.py` - Streamlit)

**Architecture**:
- **Framework**: Streamlit (Python web framework)
- **Styling**: Custom CSS with dark theme (CSS-in-Markdown)
- **State Management**: Streamlit session state for history tracking
- **Layout**: Sidebar (history) + main area (hero + form + results)

**Key Sections**:
1. **Hero Header**: Branding, system description, pipeline steps
2. **Input Form**: Text field for research topic + submit button
3. **Status Box**: Shows 4-step progress (though not live-streamed)
4. **Results Tabs**:
   - Report: Final markdown report
   - Critic Feedback: Quality review
   - Search Results: Raw API output
   - Scraped Content: Full extracted text
5. **History Sidebar**: Quick access to previous runs

**Features**:
- Download reports as Markdown
- Search history with 38-char truncation
- Clear history button
- Responsive design (wide layout)
- Real-time status updates

### Data Flow Sequence

```
User Input (Topic)
    ↓
[LLM Init] → Load API key, instantiate Groq/xAI
    ↓
[Search Agent] → web_search(topic) → 5 URLs + snippets
    ↓
state['search_results'] ← Formatted search output
    ↓
[Reader Agent] → scrape_url(best_url) → Full page text
    ↓
state['scraped_content'] ← Cleaned HTML content
    ↓
[Writer Chain] → Combine search + scraped + topic → Structured report
    ↓
state['report'] ← Markdown-formatted research report
    ↓
[Critic Chain] → Evaluate report → Score + feedback
    ↓
state['feedback'] ← Structured critique
    ↓
Return state dict → Display in Streamlit UI
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (primary), xAI Grok (fallback) | Language model inference |
| **Orchestration** | LangChain (agents, chains, tools) | Agent coordination & prompting |
| **Web Search** | Tavily API | Information retrieval |
| **Content Scraping** | BeautifulSoup, requests | HTML parsing |
| **UI Framework** | Streamlit | Web interface |
| **Environment** | python-dotenv | API key management |
| **HTTP** | requests | Web requests |
| **HTML Parser** | BeautifulSoup 4 | DOM manipulation |
| **Dependencies** | See requirements.txt | Full stack |

### Error Handling & Resilience

1. **Missing API Keys**: Raise `RuntimeError` with helpful message directing to `.env`
2. **Empty Queries**: Validate non-empty strings at tool entry points
3. **Network Failures**: Catch exceptions in `scrape_url`, return error message
4. **Invalid Input**: `run_research_pipeline()` validates topic before starting
5. **LLM Fallback**: Tries Groq first, falls back to xAI if key missing
6. **UI Error Boundary**: Streamlit catches exceptions, displays error box with details

### Performance Characteristics

- **End-to-End Runtime**: 30-60 seconds (varies by model + network)
- **Search Results**: Limited to 5 per API call (reduce rate limit impact)
- **Scrape Limit**: 3000 characters (balance depth vs. token cost)
- **Temperature**: 0 (deterministic, reproducible output)
- **Concurrency**: Sequential (no parallel agent execution currently)
- **Caching**: None (fresh data on every run)

## 📁 Project Structure

```
.
├── app.py                 # Streamlit UI for the research pipeline
├── pipeline.py            # Main orchestration logic (4-step workflow)
├── agents.py              # AI agent definitions (Search, Reader, Writer, Critic)
├── tools.py               # Tool implementations (web search, URL scraping)
├── requirements.txt       # Python dependencies
├── test_pipeline.py       # Test suite
└── README.md              # This file
```

### Key Files

- **app.py** — Streamlit frontend with dark theme, research form, and result tabs
  - Displays live pipeline progress
  - Shows report, critic feedback, search results, and scraped content
  - Maintains research history for quick access
  - Supports markdown download of final reports

- **pipeline.py** — Orchestrates the 4-agent workflow
  - Sequential execution: Search → Read → Write → Critique
  - State management and message passing between agents
  - Error handling and result extraction

- **agents.py** — LLM agent configurations using LangChain
  - Search agent: web search and result ranking
  - Reader agent: content extraction and summarization
  - Writer chain: report generation
  - Critic chain: quality review and validation
  - Automatic LLM selection (Groq or xAI) based on API keys

- **tools.py** — Utility functions
  - Web search integration (Tavily API)
  - URL content scraping (BeautifulSoup)
  - HTML/text parsing and cleanup

## ⚙️ Configuration

### LLM Models

The system automatically selects an LLM based on available API keys (in order of preference):

1. **Groq** (llama-3.3-70b-versatile) — Fast, free tier available
2. **xAI/Grok** (grok-4.5) — Advanced reasoning capabilities

Set the corresponding API key in `.env` to use each provider.

### Temperature & Parameters

- **Temperature**: 0 (deterministic output for consistent reports)
- **Model Selection**: Automatic based on available credentials

## 🎯 Features

- **End-to-End Pipeline** — Fully automated research workflow with 4 specialized agents
- **Web-Based UI** — Modern Streamlit interface with dark theme and responsive design
- **Result Transparency** — View search results, scraped content, and critic feedback
- **Research History** — Track and revisit previous research runs
- **Report Export** — Download reports as Markdown files
- **Error Handling** — Graceful error recovery and user feedback
- **Extensible Architecture** — Easy to add new agents or tools

## 📊 Pipeline Workflow

```
User Input (Research Topic)
    ↓
[1] Search Agent
    • Queries web for relevant sources
    • Returns ranked list of results
    ↓
[2] Reader Agent
    • Scrapes top-ranked source
    • Extracts and summarizes content
    ↓
[3] Writer Agent
    • Synthesizes information into structured report
    • Generates insights and recommendations
    ↓
[4] Critic Agent
    • Reviews report for accuracy and coherence
    • Provides feedback and validation
    ↓
Output (Report + Feedback)
    • Displayed in Streamlit UI
    • Available for download
```

## 💡 Use Cases

- **Academic Research** — Literature review and synthesis
- **Technical Discovery** — Learning about new technologies and frameworks
- **Business Intelligence** — Market analysis and competitive research
- **Product Research** — Feature discovery and trend analysis
- **News & Trends** — Current events synthesis and summary
- **Knowledge Work** — Rapid information gathering and organization

## 🔧 Development

### Running Tests

```bash
python -m pytest test_pipeline.py -v
```

### Adding Custom Agents

1. Define a new agent function in `agents.py`
2. Integrate into the pipeline step in `pipeline.py`
3. Update the UI in `app.py` if needed

### Adding New Tools

1. Implement tool functions in `tools.py`
2. Reference in agent prompts/configurations
3. Test in isolation before integration

## 🐛 Troubleshooting

**"API key not found"** — Ensure `.env` file exists with `GROQ_API_KEY` or `XAI_API_KEY`

**"langchain-groq not installed"** — Run `pip install langchain-groq`

**Streamlit not starting** — Verify Streamlit is installed: `pip install streamlit`

**Empty search results** — Check internet connection and API rate limits

## 📈 Performance Considerations

- Total pipeline runtime: ~30-60 seconds (varies by model and network)
- Search results limited to reduce API calls
- Content scraping is single-source (top result)
- Temperature set to 0 for consistent output

## 🚧 Future Enhancements

- **Multi-source Synthesis** — Combine insights from multiple sources
- **Fact Verification** — Cross-check claims across sources
- **Domain Specialization** — Custom agents for specific research domains
- **Memory & Context** — Persistent context for follow-up queries
- **Advanced Export** — PDF, JSON, and structured data formats
- **Batch Processing** — Research multiple topics in sequence
- **Agent Customization** — User-configurable agent behavior and prompts
- **Caching** — Store results for repeated queries

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
