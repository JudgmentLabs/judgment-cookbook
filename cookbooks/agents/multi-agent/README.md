# Multi-Agent Research System

A coordinated AI research team that works together to create comprehensive reports. A **Lead Agent** manages the project while **Research Agents** work in parallel to gather information and store findings in a persistent database.

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd mock-agent
pipenv install
pipenv shell
```

For advanced observability and tracing features:

```bash
pip install judgeval
```

📖 **Learn more**: [Judgment Labs Documentation](https://docs.judgmentlabs.ai/quickstarts)

### Set API Keys
```bash
export JUDGMENT_API_KEY='your-key-here'
export JUDGMENT_ORG_ID='your-org-id-here'
export OPENAI_API_KEY='your-key-here'
export TAVILY_API_KEY='your-key-here'
```

## 🎬 Demo First

```bash
python main.py
```
Run this to see the system in action with example research projects!

## 📊 Real-Time Monitoring

### Database Monitoring
Monitor your ChromaDB in real-time to see documents being added and research progress:

```bash
# Real-time database monitoring with live operations feed
python monitor_db.py

# Simple mode (just count changes, cleaner output)
python monitor_db.py --simple

# Detailed mode with document previews
python monitor_db.py --detailed
```

**What you'll see:**
```
[14:30:15] 📝 3 document(s) added (Total: 1,247)
[14:30:15]   ↳ New document (14:30:14): ID=abc12345, Size=156 chars
[14:30:20] 🔍 Status query: 1,247 documents, 0.045s
```

### Multi-Window Agent Tracking
When agents work in parallel, each gets its own dedicated window for real-time progress tracking:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    AGENT 1      │  │    AGENT 2      │  │    AGENT 3      │
│  Regulation     │  │  Supply Chain   │  │  Corporate      │
│  Research       │  │  Analysis       │  │  Impact         │
│                 │  │                 │  │                 │
│ 🔄 Processing   │  │ 🔍 Web search   │  │ 💾 Storing data │
│ EU regulation   │  │ for palm oil    │  │ for Unilever    │
│ details...      │  │ suppliers...    │  │ compliance...   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                          LEAD AGENT                          │
│                       (Senior Manager)                       │
│                                                              │
│  1. Background Research → 2. Plan & Delegate                 │
│  3. Collect & Synthesize → 4. Create Final Report            │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              PARALLEL RESEARCH AGENTS                   │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │  AGENT 1    │  │  AGENT 2    │  │  AGENT 3    │      │ │
│  │  │Regulation   │  │Supply Chain │  │Corporate    │      │ │
│  │  │Research     │  │Analysis     │  │Impact       │      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │ │
│  │         │                │                │             │ │
│  │         ▼                ▼                ▼             │ │
│  │  ┌─────────────────────────────────────────────────────┐│ │
│  │  │          PERSISTENT DATABASE                        ││ │
│  │  │        (Research Findings)                          ││ │
│  │  └─────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│                  Final Document (.docx)                      │
└──────────────────────────────────────────────────────────────┘

## 🛠️ Agent Toolbox

### Database Tools
- `get_database()` - Retrieve stored research findings
- `put_database()` - Store findings with citations

### Web Tools
- `web_search()` - Comprehensive web research
- `web_extract_content()` - Extract content from websites
- `web_extract_images()` - Collect relevant images
- `calculate()` - Mathematical analysis
- `delegate()` - Assign task to single research agent
- `delegate_multiple()` - Run multiple agents in parallel

### Document Tools
- `create_docx()` - Create new documents
- `add_text_to_docx()` - Add formatted content
- `add_heading_to_docx()` - Professional headings
- `add_table_to_docx()` - Data tables
- `add_image_to_docx()` - Embed images
- `delete_text_from_docx()` - Content management

## 🎯 Using the System

```python
from src.agents.lead_agent import LeadAgent

lead_agent = LeadAgent()

# Ask a complex research question
result = lead_agent.process_request(
    "Research the 2024 EU deforestation regulation impact on palm oil imports "
    "from Indonesia and Malaysia, including compliance costs for companies "
    "like Unilever and Nestlé. Create a comprehensive report."
)
```

## ⚡ What Happens

1. **Lead Agent** does background research to understand the topic
2. **Delegates** specific tasks to multiple research agents that run **in parallel**:
   - Agent 1: Research regulation details
   - Agent 2: Analyze supply chain impacts  
   - Agent 3: Investigation corporate compliance costs
3. **Research Agents** conduct deep research and store findings in database
4. **Lead Agent** collects all findings, synthesizes the information
5. **Creates professional DOCX report** with all findings, data, and citations

**Monitor the process:**
- 📊 **Database Monitor**: See documents being added in real-time
- 🪟 **Agent Windows**: Watch each agent's progress in separate windows
- 📄 **Final Output**: Professional DOCX report in `reports/` directory

## 📁 File Structure

```
mock-agent/
├── main.py                 # 🎬 Demo entry point
├── monitor_db.py          # 📊 Database monitoring tool
├── reports/               # 📄 Generated DOCX reports
├── memory_db/            # 💾 Persistent ChromaDB database
├── src/
│   ├── agents/
│   │   ├── lead_agent.py     # Senior Manager Agent
│   │   ├── agent.py         # Research Agent
│   │   └── base.py          # Shared functionality
│   └── tools/
│       ├── operations.py    # Delegation & coordination
│       ├── database.py      # Persistent memory
│       ├── docx_tools.py    # Document generation
│       ├── web.py          # Research tools
│       └── window_logger.py # Multi-window display
└── Pipfile               # Dependencies
```