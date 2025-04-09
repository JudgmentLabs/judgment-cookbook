from openai import OpenAI, Client
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer
import os

client = wrap(Client(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="chatbot_RAG_agent_demo")

# Set your API key

# -------------------------------
# Mock knowledge base
# -------------------------------
doc1 = [
    "ChatGPT is increasingly being adopted in the finance sector to streamline customer service operations.",
    "It provides round-the-clock support to answer basic queries about accounts, transactions, and policies.",
    "Financial institutions use ChatGPT to reduce the burden on human agents during high-volume periods.",
    "The AI is trained to handle common queries like loan application status or interest rate changes.",
    "It can integrate into mobile banking apps to provide conversational assistance to users.",
    "ChatGPT also assists in document analysis, helping staff summarize long financial reports.",
    "Finance professionals use it to create initial drafts of investment memos and quarterly summaries.",
    "The AI's ability to interpret natural language queries allows analysts to find data faster.",
    "It supports onboarding processes by answering policy-related questions from new clients or employees.",
    "In internal operations, ChatGPT is used to explain company expense policies to employees.",
    "Some teams use it to generate synthetic financial data for model testing and simulations.",
    "It also plays a role in internal training, simulating customer queries for practice sessions.",
    "ChatGPT can rewrite compliance documents into simpler language for better understanding.",
    "Risk teams use it to help interpret the implications of stress test results.",
    "It’s embedded into chat platforms like Slack or Microsoft Teams for instant knowledge access.",
    "Finance managers use ChatGPT to create meeting agendas based on recent financial highlights.",
    "It helps accountants check consistency in tax forms and other regulatory documents.",
    "Auditing teams use it to prepare summaries from large sets of transactional logs.",
    "It’s used to assist in ESG (Environmental, Social, Governance) reporting by reformatting text and explanations.",
    "ChatGPT reduces the effort required for reconciliation by highlighting mismatches in reports.",
    "Wealth management firms use it to personalize financial planning advice based on client preferences.",
    "It drafts client newsletters with market updates and firm-specific investment commentary.",
    "Teams use it to monitor and summarize daily financial news and alert stakeholders to key events.",
    "ChatGPT helps in contract review by identifying important clauses and suggesting summaries.",
    "Legal-financial departments use it to draft initial responses to regulatory inquiries.",
    "It can write documentation for internal finance tools, reducing the support load on dev teams.",
    "When connected to knowledge bases, ChatGPT acts as an expert assistant for complex queries.",
    "Finance students use ChatGPT to simulate interviews and explain financial models step-by-step.",
    "Firms use it to create knowledge quizzes for employee training on anti-money laundering (AML).",
    "It’s used to create call scripts for tele-support agents working in banking and credit sectors.",
    "Marketing departments in finance rely on ChatGPT to write campaign content targeting investors.",
    "It helps explain variable interest concepts to users applying for mortgage loans.",
    "ChatGPT can auto-fill report templates with numbers and language from structured data.",
    "During crises, it's used to create real-time FAQs for impacted customers.",
    "Portfolio managers use it to narrate changes in asset allocations for client transparency.",
    "The AI helps identify and flag inconsistencies in spreadsheet data entries.",
    "Banking teams use it to simulate customer objections and practice persuasive reasoning.",
    "It’s useful in summarizing sector performance in investment research documents.",
    "Consulting firms use it to create pitch decks for financial restructuring projects.",
    "Compliance teams prompt ChatGPT to test hypothetical fraud scenarios for audit planning.",
    "It helps in writing business continuity plans by templating and formatting the response workflows.",
    "Finance-focused YouTubers use it to generate video scripts about budgeting and investing.",
    "Financial educators use ChatGPT to make lesson plans and quizzes on credit and debt.",
    "It supports multilingual support for global banking operations and client onboarding.",
    "HR teams in finance use ChatGPT to create role descriptions and job advertisements.",
    "It explains investment terms like beta, alpha, and Sharpe ratio in conversational ways.",
    "Executives use it to help articulate the financial vision of the company in internal memos.",
    "Analysts use it to benchmark financial KPIs by querying performance metrics across time.",
    "ChatGPT drafts FAQs for new product launches in digital banking services.",
    "Loan officers use it to explain approval conditions to applicants in simple terms.",
    "It helps in writing follow-up emails post-client meetings with summarized action points.",
    "ChatGPT is often used to generate internal surveys regarding finance team efficiency.",
    "It can convert client meeting transcripts into concise summaries for CRM systems.",
    "Firms use it to explain product risks in a clearer manner for better disclosure.",
    "It can help advisors prep for calls by summarizing client portfolios and notes.",
    "Financial blogs leverage ChatGPT to draft engaging yet informative posts.",
    "It assists in comparative analysis of financial products by highlighting key differentiators.",
    "Insurance companies use it to describe policy terms and claim processes to users.",
    "During mergers, it's used to help translate financial merger docs for general teams.",
    "Investor relations teams use it to prepare Q&A sheets for earnings calls.",
    "It’s useful for flagging ambiguous statements in reports needing legal review.",
    "Finance tool developers use ChatGPT to create API documentation and onboarding guides.",
    "It reformats raw CSV data into readable summaries with charts and explanations.",
    "AI support agents powered by ChatGPT escalate only exceptions, reducing human workload.",
    "It can be used to build financial scenario simulators by interpreting text inputs.",
    "Robo-advisory platforms use it for client communication personalization.",
    "It supports automated writing of term sheet summaries for venture capital evaluations.",
    "Mortgage lenders rely on it to explain amortization tables to non-technical applicants.",
    "ChatGPT assists in the due diligence process by extracting key metrics from reports.",
    "It can identify risks across markets based on sentiment analysis of news feeds.",
    "Hedge funds prototype trading strategy descriptions using ChatGPT for internal proposals.",
    "It writes documentation for compliance workflows in accounting software.",
    "Pension fund managers use it to narrate asset performance to boards and investors.",
    "Retail banks use ChatGPT to guide customers through self-service portal steps.",
    "Fintech apps use ChatGPT to offer budgeting tips and automatic money-saving suggestions.",
    "It helps convert regulatory rulebooks into searchable, question-answerable formats.",
    "Finance advisors use it to create pre-meeting prep notes with financial highlights.",
    "It improves productivity in finance teams by cutting down repetitive research time.",
    "In audit prep, it helps organize supporting documentation checklists.",
    "Government agencies use it to summarize grant reports and compliance filings.",
    "It can be used to track financial term usage trends across internal documentation.",
    "In private equity, it's used to analyze exit strategies and summarize due diligence.",
    "Procurement departments use it to track and explain budget allocations.",
    "It helps visualize financial model outputs by narrating assumptions and risks.",
    "It assists in answering investor platform questions during periods of market volatility.",
    "Finance teams use ChatGPT to understand new changes in accounting regulations.",
    "It offers explanations on tax implications based on provided transactional context.",
    "In blockchain finance, it helps explain smart contract mechanics to clients.",
    "Financial aid departments in universities use it to answer student loan queries.",
    "Insurance underwriters use it to communicate premium calculation details.",
    "Finance directors use ChatGPT to simplify board packet contents before review.",
    "It turns financial data tables into executive-ready bullet-point summaries.",
    "The AI assists in talent development by suggesting finance skill tracks to employees.",
    "It's used to create personas for simulation-based training in finance education.",
    "ChatGPT makes internal auditing more efficient by helping trace data inconsistencies.",
    "It can write detailed case studies on successful financial transformations.",
    "It drafts financial proposals and grant applications for non-profit orgs.",
    "Credit analysts use it to rephrase risk narratives for different report audiences.",
    "ChatGPT can help with translation of foreign tax codes into localized explanations.",
    "It aids in identifying redundancies in financial workflows for process improvement.",
    "Overall, ChatGPT is becoming a central utility across various functions in finance."
]
documents = [ "\n".join(doc1) ]

doc2 = [
    "BloombergGPT is a large-scale language model built specifically for financial applications.",
    "It is trained on Bloomberg’s proprietary financial data and vast historical corpora.",
    "Unlike general-purpose models, BloombergGPT understands financial jargon and terminology natively.",
    "Its dataset includes earnings transcripts, news articles, regulatory filings, and economic indicators.",
    "It can accurately identify financial entities like ticker symbols, company names, and asset classes.",
    "The model is optimized for tasks like sentiment analysis of market news and financial documents.",
    "It has been evaluated for financial question answering and performs better than open models on these tasks.",
    "BloombergGPT is designed to work seamlessly within Bloomberg’s software and services ecosystem.",
    "It can generate summaries of financial events like earnings calls and economic policy changes.",
    "The model supports tasks like credit risk evaluation through document analysis.",
    "Traders use BloombergGPT to quickly parse through analyst reports and identify action items.",
    "It helps convert natural language queries into Bloomberg Terminal-compatible search strings.",
    "BloombergGPT can classify financial documents based on compliance, regulatory scope, or sector.",
    "Its architecture builds on transformer models with added finetuning for domain accuracy.",
    "The model shows superior performance in finance-specific benchmarks such as FiQA and Headline Risk.",
    "It helps analysts keep up with macroeconomic shifts by summarizing central bank policy reports.",
    "The model recognizes abbreviations commonly used in finance like EBIT, CAGR, and EPS.",
    "BloombergGPT can flag discrepancies in earnings estimates versus analyst expectations.",
    "It allows natural language interaction with structured databases of historical stock performance.",
    "BloombergGPT can detect changes in sentiment across news articles about specific sectors.",
    "It provides explainability by highlighting which parts of text influenced a particular classification.",
    "The model supports multilingual input and can interpret financial content in several languages.",
    "It integrates with Bloomberg Query Language (BQL) to improve access to structured datasets.",
    "The model is capable of named entity linking across financial texts and market databases.",
    "Its output aligns with domain-specific formatting, like bullet points for investment memos.",
    "It’s used in summarizing SEC filings, especially 10-K and 8-K reports for clients.",
    "Portfolio managers use BloombergGPT to detect correlation between news sentiment and price moves.",
    "It excels at parsing dense legal and compliance texts in the finance world.",
    "Research analysts rely on it to auto-draft investment theses based on performance data.",
    "It can detect duplication or plagiarism in financial media content.",
    "BloombergGPT is used internally to improve content search on Bloomberg Terminal.",
    "Its performance has been fine-tuned to distinguish between literal and implied financial statements.",
    "It performs better than GPT-3.5 or GPT-4 on document-level finance classification.",
    "The model is tested regularly for robustness to outdated or shifting market language.",
    "It supports document clustering for thematic investment research.",
    "The model can assist in ESG risk evaluation by scanning company disclosures.",
    "It is also capable of summarizing litigation risks from court filings.",
    "BloombergGPT is used to power chat interfaces within Bloomberg apps for real-time guidance.",
    "The model respects time-based context, understanding financial timelines and forecasts.",
    "It can assist in parsing footnotes in financial statements, often overlooked by analysts.",
    "BloombergGPT is optimized for high precision and low hallucination in finance tasks.",
    "It offers significant improvements in financial relationship extraction.",
    "Traders use it to automate risk commentary during volatile markets.",
    "It provides context-aware comparisons between financial quarters or fiscal years.",
    "BloombergGPT offers support for indexing and tagging of financial videos and audio transcripts.",
    "It can translate analyst discussions into bullet-point insights.",
    "It identifies cause-effect relationships in financial market narratives.",
    "The model is capable of adjusting for ambiguity in regulatory interpretations.",
    "It helps parse industry-specific terms like 'mark-to-market' or 'duration risk.'",
    "It can assess sentiment trajectory over time in media coverage of an asset.",
    "BloombergGPT is being explored for applications in real-time alerts and anomaly detection.",
    "The model can serve as a co-pilot for compliance professionals reviewing documentation.",
    "It helps classify clients by risk tolerance based on onboarding documents.",
    "It supports auto-generation of company profiles using both structured and unstructured data.",
    "The model can convert text-based disclosures into structured datasets for BI tools.",
    "It allows scalable analysis of financial transcripts across entire industries.",
    "It has shown promise in summarizing bond prospectuses and municipal finance documents.",
    "It is integrated into Bloomberg tools for enhanced contextual search.",
    "BloombergGPT outperforms generic models on financial document summarization benchmarks.",
    "Its vocabulary includes finance-specific metrics, acronyms, and valuation terms.",
    "It helps identify red flags in IPO prospectuses or investor presentations.",
    "It supports internal knowledge management by tagging and organizing financial memos.",
    "It generates investment commentary by synthesizing global and regional market trends.",
    "The model is used to auto-tag news stories for easier filtering by subscribers.",
    "BloombergGPT can recommend related documents or content within large financial corpora.",
    "It is embedded in internal research tools to auto-suggest headlines for investment briefs.",
    "The model can create timelines of key events for ongoing market developments.",
    "It helps identify gaps in regulatory disclosures across peer companies.",
    "It summarizes trends in commodity reports, highlighting price drivers and forecasts.",
    "BloombergGPT interprets index methodology documents for fund managers.",
    "It can align textual financial analysis with underlying quantitative models.",
    "The model helps surface related documents by topic, timeframe, and market sector.",
    "It can write introductory text for quarterly investment presentations.",
    "BloombergGPT supports the generation of sectoral insights for macroeconomic modeling.",
    "It generates scenario summaries for stress testing and contingency planning.",
    "Its outputs support decision-makers in constructing model portfolios.",
    "It can spot inconsistencies in analyst predictions versus actual earnings.",
    "It translates regulatory policy changes into plain-English impact summaries.",
    "BloombergGPT offers region-specific financial term adaptation.",
    "It can forecast headline sentiment impact on specific sectors.",
    "It supports the identification of liquidity and solvency indicators from news.",
    "BloombergGPT auto-detects named risks (e.g., currency exposure, sanctions) in text.",
    "It powers document navigation with financial topic segmentation.",
    "It performs deep parsing of earnings tables to extract insights.",
    "The model predicts shifts in tone across central bank policy statements.",
    "BloombergGPT can generate M&A deal summaries with valuation comparisons.",
    "It assists tax professionals in summarizing IRS rulings and implications.",
    "It learns continuously from new finance documents being added to its ecosystem.",
    "It automatically generates outlines for sector research briefs.",
    "It integrates with dashboards to highlight sentiment or headline risks dynamically.",
    "BloombergGPT is trusted for regulatory compliance tools within institutional finance.",
    "It enhances financial document quality assurance by spotting formatting or logic errors.",
    "It bridges structured and unstructured financial content across Bloomberg apps.",
    "It sets a benchmark for financial LLMs by outperforming general models on key tasks.",
    "BloombergGPT continues to be refined with real-world financial use cases in mind."
]

documents.append("\n".join(doc2))

doc3 = [
    "ChatGPT is a versatile language model that can engage in open-ended dialogue across countless domains.",
    "It’s based on the transformer architecture and optimized for both understanding and generation of human language.",
    "ChatGPT can write stories, essays, technical documentation, scripts, and even poetry.",
    "Its wide-ranging training corpus enables it to understand cultural, technical, and conversational nuances.",
    "People use ChatGPT for brainstorming ideas — from startups to marketing slogans to book titles.",
    "It assists developers by explaining code, generating functions, and debugging errors interactively.",
    "Students use it to learn topics by asking follow-up questions until they understand a concept deeply.",
    "ChatGPT supports dozens of languages, making it accessible to a global user base.",
    "Its conversational memory allows it to maintain multi-turn context in a discussion.",
    "It adapts tone based on user input — casual, formal, technical, or empathetic.",
    "ChatGPT can write personalized emails, apologies, wedding vows, and professional bios.",
    "It helps creators develop outlines, character arcs, and plot twists for fiction.",
    "It can impersonate writing styles of famous authors for creative mimicry and fun.",
    "ChatGPT helps clarify philosophical ideas by breaking them into understandable parts.",
    "It can generate quizzes, flashcards, or mnemonic devices for learning reinforcement.",
    "The model is used in customer support as a first-responder chatbot in web apps.",
    "It assists in tutoring students in subjects like math, science, history, and language.",
    "It translates between languages and explains idiomatic or regional expressions.",
    "ChatGPT generates use-case examples for APIs, software libraries, or new features.",
    "It can function as a therapist simulator or journaling partner for reflective writing.",
    "ChatGPT helps people improve their resumes and tailor them to job applications.",
    "It can analyze writing for tone, clarity, bias, and even SEO optimization.",
    "People use it to learn how to cook, budget, travel, or start a side hustle.",
    "It writes scripts for YouTube videos, podcasts, explainer animations, and ads.",
    "ChatGPT can generate fictional dialogues, debates, or Socratic conversations.",
    "It helps teachers build lesson plans and quiz banks tailored to their curriculum.",
    "It explains difficult academic texts in simpler terms for students and learners.",
    "It can conduct mock interviews and simulate different question styles.",
    "Game developers use it to generate lore, item descriptions, or puzzle ideas.",
    "It formats citations in APA, MLA, Chicago, and other academic styles.",
    "ChatGPT helps product managers articulate user stories and acceptance criteria.",
    "It helps people practice new languages by simulating real conversations.",
    "It’s useful for Dungeons & Dragons (D&D) and other tabletop RPG storytelling.",
    "ChatGPT has been used to simulate ancient philosophers and historical figures.",
    "It generates lyrics, song structure ideas, and even musical themes.",
    "It assists in writing grants, research proposals, and abstract summaries.",
    "Users rely on ChatGPT to rewrite sentences for conciseness or sophistication.",
    "It converts spreadsheets into natural language summaries and vice versa.",
    "Entrepreneurs use it to define value propositions and pricing strategies.",
    "It helps brainstorm domain names, taglines, and branding concepts.",
    "It creates mock data for testing systems and interfaces in development.",
    "ChatGPT acts as a co-writer for fiction, non-fiction, and hybrid genres.",
    "It generates scenarios for simulations in military, business, or education.",
    "It rewrites legal language into understandable summaries for laypeople.",
    "It drafts thank-you letters, condolence notes, or motivational speeches.",
    "It helps authors overcome writer’s block by suggesting narrative directions.",
    "It can simulate court arguments or debate formats for training purposes.",
    "It assists in writing press releases and internal company announcements.",
    "ChatGPT is used to simulate conversations for UX/UI testing prototypes.",
    "It suggests improvements in logic or structure for technical documentation.",
    "It summarizes long texts like novels, research papers, or legal rulings.",
    "It creates onboarding materials for new employees or clients.",
    "It helps plan events by generating to-do lists and checklists.",
    "It writes proposals for business deals, service offerings, and partnerships.",
    "ChatGPT is also used to simulate customer personas for design thinking.",
    "It can produce art prompts for AI-generated imagery or concept artists.",
    "It rewrites job descriptions for different seniority or industry levels.",
    "It can roleplay as a coach, mentor, or interviewer across disciplines.",
    "It explains metaphors, symbolism, and literary devices in literature.",
    "It calculates basic math problems and explains the steps involved.",
    "ChatGPT gives recommendations for books, movies, podcasts, and more.",
    "It helps people journal or write autobiographical content for therapy.",
    "It generates different variations of text for A/B testing and UX copy.",
    "It helps build FAQs for websites, apps, and internal documentation.",
    "It’s used in speech writing, debate prep, and persuasive writing contexts.",
    "It can create fictional social media posts for satire or marketing demos.",
    "It transforms technical specs into client-facing descriptions.",
    "It outlines reports, scripts, stories, articles, and whitepapers.",
    "It can reframe arguments from different cultural or ideological angles.",
    "It generates content ideas for newsletters and blog calendars.",
    "It suggests test cases and edge scenarios for QA engineers.",
    "It writes dialogue for animated or interactive storytelling.",
    "It helps compare software tools, frameworks, and vendor options.",
    "It simulates business negotiations and investor Q&A sessions.",
    "It turns meeting notes into action items and follow-up summaries.",
    "It composes opening and closing statements for presentations.",
    "ChatGPT formats JSON, YAML, and HTML from written requirements.",
    "It creates challenges for coding bootcamps or data science courses.",
    "It acts as a muse or second brain for idea incubation.",
    "It can reflect on user writing style and suggest evolution over time.",
    "It can summarize conflict in stories and help with narrative arcs.",
    "It rewrites case studies for clarity and brevity.",
    "It turns rough notes into polished documents and drafts.",
    "It converts pseudocode into working examples in multiple languages.",
    "It finds patterns in personal journal entries or behavior logs.",
    "It simulates discussions between experts on opposing sides.",
    "It turns voice memos (transcripts) into organized notes.",
    "It explains how different models of psychology interpret behaviors.",
    "It generates alternate endings for famous books or films.",
    "It breaks down tasks using time-blocking and priority methods.",
    "It helps therapists explain techniques like CBT, DBT, and EMDR.",
    "It assists in exploring hypothetical ethical dilemmas.",
    "It writes dialogue in screenwriting format for film or theater.",
    "It composes auto-replies for emails and DMs on social platforms.",
    "It helps simulate legal intake processes or client questioning.",
    "It can roleplay user personas to test usability features.",
    "It organizes long research documents into bullet summaries.",
    "It can suggest metaphors or analogies for complex concepts.",
    "It’s a sounding board for decision-making under uncertainty.",
    "ChatGPT is a bridge between raw ideas and polished articulation."
]

documents.append("\n".join(doc3))

doc4 = [
    "Finance teams use ChatGPT to automate manual documentation workflows.",
    "It assists in generating first drafts of monthly financial summaries.",
    "Teams use it to clean and normalize transaction descriptions across ledgers.",
    "It helps format Excel formulas and translate them into plain language explanations.",
    "ChatGPT auto-drafts meeting agendas and follow-ups for finance reviews.",
    "It can generate internal memos summarizing P&L highlights.",
    "The model helps explain KPIs like EBITDA, net margin, or AR turnover to non-finance stakeholders.",
    "It formats journal entry descriptions for consistency in reporting.",
    "Finance teams use it to compare budgeted vs actual costs across departments.",
    "It creates variance commentary based on accounting inputs.",
    "ChatGPT can summarize audit findings into executive-readable formats.",
    "It aids in crafting quarterly business review (QBR) presentations.",
    "The model helps analysts draft cash flow forecasts and key takeaways.",
    "It can standardize expense report policies into short bullet points.",
    "ChatGPT writes documentation for internal financial controls.",
    "It generates test cases for reconciliation tools and workflows.",
    "Teams use it to simulate what-if financial scenarios.",
    "ChatGPT helps convert SOPs into process flow diagrams using textual prompts.",
    "It summarizes vendor contract terms for finance approvals.",
    "It extracts invoice terms, payment dates, and penalties from PDFs.",
    "ChatGPT assists procurement teams in formatting RFP responses.",
    "It can generate automated replies to vendor or finance-related email queries.",
    "The model helps write business case justifications for capex requests.",
    "It summarizes key payment trends over time from ledger data.",
    "ChatGPT can help create knowledge base articles for finance tools.",
    "It acts as a virtual assistant to field repetitive finance-related questions.",
    "It summarizes payment approval chains and workflows.",
    "The model is used to write macroeconomic overviews for leadership decks.",
    "It helps document assumptions behind key financial models.",
    "ChatGPT assists with updating financial glossary and policy documents.",
    "It can convert a rough idea into a polished budget request.",
    "The model explains finance workflows to non-finance departments in plain terms.",
    "It creates onboarding materials for new hires in finance.",
    "ChatGPT is used to write footnotes and disclosures for reports.",
    "It identifies repetitive entries in ledger data for cleansing.",
    "The model formats data into pivot-ready CSV or JSON summaries.",
    "It writes disclaimers for reports, especially for forward-looking statements.",
    "ChatGPT helps track project financial performance via dashboard summaries.",
    "It translates audit points into action items and remediation tasks.",
    "It turns raw notes from controller meetings into structured action logs.",
    "The model can generate accrual narratives from invoice queues.",
    "It assists in simulating email exchanges for finance process training.",
    "ChatGPT can interpret cost center codes and reassign spend categories.",
    "It generates contingency plans and risk summaries based on financial scenarios.",
    "The model converts expense policy changes into internal comms format.",
    "ChatGPT can map out approval thresholds across different regions.",
    "It helps flag compliance gaps in vendor spend or payment timelines.",
    "It writes stakeholder communications around billing errors or issues.",
    "The model assists in explaining differences between GAAP and IFRS treatments.",
    "It creates feedback surveys for finance tool adoption.",
    "ChatGPT generates visual-ready bullet points from spreadsheets.",
    "It explains financial ratios to junior analysts during onboarding.",
    "It can help you draft justifications for budget overruns.",
    "The model rephrases accounting narratives for clarity or brevity.",
    "It writes step-by-step guides for using finance tools like SAP or NetSuite.",
    "It helps summarize tax rule updates for global teams.",
    "ChatGPT formats timelines for close activities by month or quarter.",
    "It generates checklist templates for quarter-end reconciliations.",
    "It helps standardize communication between finance and procurement.",
    "The model can translate technical audit language into layperson summaries.",
    "It assists in producing audit prep documentation from ticketing tools.",
    "ChatGPT creates training exercises for finance software rollouts.",
    "It outlines pros and cons for build vs buy decisions in finance tooling.",
    "It helps write finance-related Jira tickets more clearly.",
    "ChatGPT is used to create SOP drafts from call recordings.",
    "It can summarize month-end close blockers with issue ownership.",
    "It helps review and polish narratives in investor relations decks.",
    "It produces descriptions of finance KPIs for dashboard headers.",
    "ChatGPT can simulate cross-functional email threads for spend approval.",
    "It auto-drafts Slack updates about spend controls and budget limits.",
    "It explains FX risk and hedging in terms a business manager can grasp.",
    "It formats approval notes for long-form vendor quotes.",
    "ChatGPT can outline the life cycle of a PO from requisition to settlement.",
    "It summarizes audit logs for access reviews.",
    "It produces root cause narratives for delayed payments.",
    "The model writes internal newsletters for finance department updates.",
    "ChatGPT creates policy comparison tables across regions or teams.",
    "It simulates an FAQ for procurement onboarding.",
    "It turns system migration plans into stakeholder-ready summaries.",
    "It helps write training quizzes for policy understanding.",
    "It assists in crafting messages for finance change management rollouts.",
    "It generates confidence score explanations for finance ML models.",
    "It explains lease accounting logic to teams using new standards like ASC 842.",
    "ChatGPT helps finance teams self-document their workflows for audits.",
    "It turns retro meeting notes into streamlined lessons learned.",
    "It helps outline shared services responsibilities and SLAs.",
    "The model auto-generates suggested journal entries from raw receipts.",
    "It converts transactional finance feedback into roadmap features.",
    "ChatGPT helps flag outdated or duplicated documentation.",
    "It generates decision logs for intercompany billing setups.",
    "It helps document exceptions in cash application rules.",
    "The model explains treasury operations to interns and non-specialists.",
    "It produces a 5-minute summary of quarter-close status across regions.",
    "It helps create finance chatbot scripts to answer routine queries.",
    "ChatGPT outlines requirements for vendor onboarding checklists.",
    "It translates accounting guidance into task lists for implementation.",
    "It suggests ways to visualize multi-entity cash balances.",
    "It produces version histories for financial assumptions in forecasts.",
    "It’s a reliable co-pilot for document-heavy, process-driven finance ops."
]

documents.append("\n".join(doc4))

doc5 = [
    "BloombergGPT is more accurate on finance-specific tasks than ChatGPT.",
    "ChatGPT is more versatile across non-financial domains like education or healthcare.",
    "BloombergGPT’s training data is mostly proprietary, while ChatGPT's is broader and more public.",
    "ChatGPT can generalize better to open-ended prompts, while BloombergGPT excels in structured finance prompts.",
    "BloombergGPT is better at interpreting SEC filings and market commentary.",
    "ChatGPT is easier to fine-tune for creative, customer-facing use cases.",
    "BloombergGPT has stronger performance on finance benchmarks like FiQA and Headline Risk.",
    "ChatGPT offers better conversation memory across diverse topic shifts.",
    "ChatGPT is available on more platforms, while BloombergGPT is limited to Bloomberg systems.",
    "BloombergGPT outperforms in interpreting credit default risk narratives.",
    "ChatGPT is more widely adopted in education, content creation, and general business.",
    "BloombergGPT is specialized for internal financial workflows and analytics.",
    "ChatGPT supports more languages and use cases out of the box.",
    "BloombergGPT understands financial entities, ratios, and abbreviations better.",
    "ChatGPT can handle roleplay, scenario generation, and creative ideation.",
    "BloombergGPT delivers higher faithfulness when summarizing financial disclosures.",
    "ChatGPT offers broader API support for integration across industries.",
    "BloombergGPT’s outputs are tuned to match financial formatting and tone.",
    "ChatGPT is more transparent, with more public benchmarks and research.",
    "BloombergGPT’s accuracy is higher in parsing tables in earnings reports.",
    "ChatGPT is more suitable for brainstorming startup ideas or storytelling.",
    "BloombergGPT generates fewer hallucinations in finance-specific QA tasks.",
    "ChatGPT provides higher engagement for dialogue and interactive tutoring.",
    "BloombergGPT is better at flagging regulatory risks from raw filings.",
    "ChatGPT shines in summarizing diverse, multi-domain content quickly.",
    "BloombergGPT integrates with Bloomberg Terminal functions natively.",
    "ChatGPT works across industries like law, medicine, education, and tech.",
    "BloombergGPT is fine-tuned on structured datasets like market feeds.",
    "ChatGPT supports plugins and third-party integrations more flexibly.",
    "BloombergGPT outputs adhere to financial compliance expectations.",
    "ChatGPT can simulate fictional personas, games, and role-specific agents.",
    "BloombergGPT maintains entity resolution across multi-document contexts.",
    "ChatGPT can summarize pop culture, literature, or psychology with ease.",
    "BloombergGPT links company names to ticker symbols accurately.",
    "ChatGPT can create onboarding docs, marketing plans, and UX content.",
    "BloombergGPT is effective in extracting financial key metrics from prose.",
    "ChatGPT is more engaging for non-technical users in casual dialogue.",
    "BloombergGPT provides relevant context from recent financial reports.",
    "ChatGPT is widely used in education and language learning products.",
    "BloombergGPT maps sentiment shifts over time in sector news.",
    "ChatGPT is better at generating variants of email copy or CTAs.",
    "BloombergGPT ensures tighter confidence in factual financial claims.",
    "ChatGPT is more forgiving to open-ended or incomplete prompts.",
    "BloombergGPT understands relationships between macro indicators better.",
    "ChatGPT handles emotion, tone, and humor more naturally.",
    "BloombergGPT responds better to data-heavy prompts and charts.",
    "ChatGPT helps developers, researchers, and designers alike.",
    "BloombergGPT is designed to serve finance professionals and analysts.",
    "ChatGPT can generate and critique advertisements and slogans.",
    "BloombergGPT suggests relevant historical financial contexts.",
    "ChatGPT is often used as a creative thinking partner.",
    "BloombergGPT specializes in financial content filtering and compliance.",
    "ChatGPT adapts easily to new topics and casual conversation.",
    "BloombergGPT requires domain-specific access and configuration.",
    "ChatGPT supports broad use, including storytelling and content design.",
    "BloombergGPT shines in data-heavy financial commentary.",
    "ChatGPT has greater openness in tooling, plugins, and developer support.",
    "BloombergGPT is primarily for enterprise-grade finance use.",
    "ChatGPT balances accessibility with general-purpose intelligence.",
    "BloombergGPT focuses on enterprise-grade performance for finance.",
    "ChatGPT fosters collaboration across education, startups, and art.",
    "BloombergGPT reflects deep domain understanding of financial systems.",
    "ChatGPT adapts well to use cases outside finance.",
    "BloombergGPT performs better in summarizing complex legal financial language.",
    "ChatGPT excels at interactive ideation and brainstorming.",
    "BloombergGPT outpaces ChatGPT in parsing Bloomberg proprietary data.",
    "ChatGPT is better for generalized tasks and hybrid workflows.",
    "BloombergGPT delivers better recall of specific economic events.",
    "ChatGPT enables writing, feedback, and stylistic transformation.",
    "BloombergGPT avoids hallucinating company names or tickers.",
    "ChatGPT handles ambiguity and multi-turn queries smoothly.",
    "BloombergGPT is optimized for extracting financial footnote implications.",
    "ChatGPT serves as a knowledge assistant across many teams.",
    "BloombergGPT supports enterprise-grade compliance in responses.",
    "ChatGPT simplifies complexity across business and social topics.",
    "BloombergGPT responds reliably to technical financial prompts.",
    "ChatGPT handles ambiguous prompts more creatively.",
    "BloombergGPT focuses on predictive financial language tasks.",
    "ChatGPT is widely used in media, tech, and education sectors.",
    "BloombergGPT augments finance workflows with fewer hallucinations.",
    "ChatGPT’s flexibility allows prototyping across product teams.",
    "BloombergGPT can trace sectoral trends over historical data.",
    "ChatGPT supports research through summarization and transformation.",
    "BloombergGPT ensures financial consistency in tabular output.",
    "ChatGPT is optimized for generative, flexible text outputs.",
    "BloombergGPT provides institutional-grade answers for traders.",
    "ChatGPT enables simulation of many real-world use cases.",
    "BloombergGPT limits response drift in technical language tasks.",
    "ChatGPT excels in cross-domain knowledge synthesis.",
    "BloombergGPT is ideal when precision in finance is paramount.",
    "ChatGPT fosters creativity, adaptability, and natural flow.",
    "BloombergGPT ensures trustworthy financial insights."
]

documents.append("\n".join(doc5))

# -------------------------------
# Embed and store docs with TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# -------------------------------
# Query planner (simple splitter)
# -------------------------------
@judgment.observe(span_type="function")
def plan_query(user_input):
    system_prompt = "You are a helpful assistant that breaks down complex questions into simpler sub-questions for a RAG chatbot to handle separately."

    planner_prompt = f"""Break down the following question into a list of simpler sub-questions, if needed. 
If it's already simple, return just one. Format as a numbered list.

Question: "{user_input}"
Sub-questions:"""

    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": planner_prompt}
    ],
    temperature=0.3)

    content = response.choices[0].message.content

    # Parse response into a list of sub-questions
    sub_questions = []
    for line in content.strip().split("\n"):
        if line.strip() and any(c.isalpha() for c in line):
            q = line.strip().split(". ", 1)[-1]
            sub_questions.append(q.strip())

    return sub_questions

# -------------------------------
# Retriever (TF-IDF + cosine)
# -------------------------------
@judgment.observe(span_type="retriever")
def retrieve_relevant_docs(query, top_k=2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# -------------------------------
# Generator using GPT-4o
# -------------------------------
@judgment.observe(span_type="tool")
def generate_answer(question, context):
    prompt = f"""You are an AI assistant. Use the following context to answer the question.
    
Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7)
    return response.choices[0].message.content.strip()


@judgment.observe(span_type="function")
def evaluate_faithfulness(sub_query, answers):
    docs = retrieve_relevant_docs(sub_query)
    context = "\n".join(docs)
    answer = generate_answer(sub_query, context)
    answers.append(f"**{sub_query}**\n{answer}\n")
    retrieval_context = list()
    retrieval_context.append(context)
    judgment.async_evaluate(
            scorers=[FaithfulnessScorer(threshold=0.5)],
            input=sub_query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            model="gpt-4o",
        )
    return answer, answers
# -------------------------------
# Main function
# -------------------------------



@judgment.observe(span_type="function")
def rag_chatbot(user_input):
    sub_queries = plan_query(user_input)
    answers = []

    for sub_query in sub_queries:
        answer, answers = evaluate_faithfulness(sub_query, answers)
        answers.append(answer)

    return "\n".join(answers)

# -------------------------------
# Example run
# -------------------------------
if __name__ == "__main__":
    query = (
    "Compare the applicability and strengths of ChatGPT and BloombergGPT in the context of financial operations, "
    "including how each model assists finance teams with internal workflows, compliance, and stakeholder communication. "
    "Also evaluate how each model handles document summarization, explains financial concepts to non-experts, "
    "and supports audit or reporting tasks. Finally, comment on their versatility beyond finance and their usefulness "
    "in long-term integration across departments like procurement, IT, and legal."
    )
    result = rag_chatbot(query)
    print(result)


