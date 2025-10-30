```
agb8-core/
│
├── core/                          # البنية التحتية الأساسية للمنصة
│   ├── __init__.py
│   ├── config/                    # إعدادات عامة (بيئة، مفاتيح، إعدادات API)
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── constants.py
│   │   └── logging_config.py
│   │
│   ├── utils/                     # أدوات مساعدة عامة
│   │   ├── __init__.py
│   │   ├── file_io.py
│   │   ├── decorators.py
│   │   ├── prompts.py
│   │   └── validation.py
│   │
│   ├── llm/                       # تكاملات LLM (HuggingFace / OpenAI / etc.)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── huggingface_client.py
│   │   ├── openai_client.py
│   │   └── model_factory.py       # لتوحيد اختيار الموديلات
│   │
│   ├── integrations/              # تكاملات مخصصة (Custom Integrations)
│   │   ├── __init__.py
│   │   ├── google_sheets.py
│   │   ├── notion.py
│   │   ├── slack.py
│   │   └── custom_api_client.py
│   │
│   ├── memory/                    # الذاكرة الخاصة بالوكلاء (يمكن توصيلها بـ LlamaIndex لاحقاً)
│   │   ├── __init__.py
│   │   ├── base_memory.py
│   │   ├── vector_store.py
│   │   └── sqlite_memory.py
│   │
│   ├── agents/                    # تعريف الوكلاء وأنواعهم
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── task_agent.py
│   │   ├── research_agent.py
│   │   └── automation_agent.py
│   │
│   ├── chains/                    # تعريف السلاسل (LangChain Pipelines)
│   │   ├── __init__.py
│   │   ├── base_chain.py
│   │   ├── qa_chain.py
│   │   ├── data_analysis_chain.py
│   │   └── summarization_chain.py
│   │
│   ├── mcp/                       # تكامل MCP (Model Context Protocol)
│   │   ├── __init__.py
│   │   ├── mcp_server.py
│   │   ├── mcp_client.py
│   │   └── adapters/
│   │       ├── __init__.py
│   │       ├── langchain_adapter.py
│   │       └── llamaindex_adapter.py
│   │
│   ├── graph/                     # لدمج LangGraph لاحقاً
│   │   ├── __init__.py
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── agent_node.py
│   │   │   ├── memory_node.py
│   │   │   └── integration_node.py
│   │   └── graph_builder.py
│   │
│   └── server/                    # لتشغيل API داخلي أو HTTP endpoint
│       ├── __init__.py
│       ├── api.py
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── agents_routes.py
│       │   ├── integrations_routes.py
│       │   └── chains_routes.py
│       └── schemas/
│           ├── __init__.py
│           ├── agent_schema.py
│           └── integration_schema.py
│
├── tests/                         # اختبارات لكل وحدة
│   ├── test_agents.py
│   ├── test_integrations.py
│   ├── test_llm.py
│   └── test_memory.py
│
├── scripts/                       # سكريبتات تشغيل وصيانة
│   ├── run_server.py
│   ├── train_agent.py
│   ├── build_memory_index.py
│   └── register_integration.py
│
├── pyproject.toml / requirements.txt
├── .env.example
└── README.md
```