# Project: ContextFlow 🌊

> **Tagline:** Stop the Black Box. Orchestrate the Context. Build Transparent Agents.

## 1. Project Vision

Most Agent frameworks (like LangChain) wrap logic in "Abstractions" that hide the raw prompt and state. **ContextFlow** reverses this. It treats the **Context Window** as the primary state, providing developers with high-level tools to assemble, prune, and post-process LLM interactions without losing visibility.

---

## 2. Core Philosophy

* **Explicit > Implicit**: No hidden prompts. The developer should always be able to print the exact `messages[]` array sent to the LLM.
* **Context-First**: The framework's job is to manage the *composition* of the context, not the *magic* of the reasoning.
* **Post-Process, Don't Pre-Execute**: The framework parses LLM outputs into structured data but leaves the execution of business logic/tools to the developer (or explicit hooks).

---

## 3. System Architecture

| Component | Responsibility |
| --- | --- |
| **Context Composer** | Defines templates and logic for merging System Prompts, History, and RAG data. |
| **State Engine** | Handles `async` execution loops and manages the transition between conversation turns. |
| **Persistence Layer** | Manages Session storage (PostgreSQL) and Vector search (pgvector). |
| **The Lens (Debugger)** | A middleware that logs/visualizes the raw JSON sent to and received from the LLM. |

---

## 4. Technical Stack

* **Language**: Python 3.10+ (Type-hinting heavy)
* **API Framework**: FastAPI (for high-performance async endpoints)
* **Data Validation**: Pydantic v2
* **Database**: PostgreSQL + SQLAlchemy (Async) + `pgvector` for memory
* **LLM Clients**: LiteLLM (for multi-model support) or direct OpenAI/Anthropic SDKs

---

## 5. Development Roadmap (TODO.md)

### Phase 1: The Core (The "White Box" SDK)

* [ ] Define `ContextNode` and `ContextStack` data structures.
* [ ] Create a `Composer` class that supports dynamic slot-filling (e.g., inserting `{{current_time}}` or `{{search_results}}`).
* [ ] Implement a basic `render()` method to output the final `messages[]` list.
* [ ] Build a `ResponseParser` for structured tool-calling (JSON/Pydantic).

### Phase 2: Persistence & Session Management

* [ ] Integrate **PostgreSQL** for session storage.
* [ ] Implement `SessionManager` to handle multi-user state isolation.
* [ ] Add "Snapshot" functionality: Save the context state at every turn to allow "Undo/Revert" features.
* [ ] **Token Counter**: Implement an accurate token estimator to prevent context overflow before calling the API.

### Phase 3: Async Engine & Tooling

* [ ] Build an `AsyncAgentRunner` to handle streaming responses.
* [ ] Implement **Interceptors**: Pre-processing hooks (e.g., PII masking) and Post-processing hooks (e.g., auto-tool execution).
* [ ] Create a WebSocket handler for real-time Agent-to-UI communication.

### Phase 4: Observability & Memory

* [ ] **The Lens UI**: A simple React/Streamlit dashboard to view the "Live Context" of any active session.
* [ ] **Vector Memory**: Automated RAG integration that pulls relevant snippets into the `ContextStack` based on user input.

---

## 6. Key Implementation Notes

* **Pruning Strategies**: When the context hits the limit, don't just "forget" the oldest message. Allow developers to define strategies: `summarize`, `drop_middle`, or `keep_system_only`.
* **State Locking**: Since it's async, ensure two simultaneous messages to the same Session don't corrupt the Context.