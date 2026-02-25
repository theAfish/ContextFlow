# ContextFlow Roadmap Tracker

## Phase 1: The Core

- [x] Define `ContextNode` and `ContextStack` data structures
- [x] Create a `Composer` with dynamic slot-filling
- [x] Implement `render()` to output final `messages[]`
- [x] Build a `ResponseParser` for JSON/Pydantic

## Phase 2: Persistence & Sessions

- [ ] Integrate PostgreSQL session storage
- [x] Add base `SessionManager` (in-memory)
- [x] Add snapshot interfaces + in-memory implementation
- [ ] Add production-grade token counter

## Phase 3: Async Engine & Tooling

- [x] Scaffold `AsyncAgentRunner`
- [x] Add interceptor interfaces
- [ ] Add WebSocket runtime handler

## Phase 4: Observability & Memory

- [x] Add Lens middleware tracer
- [ ] Add vector memory integration
