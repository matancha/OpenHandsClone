# State Tracking and Progress in OpenHands

This document explains how OpenHands tracks intermediate results and manages persistent state. It addresses whether the system keeps an internal project representation and how progress is serialized between iterations.

## Overview

OpenHands maintains a `State` dataclass during execution. Each agent interaction produces `Event` objects (actions, observations, etc.) which are appended to `state.history`. The controller updates counters and metrics on every step. When the session ends, the state is pickled to disk so work can resume later.

```python
@dataclass
class State:
    session_id: str = ''
    iteration: int = 0
    local_iteration: int = 0
    max_iterations: int = 100
    history: list[Event] = field(default_factory=list)
    agent_state: AgentState = AgentState.LOADING
    metrics: Metrics = field(default_factory=Metrics)
```
{cite}`F:openhands/controller/state/state.py#77-97`

The `view` property lazily converts `state.history` into a `View` object. A checksum of the history length ensures the view is rebuilt only when new events are added.

```python
@property
def view(self) -> View:
    history_checksum = len(self.history)
    old_history_checksum = getattr(self, '_history_checksum', -1)
    if history_checksum != old_history_checksum:
        self._history_checksum = history_checksum
        self._view = View.from_events(self.history)
    return self._view
```
{cite}`F:openhands/controller/state/state.py#226-238`

## Updating and Saving State

`AgentController` increments iteration counters before each LLM step and records metrics afterwards.

```python
def update_state_before_step(self) -> None:
    self.state.iteration += 1
    self.state.local_iteration += 1

async def update_state_after_step(self) -> None:
    self.state.local_metrics = copy.deepcopy(self.agent.llm.metrics)
```
{cite}`F:openhands/controller/agent_controller.py#260-266`

Incoming events are appended to the history inside `_on_event`:

```python
if self.agent_history_filter.include(event):
    self.state.history.append(event)
```
{cite}`F:openhands/controller/agent_controller.py#414-418`

Before shutdown, `main` saves the state to the configured `FileStore`:

```python
end_state = controller.get_state()
end_state.save_to_session(
    event_stream.sid, event_stream.file_store, event_stream.user_id
)
```
{cite}`F:openhands/core/main.py#208-218`

`save_to_session` pickles the dataclass and writes it to disk:

```python
def save_to_session(self, sid: str, file_store: FileStore, user_id: str | None) -> None:
    pickled = pickle.dumps(self)
    encoded = base64.b64encode(pickled).decode('utf-8')
    file_store.write(get_conversation_agent_state_filename(sid, user_id), encoded)
```
{cite}`F:openhands/controller/state/state.py#107-116`

On startup `setup_agent` restores the previous session if available:

```python
initial_state = State.restore_from_session(
    event_stream.sid, event_stream.file_store
)
```
{cite}`F:openhands/core/setup.py#196-203`

Restoration resets the agent to `LOADING` and preserves the previous state so work continues where it left off:

```python
encoded = file_store.read(get_conversation_agent_state_filename(sid, user_id))
state = pickle.loads(base64.b64decode(encoded))
if state.agent_state in RESUMABLE_STATES:
    state.resume_state = state.agent_state
else:
    state.resume_state = None
state.agent_state = AgentState.LOADING
```
{cite}`F:openhands/controller/state/state.py#130-168`

## Memory Condensation

To prevent the history from exceeding token limits, OpenHands uses a memory condenser. The structured summary condenser summarizes old events and inserts a compact `AgentCondensationObservation` back into history.

The summary fields track high‑level progress such as completed tasks, pending tasks, files modified, and test status:

```python
class StateSummary(BaseModel):
    user_context: str = Field(default='')
    completed_tasks: str = Field(default='')
    pending_tasks: str = Field(default='')
    current_state: str = Field(default='')
    files_modified: str = Field(default='')
    function_changes: str = Field(default='')
    # ... additional fields omitted for brevity ...
```
{cite}`F:openhands/memory/condenser/impl/structured_summary_condenser.py#24-60`

When condensation is triggered, the condenser builds a prompt containing the previous summary and the events to forget, then requests a function call so the LLM returns a structured object:

```python
messages = [Message(role='user', content=[TextContent(text=prompt)])]
response = self.llm.completion(
    messages=self.llm.format_messages_for_llm(messages),
    tools=[StateSummary.tool_description()],
    tool_choice={
        'type': 'function',
        'function': {'name': 'create_state_summary'},
    },
)
```
{cite}`F:openhands/memory/condenser/impl/structured_summary_condenser.py#218-258`

The resulting `CondensationAction` replaces the forgotten events, preserving a compact summary of work done and what remains.

## Project Structure Representation

OpenHands itself does **not** maintain an internal code graph or project tree. File operations are executed directly via runtime tools (`FileReadAction`, `FileEditAction`, etc.) without building a global model of the repository. However, an optional microagent called **LocAgent** can explore a pre‑built code graph using `explore_tree_structure` and related tools. These tools rely on an external indexing step (the `openhands_aci` package) rather than an in-memory structure maintained by OpenHands.

```python
from openhands_aci.indexing.locagent.tools import (
    explore_tree_structure,
    get_entity_contents,
    search_code_snippets,
)
```
{cite}`F:openhands/runtime/plugins/agent_skills/repo_ops/repo_ops.py#1-7`

Thus the core system infers project structure dynamically from disk reads and optional microagents but does not keep a persistent AST or dependency graph of its own.

## Progress Awareness

Because the condensed history retains summaries of prior work, each new LLM call receives a short description of completed and pending tasks. The LLM therefore remains aware of the overall progress without needing the entire conversation. The event history plus these summaries provide enough context for the agent to continue where it left off after any interruption.

## Serialization and Resumption

State is serialized to disk via `save_to_session` and restored with `restore_from_session`. Events themselves are stored as individual JSON files through `EventStore`, allowing the controller to replay history if needed. A cached view of recent events plus structured summaries keeps prompts concise while ensuring the agent remembers what has been done and what still needs attention.

## Summary Table

| Feature | Explicitly Tracked? | Mechanism |
| --- | --- | --- |
| Completed Tasks | ✅ | `StateSummary.completed_tasks` |
| Pending Tasks | ✅ | `StateSummary.pending_tasks` |
| File Modifications | ✅ | `AgentCondensationObservation` |
| Project Tree / Code Graph | ❌ | Not tracked (optional via `LocAgent`) |
| Task Plan / Scratchpad | ❌ | No persistent task structure |
| Agent Resumption | ✅ | `State.resume_state`, `restore_from_session` |

## Design Commentary

OpenHands tracks state through structured event logs and summarization rather than a persistent task planner or code graph. This design favors simplicity and auditability—all progress is encoded in observable actions and LLM-generated summaries. However, the lack of a structured task graph limits long-term decomposition and reasoning across large projects. Optional microagents like LocAgent can provide code graph exploration but require extra setup.

