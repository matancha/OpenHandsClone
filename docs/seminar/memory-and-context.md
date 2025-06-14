# Memory and Context Management in OpenHands

OpenHands maintains a structured conversation history rather than a raw text log. Each iteration of the agent generates **events** which are stored in the persistent `State` object. Events include the user prompts, actions produced by the LLM, and the runtime observations that follow. The design lets the system reconstruct the relevant conversation context for every LLM call while keeping long-term storage compact.

## The `State` dataclass

The core of persistence is the `State` dataclass in `openhands/controller/state/state.py`. It tracks counters, the list of events and token metrics. When OpenHands shuts down, `save_to_session` pickles this dataclass to disk. On start-up it is loaded with `restore_from_session`:

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

This ensures multi-step tasks can resume even after interruptions.

### Persistent vs Ephemeral Memory

`save_to_session` writes the `State` dataclass to disk while
`restore_from_session` reloads it when the session restarts.
The raw `history` field in `State` stores a list of events.
When preparing input for the LLM a transient `View` is built from these events.
This view is not saved to disk&mdash;it is recomputed on demand whenever the event history changes:

```python
def save_to_session(self, sid: str, file_store: FileStore, user_id: str | None) -> None:
    pickled = pickle.dumps(self)
    file_store.write(get_conversation_agent_state_filename(sid, user_id), encoded)

def restore_from_session(sid: str, file_store: FileStore, user_id: str | None = None) -> "State":
    encoded = file_store.read(get_conversation_agent_state_filename(sid, user_id))
    pickled = base64.b64decode(encoded)
    state = pickle.loads(pickled)
```
{cite}`F:openhands/controller/state/state.py#107-166`

The view used for LLM prompts is recomputed whenever the history changes:

```python
history_checksum = len(self.history)
if history_checksum != old_history_checksum:
    self._history_checksum = history_checksum
    self._view = View.from_events(self.history)
return self._view
```
{cite}`F:openhands/controller/state/state.py#228-239`

This ephemeral view is held in memory only during runtime.

Events stored in `State.history` capture everything the agent does or observes:

- **MessageAction** – user and agent chat messages
- **CmdRunAction** and **FileEditAction** – code execution or edits
- **Observations** – command output, file contents, or errors
- **AgentCondensationObservation** – summaries of forgotten history


## Short Term History and Memory Condenser

The in-memory conversation history can grow large. `openhands/memory/README.md` describes two components:

- **Short Term History** – filters incoming events and injects them into the LLM context. When the context window would be exceeded it replaces old segments with summaries.
- **Memory Condenser** – summarises forgotten chunks of events using the LLM and stores those summaries back into the state.

```text
- Short term history filters the event stream and computes the messages that are injected into the context
- When the context window or the token limit set by the user is exceeded, history starts condensing: chunks of messages into summaries.
- Each summary is then injected into the context, in the place of the respective chunk it summarizes
```
{cite}`F:openhands/memory/README.md#6-18`

The structured summary condenser builds a prompt containing the previous summary and the events about to be forgotten. It requests a function call `create_state_summary` so the LLM returns a structured object:

```python
prompt = "You are maintaining a context-aware state summary for an interactive software agent..."
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

The result is inserted back into the history as an `AgentCondensationObservation`, effectively compressing past conversation into a short structured state summary.
Condensation is lossy but structured &mdash; summaries capture intent and key actions, not raw logs, enabling effective task resumption without exceeding token budgets.

## Building the Prompt

Before the LLM is called, `ConversationMemory` converts the condensed history into a list of `Message` objects. It ensures a system prompt is present, inserts the initial user request if missing, and merges repository information with any microagents triggered by keywords:

```python
formatted_workspace_text = self.prompt_manager.build_workspace_context(
    repository_info=repo_info,
    runtime_info=runtime_info,
    conversation_instructions=conversation_instructions,
    repo_instructions=repo_instructions,
)
message_content.append(TextContent(text=formatted_workspace_text))
if has_microagent_knowledge:
    formatted_microagent_text = self.prompt_manager.build_microagent_info(
        triggered_agents=filtered_agents,
    )
    message_content.append(TextContent(text=formatted_microagent_text))
message = Message(role='user', content=message_content)
```
{cite}`F:openhands/memory/conversation_memory.py#488-535`

Microagents include `.openhands/microagents/repo.md` and any keyword-triggered knowledge agents. They are inserted as user messages wrapped in `<EXTRA_INFO>` blocks.

### Rebuilding Context Each Step

`ConversationMemory.process_events` runs on every iteration, so the prompt is rebuilt from the condensed history. When the context window is exceeded the controller recomputes a view and trims old events:

```python
current_view = View.from_events(self.state.history)
kept_events = self._apply_conversation_window(current_view.events)
```
{cite}`F:openhands/controller/agent_controller.py#1164-1167`

No intermediate LLM responses are cached beyond optional prompt caching.

### Delegation and Memory Sharing

When a parent agent delegates to a child, the new controller inherits metrics and shares the same event stream:

```python
state = State(
    session_id=self.id.removesuffix('-delegate'),
    inputs=action.inputs or {},
    local_iteration=0,
    iteration=self.state.iteration,
    max_iterations=self.state.max_iterations,
    delegate_level=self.state.delegate_level + 1,
    metrics=self.state.metrics,
    start_id=self.event_stream.get_latest_event_id() + 1,
)
self.delegate = AgentController(
    sid=self.id + '-delegate',
    agent=delegate_agent,
    event_stream=self.event_stream,
    max_iterations=self.state.max_iterations,
    max_budget_per_task=self.max_budget_per_task,
    agent_to_llm_config=self.agent_to_llm_config,
    agent_configs=self.agent_configs,
    initial_state=state,
    is_delegate=True,
    headless_mode=self.headless_mode,
)
```
{cite}`F:openhands/controller/agent_controller.py#670-720`

When the subtask ends the parent updates its iteration count and records the delegate result:

```python
self.state.iteration = self.delegate.state.iteration
asyncio.get_event_loop().run_until_complete(self.delegate.close())
obs = AgentDelegateObservation(outputs=delegate_outputs, content=content)
self.event_stream.add_event(obs, EventSource.AGENT)
```
{cite}`F:openhands/controller/agent_controller.py#722-774`

## No Vector Store or Scratchpad

The repository search shows no implementation of a vector store or external knowledge base. Memory is derived solely from the event history and the optional microagent files. Summaries are produced by the LLM itself via the condenser; there is no retrieval from a vector database or long term scratchpad beyond the saved `State` pickles.

## Conclusion

OpenHands manages multi-step tasks by tracking a structured event history, condensing older events into summaries when needed, and rebuilding the LLM prompt on each iteration. It does not rely on external vector stores but instead persists a lightweight state file that includes a condensed summary of past work. This approach allows the agent to work on complex software tasks across multiple iterations without exceeding context limits.
By separating short-term view generation from long-term persistence, OpenHands ensures stateless, repeatable LLM prompts while maintaining continuity across iterations&mdash;a design that balances context-window constraints with scalability.

