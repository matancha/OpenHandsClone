# OpenHands Prompt Flow Overview

This document summarizes how OpenHands processes user prompts and prepares messages for the language model. It is intended as reference material for seminar participants exploring the codebase.

## Prompt Types Accepted

OpenHands accepts natural language tasks from the user. These can range from high level feature requests to specific bug fixes. Good prompts are concrete, location‑specific and scoped to a small task, as shown in the documentation:

```
Add a function `calculate_average` in `utils/math_operations.py` that takes a list of numbers as input and returns their average.
```

Poor prompts like "Make the code better" are discouraged. See `docs/usage/prompting/prompting-best-practices.mdx` for full guidance.

### Microagents

The system can extend prompts with **microagents** – markdown files that contain reusable instructions. These microagents are loaded either from the repository itself or from shared public microagents:

```text
OpenHands/microagents/
├── # Keyword-triggered expertise
│   ├── git.md         # Git operations
│   ├── testing.md     # Testing practices
│   └── docker.md      # Docker guidelines
└── # These microagents are always loaded
    ├── pr_review.md   # PR review process
    ├── bug_fix.md     # Bug fixing workflow
    └── feature.md     # Feature implementation
```
{cite}`F:microagents/README.md#1-32`

Keyword-triggered microagents use YAML frontmatter to specify `triggers` that activate them when those keywords appear in the user prompt:

```yaml
---
triggers:
- yummyhappy
- happyyummy
---
The user has said the magic word. Respond with "That was delicious!"
```
{cite}`F:docs/usage/prompting/microagents-keyword.mdx#25-33`

Repository microagents (e.g. `.openhands/microagents/repo.md`) are always loaded when OpenHands is run inside that repository and can document project-specific workflows.

### Prompt Construction

When a message is processed, `Memory` scans it for trigger keywords. Each match creates a `MicroagentKnowledge` entry. During the workspace recall step `ConversationMemory` builds a single **user** message that contains both the workspace context and any triggered microagent text:

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

The microagent content itself is inserted verbatim using the `microagent_info.j2` template which wraps each block in `<EXTRA_INFO>` tags:

```jinja
{% for agent_info in triggered_agents %}
<EXTRA_INFO>
The following information has been included based on a keyword match for "{{ agent_info.trigger }}".
It may or may not be relevant to the user's request.

{{ agent_info.content }}
</EXTRA_INFO>
{% endfor %}
```
{cite}`F:openhands/agenthub/codeact_agent/prompts/microagent_info.j2#1-8`

All triggered microagents are concatenated in a single user message. This prevents role repetition and allows multiple microagents to share the same `<EXTRA_INFO>` wrapper. The wrapper helps the LLM distinguish core task instructions from auxiliary background, reducing the risk of prompt injection and unintended completions. Both the workspace context and microagent blocks are treated as **user** messages in the final prompt list.

### Trigger Conflicts and Deduplication

If several microagents match the same prompt, `Memory` collects them all without ranking:

```python
for name, microagent in self.knowledge_microagents.items():
    trigger = microagent.match_trigger(query)
    if trigger:
        recalled_content.append(
            MicroagentKnowledge(name=microagent.name, trigger=trigger, content=microagent.content)
        )
```
{cite}`F:openhands/memory/memory.py#215-241`

Before insertion, `ConversationMemory` filters out microagents that have already appeared earlier in the conversation to avoid bloat:

```python
def _filter_agents_in_microagent_obs(self, obs, current_index, events):
    if obs.recall_type != RecallType.KNOWLEDGE:
        return obs.microagent_knowledge
    filtered_agents = []
    for agent in obs.microagent_knowledge:
        if not self._has_agent_in_earlier_events(agent.name, current_index, events):
            filtered_agents.append(agent)
    return filtered_agents
```
{cite}`F:openhands/memory/conversation_memory.py#614-638`

This means overlapping triggers simply result in multiple blocks in the same message. There is no scoring system; deduplication only ensures the same microagent isn't injected twice.

## Breaking Down Tasks

OpenHands can delegate subtasks to other agents. In the architecture notes this is described as a multi‑agent system where a task may consist of multiple `subtasks`, each executed by an agent:

```python
async def start_delegate(self, action: AgentDelegateAction) -> None:
    """Start a delegate agent to handle a subtask."""
    # simplified excerpt
    delegate_agent = agent_cls(llm=llm, config=agent_config)
    state = State(session_id=self.id.removesuffix('-delegate'), ...)
    self.delegate = AgentController(...)
```
{cite}`F:docs/usage/architecture/llm-state-management.mdx#560-620`

This mechanism allows the system to break larger goals into smaller pieces when necessary.

## Preprocessing Pipeline

Before the user's message reaches the LLM, several preprocessing steps occur:

1. **System Prompt Injection** – `ConversationMemory` ensures a `SystemMessageAction` containing the system prompt is present. If missing, it inserts one using the `PromptManager`:

```python
def _ensure_system_message(self, events: list[Event]) -> None:
    has_system_message = any(isinstance(event, SystemMessageAction) for event in events)
    if not has_system_message:
        system_prompt = self.prompt_manager.get_system_message()
        if system_prompt:
            events.insert(0, SystemMessageAction(content=system_prompt))
```
{cite}`F:openhands/memory/conversation_memory.py#717-736`

2. **Initial User Message Check** – the second message must be the user's initial request. If not, it is inserted:

```python
if len(events) == 1:
    events.insert(1, initial_user_action)
elif not isinstance(events[1], MessageAction) or events[1].source != 'user':
    events.insert(1, initial_user_action)
```
{cite}`F:openhands/memory/conversation_memory.py#754-766`

3. **Workspace Context and Microagent Knowledge** – the combined user message
shown in the previous section is appended to the conversation. This step adds
repository information and any triggered microagents before sending the prompt
to the LLM.

4. **Condensation** – if the conversation becomes too long, a condenser summarizes earlier events. The structured summary condenser constructs a prompt like this:

```python
prompt = "You are maintaining a context-aware state summary for an interactive software agent..."
# previous summary and forgotten events are appended
messages = [Message(role='user', content=[TextContent(text=prompt)])]
response = self.llm.completion(...)
```
{cite}`F:openhands/memory/condenser/impl/structured_summary_condenser.py#218-258`

The result is returned as a `CondensationAction` which updates the event history.

After these steps, the final list of `Message` objects is sent to the LLM for completion.

## Persistent vs Transient Memory

OpenHands stores long‑term state in the `State` dataclass. When a session ends, `State.save_to_session` pickles the object and clears the message history so only essential metadata persists. Later `restore_from_session` rebuilds the state and reloads events from disk:

```python
@staticmethod
def restore_from_session(sid: str, file_store: FileStore, user_id: str | None = None) -> 'State':
    encoded = file_store.read(get_conversation_agent_state_filename(sid, user_id))
    pickled = base64.b64decode(encoded)
    state = pickle.loads(pickled)
    if state.agent_state in RESUMABLE_STATES:
        state.resume_state = state.agent_state
    else:
        state.resume_state = None
    state.agent_state = AgentState.LOADING
    return state
```
{cite}`F:openhands/controller/state/state.py#120-168`

The event history itself is transient. Each call to the LLM rebuilds the prompt from the condensed history and freshly retrieved microagents, so only the summary of prior work is remembered. This separation keeps the persistent session small while still providing the LLM with relevant context.

## Summary

OpenHands accepts free-form natural language prompts. Microagents and repository instructions augment these prompts with additional guidelines when certain keywords are present or when working in a specific project. The system can spawn delegate agents to tackle subtasks, forming a multi-agent workflow. Before calling the LLM, `ConversationMemory` injects a system prompt if needed, ensures the first user message is included, adds workspace context and microagent text, and may condense history into summaries. This preprocessing pipeline provides the LLM with structured context while keeping prompts concise.
