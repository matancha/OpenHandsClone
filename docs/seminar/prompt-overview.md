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

3. **Workspace Context and Microagent Knowledge** – any repository info, runtime instructions and triggered microagent content are rendered through templates and inserted as a user message:

```python
formatted_workspace_text = self.prompt_manager.build_workspace_context(
    repository_info=repo_info,
    runtime_info=runtime_info,
    conversation_instructions=conversation_instructions,
    repo_instructions=repo_instructions,
)
message_content.append(TextContent(text=formatted_workspace_text))
# Microagent knowledge
formatted_microagent_text = self.prompt_manager.build_microagent_info(
    triggered_agents=filtered_agents,
)
message_content.append(TextContent(text=formatted_microagent_text))
```
{cite}`F:openhands/memory/conversation_memory.py#492-535`

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

## Summary

OpenHands accepts free-form natural language prompts. Microagents and repository instructions augment these prompts with additional guidelines when certain keywords are present or when working in a specific project. The system can spawn delegate agents to tackle subtasks, forming a multi-agent workflow. Before calling the LLM, `ConversationMemory` injects a system prompt if needed, ensures the first user message is included, adds workspace context and microagent text, and may condense history into summaries. This preprocessing pipeline provides the LLM with structured context while keeping prompts concise.
