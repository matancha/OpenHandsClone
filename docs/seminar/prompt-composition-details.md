# Prompt Composition Internals

This document explains in detail how OpenHands builds prompts for the LLM, including template usage, context insertion, and the summarization mechanism used when the conversation grows too large. It is intended for seminar participants exploring the repository.

These layers ensure consistent behavior across turns, inject relevant contextual knowledge, and help the LLM recover from long or complex sessions.

## Template-Based Prompts

`PromptManager` loads a set of Jinja2 templates from the `prompts` folder when an agent starts. It expects files named `system_prompt.j2`, `user_prompt.j2`, `additional_info.j2` and `microagent_info.j2`:

```python
self.system_template: Template = self._load_template('system_prompt')
self.user_template: Template = self._load_template('user_prompt')
self.additional_info_template: Template = self._load_template('additional_info')
self.microagent_info_template: Template = self._load_template('microagent_info')
```
{cite}`F:openhands/utils/prompt.py#52-60`

The system template defines global guidelines such as coding practices and version control rules:

```jinja
{% raw %}You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively...{% endraw %}
```
{cite}`F:openhands/agenthub/codeact_agent/prompts/system_prompt.j2#1-31`

Microagent instructions are inserted with another template that wraps each block in `<EXTRA_INFO>` tags:

```jinja
{% raw %}{% for agent_info in triggered_agents %}
<EXTRA_INFO>
The following information has been included based on a keyword match for "{{ agent_info.trigger }}".
It may or may not be relevant to the user's request.

{{ agent_info.content }}
</EXTRA_INFO>
{% endfor %}{% endraw %}
```
{cite}`F:openhands/agenthub/codeact_agent/prompts/microagent_info.j2#1-8`

These templates keep prompts consistent across iterations and agents.

## Constructing the Message List

`ConversationMemory.process_events` turns the condensed history into `Message` objects for the LLM. When a workspace context observation is encountered, repository information, runtime data and any triggered microagents are combined into a single user message:

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
{cite}`F:openhands/memory/conversation_memory.py#511-535`

Each element added to `message_content` is a **TextContent** object. This wrapper
is later flattened by `Message.serialize_model` into either a single string or a
list of structured chunks before the call to `LLM.completion`. This keeps the
intermediate representation flexible while still conforming to provider APIs
that expect role-based message entries.

At this point the message stack conceptually looks like:

```
[System Prompt] -> [Initial User Message] -> [Workspace Context + Microagent Info] -> [...follow-up messages]
```

Including a simple diagram of this flow can help the audience visualize how prompts are layered.

The helper methods `_ensure_system_message` and `_ensure_initial_user_message` guarantee that the conversation always begins with a system prompt followed by the first user request:

```python
def _ensure_system_message(self, events: list[Event]) -> None:
    has_system_message = any(isinstance(event, SystemMessageAction) for event in events)
    if not has_system_message:
        system_prompt = self.prompt_manager.get_system_message()
        if system_prompt:
            events.insert(0, SystemMessageAction(content=system_prompt))
```
{cite}`F:openhands/memory/conversation_memory.py#717-734`

```python
def _ensure_initial_user_message(self, events: list[Event], initial_user_action: MessageAction) -> None:
    if len(events) == 1:
        events.insert(1, initial_user_action)
    elif not isinstance(events[1], MessageAction) or events[1].source != 'user':
        events.insert(1, initial_user_action)
```
{cite}`F:openhands/memory/conversation_memory.py#739-766`

After conversion, the list of `Message` objects is passed to the LLM for completion.

## Context Summarization

OpenHands supports several condenser strategies, including `StructuredSummaryCondenser`, `LLMSummarizingCondenser`, `RecentEventsCondenser` and `ObservationMaskingCondenser`. When event history exceeds the configured window, these condensers reduce it. The `StructuredSummaryCondenser` summarizes forgotten events by prompting an LLM with the previous summary and a list of events to be dropped:

```python
prompt = "You are maintaining a context-aware state summary for an interactive software agent..."
summary_event_content = self._truncate(
    summary_event.message if summary_event.message else ''
)
for forgotten_event in forgotten_events:
    event_content = self._truncate(str(forgotten_event))
    prompt += f'<EVENT id={forgotten_event.id}>\n{event_content}\n</EVENT>\n'
response = self.llm.completion(
    messages=self.llm.format_messages_for_llm(messages),
    tools=[StateSummary.tool_description()],
    tool_choice={'type': 'function', 'function': {'name': 'create_state_summary'}},
)
```
{cite}`F:openhands/memory/condenser/impl/structured_summary_condenser.py#214-261`

The returned `StateSummary` is stored inside a `CondensationAction` so the `View` can later reinsert this summary at a fixed offset:

```python
return Condensation(
    action=CondensationAction(
        forgotten_events_start_id=min(event.id for event in forgotten_events),
        forgotten_events_end_id=max(event.id for event in forgotten_events),
        summary=str(summary),
        summary_offset=self.keep_first,
    )
)
```
{cite}`F:openhands/memory/condenser/impl/structured_summary_condenser.py#296-305`

`View.from_events` then injects that summary back into the list when reconstructing the conversation:

```python
for event in reversed(events):
    if isinstance(event, CondensationAction):
        if event.summary is not None and event.summary_offset is not None:
            summary = event.summary
            summary_offset = event.summary_offset
            break
if summary is not None and summary_offset is not None:
    kept_events.insert(
        summary_offset, AgentCondensationObservation(content=summary)
    )
```
{cite}`F:openhands/memory/view.py#60-75`

## Prompt Compression

OpenHands includes several condenser strategies. The `RecentEventsCondenser` simply keeps the newest events, while the `ObservationMaskingCondenser` replaces old observation bodies with `<MASKED>` markers. The structured summary condenser above provides the most detailed compression, transforming many events into a single summary to keep the history within token limits.
The `StructuredSummaryCondenser` is usually enabled explicitly when a rolling
summary of all past work is desired; otherwise the lighter `LLMSummarizingCondenser`
is the typical default.

### Condenser Selection and Configuration

Which condenser runs depends on configuration. When `enable_default_condenser` is
true (see `config.template.toml`), the platform inserts an `LLMSummarizingCondenser`
by default if no condenser section is specified. If this flag is false, a
`NoOpCondenser` preserves the full history instead. Users can explicitly choose
other strategies in the `[condenser]` section of their configuration file.
The relevant logic can be seen in `config/utils.py` where a default
`LLMSummarizingCondenser` is added if no condenser is configured, and in
`session.py` where a pipeline condenser is attached when sessions start.
{cite}`F:openhands/core/config/utils.py#248-268`
{cite}`F:openhands/server/session/session.py#149-171`

Summarization ensures the LLM prompt remains concise yet informative even for long sessions.

## Conclusion

Prompts in OpenHands are built from reusable templates. `ConversationMemory` collects system guidelines, repository context and microagent instructions into a structured list of messages, always beginning with the system prompt and original user request. When the conversation grows too large, a condenser summarizes older events using the LLM itself and injects this summary back into the history. Together these mechanisms maintain continuity across iterations while respecting context window limits.
