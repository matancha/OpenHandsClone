# Frontend vs Backend Roles and Orchestration

This document explains how user interactions flow through OpenHands and where orchestration happens. It is intended for seminar participants exploring the codebase.

## 1. Frontend Role

OpenHands has both a CLI and a web UI. The CLI logic lives in `openhands/cli/` while the server serving the React frontend lives in `openhands/server/`.

The CLI collects prompts from the user whenever the agent reaches `AWAITING_USER_INPUT` or `AWAITING_USER_CONFIRMATION`:

```python
# openhands/cli/main.py
async def prompt_for_next_task(agent_state: str) -> None:
    while True:
        next_message = await read_prompt_input(
            agent_state, multiline=config.cli_multiline_input
        )
        if not next_message.strip():
            continue
        ...
```

When events indicate the agent needs confirmation, the CLI calls `read_confirmation_input` before sending a response to the event stream.

The web frontend communicates with the backend via HTTP and WebSocket endpoints. For example the `/events` endpoint posts user messages into the event stream:

```python
# openhands/server/routes/conversation.py
@app.post('/events')
async def add_event(request: Request, conversation: ServerConversation = Depends(get_conversation)):
    data = request.json()
    await conversation_manager.send_to_event_stream(conversation.sid, data)
    return JSONResponse({'success': True})
```

Thus the frontend (CLI or browser) is responsible for:

- Gathering the initial task and any follow‑up prompts from the user.
- Displaying events from the agent (messages, observations, errors).
- Sending additional input or confirmation back to the backend.

It does **not** directly execute agent logic—it merely relays user input and renders output.

## 2. Backend Role

The backend initializes the agent, runtime and memory, then drives the main loop. In `openhands/core/main.py`, `run_controller` creates these components and inserts the initial user action into the `EventStream`:

```python
# openhands/core/main.py
controller, initial_state = create_controller(agent, runtime, config)
...
if initial_state is not None and initial_state.last_error:
    event_stream.add_event(
        MessageAction(content="Let's get back on track..."),
        EventSource.USER,
    )
else:
    event_stream.add_event(initial_user_action, EventSource.USER)
```

`run_agent_until_done` then drives the controller by repeatedly calling `controller.step` until a terminal state is reached:

```python
# openhands/core/loop.py
async def run_agent_until_done(controller: AgentController, runtime: Runtime, memory: Memory, end_states: list[AgentState]) -> None:
    runtime.status_callback = status_callback
    controller.status_callback = status_callback
    memory.status_callback = status_callback
    while controller.state.agent_state not in end_states:
        await asyncio.sleep(1)
```

The server variant uses `ConversationManager` to manage sessions and attach front‑end connections. When the frontend posts an event, `ConversationManager.send_to_event_stream` injects it into the running agent loop.

## 3. Prompt‑Orchestrated Behavior

Inside the agent’s `step` method (`CodeActAgent.step`) the backend constructs messages for the LLM and parses the response into actions:

```python
# openhands/agenthub/codeact_agent/codeact_agent.py
messages = self._get_messages(condensed_history, initial_user_message)
params = {
    'messages': self.llm.format_messages_for_llm(messages),
    'tools': check_tools(self.tools, self.llm.config),
    'extra_body': {'metadata': state.to_llm_metadata(agent_name=self.name)},
}
response = self.llm.completion(**params)
actions = self.response_to_actions(response)
for action in actions:
    self.pending_actions.append(action)
return self.pending_actions.popleft()
```

However, the high‑level workflow the LLM follows (exploration → analysis → testing → implementation) is embedded in the system prompt template:

```jinja
<!-- openhands/agenthub/codeact_agent/prompts/system_prompt.j2 -->
<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING: ...
4. IMPLEMENTATION: ...
5. VERIFICATION: ...
</PROBLEM_SOLVING_WORKFLOW>
```

Therefore the backend provides the event loop and tool execution framework, while the prompt guides the model on *how* to behave at each step.

## 4. Where Orchestration Lives

- **Backend**: Controls session management, event streaming, when to call the LLM, how to execute tool actions, and how to maintain state across iterations.
- **Prompts**: Tell the LLM the overall workflow and coding conventions. The LLM uses these instructions to self‑direct within the backend’s loop.

In practice, orchestration is shared. The backend ensures consistent stepping and tool execution, while the prompts embed the high‑level strategy the agent should follow.
