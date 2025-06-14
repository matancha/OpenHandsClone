# Agent Architecture in OpenHands

This document explains whether OpenHands uses a single agent or multiple cooperating agents and how these agents interact with the language model. It is intended for seminar participants exploring the codebase.

## Multi‑Agent Support

OpenHands explicitly supports delegation between agents. The `AgentController` contains a `start_delegate` method which spawns another `AgentController` with a new `Agent` and its own `LLM` instance:

```python
# openhands/controller/agent_controller.py
```
{cite}`F:openhands/controller/agent_controller.py#670-719`

The docstring inside this method clarifies the terminology of *tasks* and *subtasks* and states that OpenHands is a *multi‑agentic system*.

### Example from Documentation

The developer guide further illustrates delegation with a step‑by‑step example:

```text
```
{cite}`F:openhands/agenthub/README.md#95-142`

In this scenario the default `CodeActAgent` delegates to a `BrowsingAgent` to fetch information, then resumes control after the delegate finishes. Iteration counters are shared globally while each subtask maintains its own local iteration.

## Agent Roles

OpenHands does not define separate `PlannerAgent`, `CoderAgent`, or `FixerAgent` classes. Instead the main `CodeActAgent` performs planning, execution and communication in a single loop.

Tools for shell commands, IPython execution, file editing and browsing are injected when the agent is constructed:

```python
# openhands/agenthub/codeact_agent/codeact_agent.py
```
{cite}`F:openhands/agenthub/codeact_agent/codeact_agent.py#64-90`

The agent’s `step` method sends the condensed history to the LLM, receives a response and queues resulting actions:

```python
# openhands/agenthub/codeact_agent/codeact_agent.py
```
{cite}`F:openhands/agenthub/codeact_agent/codeact_agent.py#160-196`

Delegation is triggered by emitting an `AgentDelegateAction`. The parent controller then launches a new agent with its own LLM configuration.

```python
actions.append(
    AgentDelegateAction(
        agent='BrowsingAgent',
        inputs={'query': 'Search for recent API changes...'}
    )
)
```

This action is queued inside `response_to_actions` and causes `start_delegate` to
spawn the browsing agent.

## Microagents

Domain‑specific behavior is modularized using *microagents* – markdown files that contain instructions, optional triggers and metadata. They are loaded at runtime and appended to the user prompt when relevant.

The microagent loader iterates over a directory and categorizes each file as repository knowledge or general knowledge:

```python
# openhands/microagent/microagent.py
```
{cite}`F:openhands/microagent/microagent.py#239-285`

The public microagents repository outlines the two sources of microagents (shareable agents and repository instructions):

```text
```
{cite}`F:microagents/README.md#1-32`

During setup, `create_memory` loads microagents from the selected repository and stores them in `Memory` for recall later:

```python
# openhands/core/setup.py
```
{cite}`F:openhands/core/setup.py#148-170`

## Session and State Tracking

The `State` dataclass records global and local iteration counters and a `delegate_level` value so that nested subtasks are tracked correctly:

```python
# openhands/controller/state/state.py
```
{cite}`F:openhands/controller/state/state.py#44-84`

The current state is saved to disk between runs, ensuring that tasks can be resumed with the full history intact.

While OpenHands can spawn multiple agents, they execute sequentially. A parent
agent waits for its delegate to finish before continuing its own loop. This
ensures that only one agent is active at a time and simplifies state
management.

## Conclusion

OpenHands is a **multi‑agent** system. Delegation is optional, so simple tasks may only involve the main `CodeActAgent`, but the architecture allows that agent to spawn specialized sub‑agents when needed. Each agent interacts with its own `LLM` instance and shares high‑level state through the event stream. Specialized knowledge or workflows are injected via markdown microagents rather than through separate agent classes.
