# Agent Architecture in OpenHands

This document explains whether OpenHands uses a single agent or multiple cooperating agents and how these agents interact with the language model. It is intended for seminar participants exploring the codebase.

## Multi‑Agent Support

OpenHands explicitly supports delegation between agents. The `AgentController` contains a `start_delegate` method which spawns another `AgentController` with a new `Agent` and its own `LLM` instance:

```python
# openhands/controller/agent_controller.py
async def start_delegate(self, action: AgentDelegateAction) -> None:
    """Start a delegate agent to handle a subtask.

    OpenHands is a multi-agentic system. A `task` is a conversation between
    OpenHands (the whole system) and the user, which might involve one or more
    inputs from the user. A `subtask` is a conversation between an agent and the
    user, or another agent.
    """
    agent_cls: type[Agent] = Agent.get_cls(action.agent)
    agent_config = self.agent_configs.get(action.agent, self.agent.config)
    llm_config = self.agent_to_llm_config.get(action.agent, self.agent.llm.config)
    llm = LLM(config=llm_config, retry_listener=self._notify_on_llm_retry)
    delegate_agent = agent_cls(llm=llm, config=agent_config)
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
{cite}`F:openhands/controller/agent_controller.py#670-719`

The docstring inside this method clarifies the terminology of *tasks* and *subtasks* and states that OpenHands is a *multi‑agentic system*.

### Example from Documentation

The developer guide further illustrates delegation with a step‑by‑step example:

```text
-- TASK STARTS (SUBTASK 0 STARTS) --
DELEGATE_LEVEL 0, ITERATION 0, LOCAL_ITERATION 0
CodeActAgent: I should request help from BrowsingAgent

-- DELEGATE STARTS (SUBTASK 1 STARTS) --

DELEGATE_LEVEL 1, ITERATION 1, LOCAL_ITERATION 0
BrowsingAgent: Let me find the answer on GitHub

DELEGATE_LEVEL 1, ITERATION 2, LOCAL_ITERATION 1
BrowsingAgent: I found the answer, let me convey the result and finish

-- DELEGATE ENDS (SUBTASK 1 ENDS) --

DELEGATE_LEVEL 0, ITERATION 3, LOCAL_ITERATION 1
CodeActAgent: I got the answer from BrowsingAgent, let me convey the result
and finish

-- TASK ENDS (SUBTASK 0 ENDS) --
```
{cite}`F:openhands/agenthub/README.md#95-142`

In this scenario the default `CodeActAgent` delegates to a `BrowsingAgent` to fetch information, then resumes control after the delegate finishes. Iteration counters are shared globally while each subtask maintains its own local iteration.

## Agent Roles

OpenHands does not define separate `PlannerAgent`, `CoderAgent`, or `FixerAgent` classes. Instead the main `CodeActAgent` performs planning, execution and communication in a single loop.

Tools for shell commands, IPython execution, file editing and browsing are injected when the agent is constructed:

```python
# openhands/agenthub/codeact_agent/codeact_agent.py
```
    # NOTE: AgentSkillsRequirement need to go before JupyterRequirement,
    # since AgentSkillsRequirement provides many Python helper functions.
    AgentSkillsRequirement(),
    JupyterRequirement(),

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        """Initializes a new instance of CodeActAgent."""
        super().__init__(llm, config)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)
        self.condenser = Condenser.from_config(self.config.condenser)
```
{cite}`F:openhands/agenthub/codeact_agent/codeact_agent.py#64-90`

The agent’s `step` method sends the condensed history to the LLM, receives a response and queues resulting actions:

```python
# openhands/agenthub/codeact_agent/codeact_agent.py
```
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        params['tools'] = check_tools(self.tools, self.llm.config)
        params['extra_body'] = {'metadata': state.to_llm_metadata(agent_name=self.name)}
        response = self.llm.completion(**params)
        actions = self.response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()
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
def load_microagents_from_dir(
    microagent_dir: Union[str, Path],
) -> tuple[dict[str, RepoMicroagent], dict[str, KnowledgeMicroagent]]:
    """Load all microagents from the given directory."""
    if isinstance(microagent_dir, str):
        microagent_dir = Path(microagent_dir)

    repo_agents = {}
    knowledge_agents = {}

    logger.debug(f'Loading agents from {microagent_dir}')
    if microagent_dir.exists():
        for file in microagent_dir.rglob('*.md'):
            if file.name == 'README.md':
                continue
            agent = BaseMicroagent.load(file, microagent_dir)
            if isinstance(agent, RepoMicroagent):
                repo_agents[agent.name] = agent
            elif isinstance(agent, KnowledgeMicroagent):
                knowledge_agents[agent.name] = agent
    logger.debug(
        f'Loaded {len(repo_agents) + len(knowledge_agents)} microagents: '
        f'{[*repo_agents.keys(), *knowledge_agents.keys()]}'
    )
    return repo_agents, knowledge_agents
```
{cite}`F:openhands/microagent/microagent.py#239-285`

The public microagents repository outlines the two sources of microagents (shareable agents and repository instructions):

```text
# OpenHands Microagents

Microagents are specialized prompts that enhance OpenHands with domain-specific
knowledge and task-specific workflows. They are loaded from two sources:

### 1. Shareable Microagents (Public)
OpenHands/microagents/
├── git.md         # Git operations
├── testing.md     # Testing practices
├── docker.md      # Docker guidelines
└── # These microagents are always loaded
    ├── pr_review.md   # PR review process
    ├── bug_fix.md     # Bug fixing workflow
    └── feature.md     # Feature implementation

### 2. Repository Instructions (Private)
Each repository can have its own instructions in `.openhands/microagents/repo.md`.
These instructions are automatically loaded when working with that repository.
```
{cite}`F:microagents/README.md#1-32`

During setup, `create_memory` loads microagents from the selected repository and stores them in `Memory` for recall later:

```python
# openhands/core/setup.py
```
    memory = Memory(
        event_stream=event_stream,
        sid=sid,
        status_callback=status_callback,
    )

    memory.set_conversation_instructions(conversation_instructions)

    if runtime:
        memory.set_runtime_info(runtime, {})
        microagents: list[BaseMicroagent] = runtime.get_microagents_from_selected_repo(
            selected_repository
        )
        memory.load_user_workspace_microagents(microagents)

        if selected_repository and repo_directory:
            memory.set_repository_info(selected_repository, repo_directory)
```
{cite}`F:openhands/core/setup.py#148-170`

## Session and State Tracking

The `State` dataclass records global and local iteration counters and a `delegate_level` value so that nested subtasks are tracked correctly:

```python
# openhands/controller/state/state.py
```
class State:
    """Represents the running state of an agent."""

    # multi-agent state
    session_id: str = ''
    iteration: int = 0          # global iteration for the task
    local_iteration: int = 0    # local iteration for the subtask
    max_iterations: int = 100
    confirmation_mode: bool = False
    history: list[Event] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    agent_state: AgentState = AgentState.LOADING
    resume_state: AgentState | None = None
    traffic_control_state: TrafficControlState = TrafficControlState.NORMAL
    metrics: Metrics = field(default_factory=Metrics)
    local_metrics: Metrics = field(default_factory=Metrics)
    delegate_level: int = 0
    start_id: int = -1
    end_id: int = -1
```
{cite}`F:openhands/controller/state/state.py#44-84`

The current state is saved to disk between runs, ensuring that tasks can be resumed with the full history intact.

While OpenHands can spawn multiple agents, they execute sequentially. A parent
agent waits for its delegate to finish before continuing its own loop. This
ensures that only one agent is active at a time and simplifies state
management.

## Conclusion

OpenHands is a **multi‑agent** system. Delegation is optional, so simple tasks may only involve the main `CodeActAgent`, but the architecture allows that agent to spawn specialized sub‑agents when needed. Each agent interacts with its own `LLM` instance and shares high‑level state through the event stream. Specialized knowledge or workflows are injected via markdown microagents rather than through separate agent classes.
