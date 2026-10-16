# Hybrid V2 Multi-Agent Execution Strategy Guide

## How the System Decides Which Strategy to Use

The Hybrid V2 Multi-Agent Executor uses a sophisticated decision-making system that operates at multiple levels to determine execution strategies and agent coordination.

## 1. Strategy Selection Hierarchy

### Level 1: Initial Strategy Selection
```python
# Default strategy when no specific strategy is provided
ExecutionStrategy.DYNAMIC  # Default - most intelligent option

# Available strategies
ExecutionStrategy.SEQUENTIAL  # Traditional linear execution
ExecutionStrategy.PARALLEL   # Simultaneous execution
ExecutionStrategy.CONDITIONAL # Condition-based execution  
ExecutionStrategy.DYNAMIC    # Runtime intelligent selection
```

### Level 2: Manager Agent Planning
The **Manager Agent** receives the user request and creates an intelligent execution plan:

```
Input to Manager:
- User Request: "Analyze sales data and create report"
- Available Agents: [data_cleaner, analyzer, reporter, visualizer]
- Requested Strategy: DYNAMIC
- File/History context

Output from Manager:
[{
    "name": "data_preparation",
    "strategy": "sequential",
    "agents": ["data_cleaner"],
    "domain": "data"
}, {
    "name": "analysis_phase", 
    "strategy": "parallel",
    "agents": ["analyzer", "visualizer"],
    "domain": "analysis"
}, {
    "name": "output_phase",
    "strategy": "sequential", 
    "agents": ["reporter"],
    "domain": "output"
}]
```

### Level 3: Domain Manager Decision-Making
Each domain has specialized managers that make runtime decisions:

- **DataDomainManager**: Decides data processing agents
- **AnalysisDomainManager**: Chooses analysis approaches  
- **SynthesisDomainManager**: Manages integration logic
- **OutputDomainManager**: Controls output formatting

## 2. Execution Strategy Details

### SEQUENTIAL Strategy
**When Used**: Linear processing where each step depends on the previous
**How It Works**:
1. Execute agents one by one in defined order
2. Each agent's output becomes the next agent's input
3. Perfect for workflows with strict dependencies

```python
async def _execute_sequential_v3(self, phase, phase_input, history_context):
    current_input = phase_input
    
    for agent_id in phase.agents:
        # Domain manager checks if agent should run
        if domain_manager.should_execute_agent(agent_id, current_input):
            result = await executor.execute_task(...)
            current_input = result  # Chain outputs
        else:
            # Skip agent based on domain logic
            context.log_skipped_agent(agent_id, "Domain decision")
    
    return current_input
```

**Decision Factors**:
- Strong dependencies between agents
- Sequential data processing required
- Risk of conflicts in parallel execution

### PARALLEL Strategy  
**When Used**: Independent tasks that can run simultaneously
**How It Works**:
1. Identify agents that can work on the same input
2. Execute all agents simultaneously using asyncio
3. Collect and combine results

```python
async def _execute_parallel_v3(self, phase, phase_input, history_context):
    tasks = []
    
    for agent_id in phase.agents:
        if domain_manager.should_execute_agent(agent_id, phase_input):
            task = executor.execute_task(...)
            tasks.append((agent_id, task))
    
    # Execute all in parallel
    results = await asyncio.gather(*[task for _, task in tasks])
    
    return combined_results_dict
```

**Decision Factors**:
- Agents can work independently
- No shared resource conflicts  
- Time-sensitive processing
- Multiple perspectives needed

### CONDITIONAL Strategy
**When Used**: Runtime conditions determine which agents execute
**How It Works**:
1. Evaluate conditions for each agent
2. Execute only agents whose conditions are met
3. Skip agents that don't meet criteria

```python
async def _execute_conditional_v3(self, phase, phase_input, history_context):
    # Domain manager evaluates conditions
    conditions_met = await domain_manager.evaluate_conditions(
        phase.conditions, execution_context
    )
    
    results = {}
    for agent_id in phase.agents:
        if conditions_met.get(agent_id, False):
            result = await executor.execute_task(...)
            results[agent_id] = result
        else:
            context.log_skipped_agent(agent_id, "Condition not met")
    
    return results
```

**Decision Factors**:
- Data-dependent execution paths
- Optional processing steps
- Resource availability checks
- Quality gates and validations

### DYNAMIC Strategy
**When Used**: Real-time agent selection based on intermediate results  
**How It Works**:
1. Domain manager chooses next agent based on current state
2. Execute chosen agent
3. Use result to determine next agent
4. Continue until completion criteria met

```python
async def _execute_dynamic_v3(self, phase, phase_input, history_context):
    results = []
    current_input = phase_input
    iteration = 0
    
    while iteration < max_iterations:
        # Domain manager chooses next agent intelligently
        next_agent = await domain_manager.choose_next_agent(
            available_agents=phase.agents,
            current_input=current_input,
            context=execution_context,
            execution_history=results
        )
        
        if not next_agent:
            break  # No more agents needed
            
        result = await executor.execute_task(...)
        results.append({
            "agent_id": next_agent,
            "result": result,
            "iteration": iteration
        })
        
        current_input = result  # Result informs next choice
        iteration += 1
    
    return results
```

**Decision Factors**:
- Complex problem-solving workflows
- Adaptive processing requirements
- Unknown optimal path ahead of time
- Multi-step reasoning needed

## 3. Domain Manager Intelligence

### DataDomainManager Decision Logic
```python
async def should_execute_agent(self, agent_id, input_data, context):
    # Skip data cleaner if data appears clean
    if agent_id.endswith("_cleaner") and self._is_data_clean(input_data):
        return False
    
    # Skip validator if previous validation passed
    if agent_id.endswith("_validator") and context.has_validation_passed():
        return False
        
    return True
```

### AnalysisDomainManager Decision Logic  
```python
async def choose_next_agent(self, available_agents, current_input, context, history):
    # Choose heavy-duty analyzers for large datasets
    if self._is_large_dataset(current_input):
        heavy_agents = [a for a in available_agents if "heavy" in a]
        if heavy_agents:
            return heavy_agents[0]
    
    # Choose statistical agents for numerical data
    if self._is_numerical_data(current_input):
        stats_agents = [a for a in available_agents if "stats" in a]
        if stats_agents:
            return stats_agents[0]
    
    return available_agents[0]  # Default choice
```

## 4. Strategy Selection Examples

### Example 1: Data Analysis Pipeline
```
User Input: "Analyze customer sales data and create dashboard"

Manager Decision:
Phase 1 (Data): SEQUENTIAL
  - data_cleaner → data_validator → data_preprocessor
  
Phase 2 (Analysis): PARALLEL  
  - trend_analyzer + pattern_detector + segment_analyzer
  
Phase 3 (Output): CONDITIONAL
  - dashboard_creator (if analysis successful)
  - error_reporter (if analysis failed)
```

### Example 2: Content Generation
```
User Input: "Create marketing content for new product"

Manager Decision:
Phase 1 (Research): DYNAMIC
  - Start with market_researcher
  - Based on findings, choose competitor_analyzer OR customer_surveyor
  - Adapt based on intermediate results

Phase 2 (Creation): PARALLEL
  - content_writer + designer + social_media_specialist
  
Phase 3 (Review): SEQUENTIAL
  - editor → approver → publisher
```

### Example 3: Complex Problem Solving
```
User Input: "Optimize our delivery routes and reduce costs"

Manager Decision:
Phase 1 (Data Collection): PARALLEL
  - route_analyzer + cost_calculator + traffic_predictor

Phase 2 (Optimization): DYNAMIC
  - Start with heuristic_optimizer
  - If insufficient, escalate to ml_optimizer
  - If still poor, try genetic_algorithm_optimizer

Phase 3 (Validation): CONDITIONAL
  - simulation_validator (if optimization found)
  - fallback_recommender (if no optimization found)
```

## 5. Fallback Mechanisms

### When Intelligent Planning Fails
```python
def _create_fallback_execution_plan(self, strategy):
    if strategy == ExecutionStrategy.PARALLEL:
        # Group by domain
        return phases_grouped_by_domain
    else:
        # Simple sequential fallback
        return [ExecutionPhase(
            name="main_phase",
            strategy=ExecutionStrategy.SEQUENTIAL,
            agents=all_agent_ids
        )]
```

### When Strategy Execution Fails
- V3 execution failures gracefully fall back to V2 sequential mode
- Phase failures continue with error propagation
- Agent failures skip to next agent in phase

## 6. Benefits of This Approach

1. **Intelligence**: Manager agent makes contextual decisions
2. **Flexibility**: Multiple strategies for different scenarios  
3. **Efficiency**: Parallel execution when possible
4. **Reliability**: Comprehensive fallback mechanisms
5. **Adaptability**: Dynamic strategies adjust based on results
6. **Scalability**: Domain managers handle complexity growth

## 7. Configuration and Usage

### Basic Usage
```python
# Use default DYNAMIC strategy
result = await executor.execute_task_v3(user_input)

# Specify strategy
result = await executor.execute_parallel(user_input)  
result = await executor.execute_sequential(user_input)
result = await executor.execute_conditional(user_input) 
result = await executor.execute_dynamic(user_input)
```

### Custom Phase Creation
```python
# Create custom execution plan
phases = [
    executor.create_custom_phase(
        name="data_prep",
        agents=["cleaner", "validator"], 
        strategy=ExecutionStrategy.SEQUENTIAL
    ),
    executor.create_custom_phase(
        name="analysis",
        agents=["analyzer1", "analyzer2"],
        strategy=ExecutionStrategy.PARALLEL
    )
]

result = await executor.execute_task_v3(
    user_input, 
    custom_phases=phases
)
```

This hybrid approach combines the reliability of proven single-agent execution with the intelligence and efficiency of multi-agent coordination, making it suitable for a wide range of complex workflows and problem-solving scenarios.
