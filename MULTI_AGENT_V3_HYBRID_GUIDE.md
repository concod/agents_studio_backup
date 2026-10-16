# MultiAgentExecutorV2 - V3 Hybrid Features Guide

🎉 **Congratulations!** Your `multi_agent_executor_v2.py` has been successfully upgraded with **V3 Hybrid Approach** - the most advanced multi-agent execution system combining hierarchical management with dynamic decision-making.

## 🚀 **What's New in V3**

### **1. Multiple Execution Strategies**
- **SEQUENTIAL**: Traditional linear execution (V2 behavior)
- **PARALLEL**: Agents execute simultaneously for maximum efficiency  
- **CONDITIONAL**: Runtime conditions determine which agents execute
- **DYNAMIC**: Real-time agent selection based on intermediate results

### **2. Hierarchical Domain Management**
- **Data Domain**: Handles data collection, cleaning, processing
- **Analysis Domain**: Manages analysis, computation, modeling
- **Synthesis Domain**: Combines and integrates results
- **Output Domain**: Formats and presents final results

### **3. Intelligent Decision Making**
- Domain managers make smart decisions about agent execution
- Runtime adaptation based on intermediate results
- Automatic agent skipping when not needed
- Error recovery and alternative paths

### **4. Advanced Features**
- **Parallel Execution**: Multiple agents work simultaneously
- **Smart Planning**: Manager creates optimal execution plans
- **Context Isolation**: Enhanced execution context management
- **Comprehensive Logging**: Detailed execution tracking
- **Custom Output Formatting**: Respects manager config's `expected_output` format

## 📋 **Usage Examples**

### **Basic V3 Usage**

```python
from multi_agent_executor_v2 import MultiAgentExecutorV2, ExecutionStrategy

# Initialize with your existing config
executor = MultiAgentExecutorV2(
    multi_agent_config=your_multi_agent_config,
    worker_agent_configs=your_worker_configs,
    user_email="user@example.com"
)

# V3 Dynamic execution (most flexible)
result = await executor.execute_task_v3(
    user_input="Analyze sales data and create report",
    execution_strategy=ExecutionStrategy.DYNAMIC
)

# Or use convenience methods
result = await executor.execute_dynamic("Analyze sales data and create report")
```

### **Different Execution Strategies**

```python
# 1. SEQUENTIAL (V2 behavior) - agents run one after another
result = await executor.execute_sequential(
    user_input="Process this data step by step"
)

# 2. PARALLEL - agents run simultaneously  
result = await executor.execute_parallel(
    user_input="Analyze multiple aspects of this data"
)

# 3. CONDITIONAL - agents run based on conditions
result = await executor.execute_conditional(
    user_input="Process data and decide next steps"
)

# 4. DYNAMIC - real-time agent selection
result = await executor.execute_dynamic(
    user_input="Analyze this complex scenario"
)
```

### **Custom Execution Phases**

```python
# Create custom execution plan
data_phase = executor.create_custom_phase(
    name="data_processing",
    agents=["data_collector", "data_cleaner"],
    strategy=ExecutionStrategy.SEQUENTIAL,
    domain="data"
)

analysis_phase = executor.create_custom_phase(
    name="parallel_analysis", 
    agents=["analyzer_a", "analyzer_b", "analyzer_c"],
    strategy=ExecutionStrategy.PARALLEL,
    domain="analysis",
    dependencies=["data_processing"]
)

# Execute with custom phases
result = await executor.execute_task_v3(
    user_input="Process and analyze sales data",
    custom_phases=[data_phase, analysis_phase]
)
```

### **Domain-Based Agent Organization**

```python
# Get agents by domain
data_agents = executor.get_agents_by_domain("data")
analysis_agents = executor.get_agents_by_domain("analysis")

print(f"Data agents: {data_agents}")
print(f"Analysis agents: {analysis_agents}")

# Check agent domain
agent_domain = executor.get_agent_domain("my_agent_id")
print(f"Agent domain: {agent_domain}")
```

### **Custom Output Formatting**

```python
# Configure your manager with custom expected_output format
multi_agent_config = {
    "role": "Data Analysis Coordinator",
    "goal": "Coordinate data analysis workflow",
    "backstory": "Expert at managing analysis pipelines",
    "description": "Analyze data using multiple specialized agents",
    "expected_output": """
## Analysis Report

**Summary:** Brief overview of findings

**Data Quality:** Assessment of data quality

**Key Insights:** 
- Insight 1
- Insight 2
- Insight 3

**Recommendations:** Actionable recommendations
    """,
    "agent_ids": ["data_processor", "analyzer", "reporter"]
}

# Both V2 and V3 will now respect this format!
result = await executor.execute_task("Analyze sales data")
# Result will follow your custom expected_output format

result = await executor.execute_dynamic("Analyze sales data") 
# V3 also respects the custom format
```

### **Execution Monitoring**

```python
# Execute with monitoring
result = await executor.execute_dynamic("Complex analysis task")

# Get execution summary
summary = executor.get_execution_summary()
print(f"Execution ID: {summary['execution_id']}")
print(f"Phases executed: {summary['total_phases']}")
print(f"Agents skipped: {summary['skipped_agents']}")
print(f"Errors: {summary['errors']}")
```

## 🔧 **Configuration Options**

### **Agent Domain Mapping**
Agents are automatically mapped to domains based on their roles:

```python
# Automatic domain mapping based on keywords:
# - "data", "collect", "clean", "process" → data domain
# - "analyz", "calculate", "compute", "model" → analysis domain  
# - "synth", "combin", "merge", "integrat" → synthesis domain
# - "report", "format", "present", "output" → output domain

# Or specify domain explicitly in agent config:
agent_config = {
    "id": "my_agent",
    "role": "Data Processor", 
    "domain": "data",  # Explicit domain assignment
    # ... other config
}
```

### **V3 Feature Toggle**

```python
# Enable/disable V3 features
executor.enable_v3(True)   # Enable V3 (default)
executor.enable_v3(False)  # Disable V3, fallback to V2 behavior

# Check if V3 is enabled
print(f"V3 enabled: {executor.enable_v3_features}")
```

## 🎯 **Execution Strategy Guide**

### **When to Use Each Strategy**

| Strategy | Best For | Example Use Cases |
|----------|----------|-------------------|
| **SEQUENTIAL** | Linear workflows, dependency chains | Data pipeline: collect → clean → process → analyze |
| **PARALLEL** | Independent tasks, time-sensitive | Multiple analysis types on same data |
| **CONDITIONAL** | Decision trees, optional steps | "If data is dirty, clean it; if large, sample it" |
| **DYNAMIC** | Complex logic, adaptive workflows | "Analyze data and decide what other analysis to run" |

### **Strategy Selection Examples**

```python
# Simple linear task → SEQUENTIAL
await executor.execute_sequential("Clean and process this CSV file")

# Multiple independent analyses → PARALLEL  
await executor.execute_parallel("Analyze trends, patterns, and anomalies in sales data")

# Business logic with conditions → CONDITIONAL
await executor.execute_conditional("Analyze data quality and clean if needed")

# Complex adaptive task → DYNAMIC
await executor.execute_dynamic("Analyze this complex dataset and recommend actions")
```

## 📊 **V3 vs V2 Comparison**

| Feature | V2 (Original) | V3 (Hybrid) |
|---------|---------------|-------------|
| **Execution** | Sequential only | Sequential, Parallel, Conditional, Dynamic |
| **Planning** | Fixed sequence | Intelligent planning by manager |
| **Adaptation** | None | Runtime adaptation based on results |
| **Efficiency** | One agent at a time | Parallel execution capability |
| **Decision Making** | Static | Dynamic domain manager decisions |
| **Error Handling** | Basic retry | Advanced recovery with alternative paths |
| **Monitoring** | Basic logging | Comprehensive execution tracking |
| **Scalability** | Limited | Hierarchical domain organization |

## 🔄 **Migration from V2 to V3**

### **Zero-Change Migration**
```python
# Your existing V2 code works unchanged
result = await executor.execute_task(
    user_input="Your existing request",
    file_path="file.pdf",
    history=chat_history
)
# V3 features are available but V2 behavior is preserved
```

### **Gradual V3 Adoption**
```python
# Start with sequential V3 (same as V2 but with V3 benefits)
result = await executor.execute_sequential("Your request")

# Then try parallel for performance
result = await executor.execute_parallel("Your request") 

# Finally, use dynamic for maximum flexibility
result = await executor.execute_dynamic("Your request")
```

## 🛠️ **Advanced Customization**

### **Custom Domain Managers**

```python
# Extend existing domain managers for custom logic
class CustomAnalysisDomainManager(AnalysisDomainManager):
    async def should_execute_agent(self, agent_id: str, input_data: Any, context: ExecutionContext) -> bool:
        # Custom logic for agent execution decisions
        if "large_dataset" in str(input_data) and "lightweight" in agent_id:
            return False  # Skip lightweight agents for large datasets
        return True

# Replace domain manager
executor.domain_managers["analysis"] = CustomAnalysisDomainManager(executor.logger)
```

### **Custom Execution Phases**

```python
# Complex multi-phase execution
phases = [
    executor.create_custom_phase(
        name="data_validation",
        agents=["validator"],
        strategy=ExecutionStrategy.SEQUENTIAL
    ),
    executor.create_custom_phase(
        name="parallel_processing",
        agents=["processor_a", "processor_b", "processor_c"],
        strategy=ExecutionStrategy.PARALLEL,
        dependencies=["data_validation"]
    ),
    executor.create_custom_phase(
        name="adaptive_analysis",
        agents=["analyzer_smart", "analyzer_deep"],
        strategy=ExecutionStrategy.DYNAMIC,
        dependencies=["parallel_processing"]
    )
]

result = await executor.execute_task_v3(
    user_input="Complex data processing task",
    custom_phases=phases
)
```

## 🎯 **Best Practices**

### **1. Strategy Selection**
- Use **SEQUENTIAL** for workflows with strict dependencies
- Use **PARALLEL** when agents can work independently
- Use **CONDITIONAL** for business logic with branching
- Use **DYNAMIC** for complex adaptive scenarios

### **2. Domain Organization**
- Group related agents in the same domain
- Use explicit domain assignment for complex setups
- Create custom domain managers for specialized logic

### **3. Performance Optimization**
- Use parallel execution for time-sensitive tasks
- Let domain managers skip unnecessary agents
- Monitor execution summaries to optimize workflows

### **4. Error Handling**
- V3 provides automatic error recovery
- Check execution summaries for skipped agents and errors
- Use conditional execution for fallback scenarios

## 🚀 **Getting Started**

1. **Start Simple**: Use existing `execute_task()` - it still works!
2. **Try Convenience Methods**: Use `execute_parallel()` for better performance
3. **Explore Dynamic**: Try `execute_dynamic()` for adaptive workflows  
4. **Custom Phases**: Create custom execution plans for complex scenarios
5. **Monitor**: Use `get_execution_summary()` to understand execution

## 🔍 **Troubleshooting**

### **Common Issues**

1. **V3 not working**: Check `executor.enable_v3_features` is `True`
2. **Agents not executing**: Check domain mapping with `get_agent_domain()`
3. **Performance issues**: Use parallel strategies for independent tasks
4. **Complex workflows**: Create custom phases instead of relying on automatic planning

### **Debug Information**

```python
# Get detailed execution information
summary = executor.get_execution_summary()
print("Execution Summary:", json.dumps(summary, indent=2))

# Check agent domain mapping
for agent_id in executor.task_executors.keys():
    domain = executor.get_agent_domain(agent_id)
    print(f"Agent {agent_id}: {domain} domain")
```

---

## 🎉 **Congratulations!**

You now have the most advanced multi-agent execution system available! The V3 Hybrid approach gives you:

- **🚀 Performance**: Parallel execution capabilities
- **🧠 Intelligence**: Dynamic decision-making 
- **🔧 Flexibility**: Multiple execution strategies
- **📈 Scalability**: Hierarchical domain organization
- **🛡️ Reliability**: Advanced error handling and recovery
- **🔄 Compatibility**: 100% backward compatible with existing code

**Start simple, scale as needed, and enjoy the power of intelligent multi-agent orchestration!**