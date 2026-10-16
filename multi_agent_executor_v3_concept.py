"""
Multi-Agent Executor V3 - Hybrid Hierarchical + Dynamic Approach
Combines the best of hierarchical management with dynamic decision making
"""

import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    CONDITIONAL = "conditional"
    DYNAMIC = "dynamic"

@dataclass
class ExecutionPhase:
    name: str
    strategy: ExecutionStrategy
    agents: List[str]
    conditions: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None

class HybridMultiAgentExecutorV3:
    """
    Advanced multi-agent executor with hierarchical management and dynamic decisions
    """
    
    def __init__(self, config):
        # Hierarchical structure
        self.master_manager = MasterManagerAgent()
        self.domain_managers = {
            "data": DataDomainManager(),
            "analysis": AnalysisDomainManager(), 
            "synthesis": SynthesisDomainManager(),
            "output": OutputDomainManager()
        }
        
        # Worker agents organized by domain
        self.worker_agents = self._initialize_workers(config)
        
        # Execution state management
        self.execution_context = ExecutionContext()
    
    async def execute_task(self, user_input: str, **kwargs) -> str:
        """
        Main execution method with hybrid approach
        """
        try:
            # Phase 1: Master planning with hierarchical breakdown
            execution_plan = await self.master_manager.create_execution_plan(
                user_input=user_input,
                available_domains=list(self.domain_managers.keys()),
                context=kwargs
            )
            
            # Phase 2: Dynamic execution with real-time decisions
            results = await self._execute_plan_dynamically(execution_plan, user_input)
            
            # Phase 3: Final synthesis
            return await self.master_manager.synthesize_final_result(results)
            
        except Exception as e:
            return await self._handle_execution_error(e)
    
    async def _execute_plan_dynamically(self, plan: List[ExecutionPhase], user_input: str) -> Dict[str, Any]:
        """
        Execute plan with dynamic decision making at each phase
        """
        results = {"original_input": user_input}
        
        for phase in plan:
            self.execution_context.current_phase = phase.name
            
            # Dynamic decision: Should we execute this phase?
            should_execute = await self._should_execute_phase(phase, results)
            if not should_execute:
                continue
            
            # Execute based on strategy
            if phase.strategy == ExecutionStrategy.SEQUENTIAL:
                results[phase.name] = await self._execute_sequential(phase, results)
            elif phase.strategy == ExecutionStrategy.PARALLEL:
                results[phase.name] = await self._execute_parallel(phase, results)
            elif phase.strategy == ExecutionStrategy.CONDITIONAL:
                results[phase.name] = await self._execute_conditional(phase, results)
            elif phase.strategy == ExecutionStrategy.DYNAMIC:
                results[phase.name] = await self._execute_dynamic(phase, results)
            
            # Dynamic decision: Should we continue or modify the plan?
            plan_modification = await self._evaluate_plan_modification(phase, results)
            if plan_modification:
                plan = await self._modify_plan(plan, plan_modification, results)
        
        return results
    
    async def _execute_sequential(self, phase: ExecutionPhase, context: Dict[str, Any]) -> Any:
        """Sequential execution within a phase"""
        domain_manager = self.domain_managers[self._get_phase_domain(phase)]
        
        current_input = self._prepare_phase_input(phase, context)
        
        for agent_id in phase.agents:
            agent = self.worker_agents[agent_id]
            
            # Dynamic decision: Should we execute this specific agent?
            if await domain_manager.should_execute_agent(agent_id, current_input, context):
                result = await agent.execute(
                    input=current_input,
                    context=context,
                    phase_info=phase
                )
                current_input = result
            else:
                # Skip agent but log the decision
                self.execution_context.log_skipped_agent(agent_id, "Dynamic decision")
        
        return current_input
    
    async def _execute_parallel(self, phase: ExecutionPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel execution within a phase"""
        tasks = []
        phase_input = self._prepare_phase_input(phase, context)
        
        for agent_id in phase.agents:
            agent = self.worker_agents[agent_id]
            task = agent.execute(
                input=phase_input,
                context=context,
                phase_info=phase
            )
            tasks.append((agent_id, task))
        
        # Execute all agents in parallel
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (agent_id, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[agent_id] = {"error": str(result)}
            else:
                results[agent_id] = result
        
        return results
    
    async def _execute_conditional(self, phase: ExecutionPhase, context: Dict[str, Any]) -> Any:
        """Conditional execution based on runtime conditions"""
        domain_manager = self.domain_managers[self._get_phase_domain(phase)]
        
        # Evaluate conditions dynamically
        conditions_met = await domain_manager.evaluate_conditions(
            phase.conditions, 
            context
        )
        
        # Execute only agents whose conditions are met
        results = {}
        for agent_id in phase.agents:
            if conditions_met.get(agent_id, False):
                agent = self.worker_agents[agent_id]
                phase_input = self._prepare_phase_input(phase, context)
                results[agent_id] = await agent.execute(
                    input=phase_input,
                    context=context,
                    phase_info=phase
                )
        
        return results
    
    async def _execute_dynamic(self, phase: ExecutionPhase, context: Dict[str, Any]) -> Any:
        """Fully dynamic execution - agents chosen at runtime"""
        domain_manager = self.domain_managers[self._get_phase_domain(phase)]
        
        current_input = self._prepare_phase_input(phase, context)
        results = []
        
        # Continue until domain manager decides we're done
        while True:
            # Dynamic decision: Which agent should execute next?
            next_agent_id = await domain_manager.choose_next_agent(
                available_agents=phase.agents,
                current_input=current_input,
                context=context,
                execution_history=results
            )
            
            if not next_agent_id:
                break  # Domain manager decided we're done
            
            agent = self.worker_agents[next_agent_id]
            result = await agent.execute(
                input=current_input,
                context=context,
                phase_info=phase
            )
            
            results.append({
                "agent_id": next_agent_id,
                "result": result
            })
            
            current_input = result  # Output becomes input for next iteration
            
            # Safety check to prevent infinite loops
            if len(results) > 10:  # Configurable limit
                break
        
        return results
    
    async def _should_execute_phase(self, phase: ExecutionPhase, context: Dict[str, Any]) -> bool:
        """Dynamic decision: Should this phase be executed?"""
        domain = self._get_phase_domain(phase)
        domain_manager = self.domain_managers[domain]
        
        return await domain_manager.should_execute_phase(phase, context)
    
    async def _evaluate_plan_modification(self, completed_phase: ExecutionPhase, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dynamic decision: Should we modify the remaining execution plan?"""
        return await self.master_manager.evaluate_plan_modification(
            completed_phase, 
            context, 
            self.execution_context
        )
    
    def _get_phase_domain(self, phase: ExecutionPhase) -> str:
        """Determine which domain manager should handle this phase"""
        # Logic to map phase to domain - could be configurable
        domain_mapping = {
            "data_collection": "data",
            "data_processing": "data", 
            "analysis": "analysis",
            "synthesis": "synthesis",
            "reporting": "output"
        }
        return domain_mapping.get(phase.name, "analysis")  # Default domain
    
    def _prepare_phase_input(self, phase: ExecutionPhase, context: Dict[str, Any]) -> Any:
        """Prepare input for a phase based on dependencies"""
        if not phase.dependencies:
            return context.get("original_input")
        
        # Combine inputs from dependent phases
        combined_input = {}
        for dep in phase.dependencies:
            if dep in context:
                combined_input[dep] = context[dep]
        
        return combined_input if combined_input else context.get("original_input")

# Domain-specific managers for hierarchical organization
class DataDomainManager:
    """Manages data-related agents and decisions"""
    
    async def should_execute_phase(self, phase: ExecutionPhase, context: Dict[str, Any]) -> bool:
        # Domain-specific logic for data phases
        return True
    
    async def should_execute_agent(self, agent_id: str, input_data: Any, context: Dict[str, Any]) -> bool:
        # Agent-specific execution decisions
        return True
    
    async def choose_next_agent(self, available_agents: List[str], current_input: Any, 
                              context: Dict[str, Any], execution_history: List[Dict]) -> Optional[str]:
        # Dynamic agent selection logic
        return available_agents[0] if available_agents else None

class AnalysisDomainManager:
    """Manages analysis-related agents and decisions"""
    # Similar structure to DataDomainManager but with analysis-specific logic
    pass

class ExecutionContext:
    """Maintains execution state and history"""
    
    def __init__(self):
        self.current_phase = None
        self.execution_history = []
        self.skipped_agents = []
        self.errors = []
    
    def log_skipped_agent(self, agent_id: str, reason: str):
        self.skipped_agents.append({"agent_id": agent_id, "reason": reason})