// This is how the updated launchAgent function should look
function launchAgent(agentId) {
    selectedAgentId = agentId;
    loadPage('launch-agent');
    
    // Fetch agent data and populate the form
    showLoading();
    fetch(`/api/agents/${agentId}`)
        .then(response => response.json())
        .then(agent => {
            setTimeout(() => {

                hideLoading();

                // Update page title
                const header = document.querySelector('.page-header h1');
                if (header) {
                    header.textContent = ` ${agent.name} ⛳ Playground`;
                }
                
                // Fill form fields - Check if element exists before setting value
                const agentNameInput = document.getElementById('agentName');
                if (agentNameInput) {
                    agentNameInput.value = agent.name;
                }
                
                const agentDescriptionInput = document.getElementById('agentDescription');
                if (agentDescriptionInput) {
                    agentDescriptionInput.value = agent.description;
                }

                const llmProviderSelect = document.getElementById('llmProvider');
                if (llmProviderSelect) {
                    llmProviderSelect.value = agent.llmProvider;
                }

                const llmModelSelect = document.getElementById('llmModel');
                if (llmModelSelect) {
                    llmModelSelect.value = agent.llmModel;
                }

                const apiKeyInput = document.getElementById('apiKey');
                if (apiKeyInput) {
                    apiKeyInput.value = agent.apiKey;
                }

                const agentRoleInput = document.getElementById('agentRole');
                if (agentRoleInput) {
                    agentRoleInput.value = agent.role;
                }

                const agentGoalInput = document.getElementById('agentGoal');
                if (agentGoalInput) {
                    agentGoalInput.value = agent.goal || '';
                }

                const expectedOutputInput = document.getElementById('expectedOutput');
                if (expectedOutputInput) {
                    expectedOutputInput.value = agent.expectedOutput || '';
                }
                
                const agentBackstoryInput = document.getElementById('agentBackstory');
                if (agentBackstoryInput) {
                    agentBackstoryInput.value = agent.backstory || '';
                }
                
                const agentInstructionsInput = document.getElementById('agentInstructions');
                if (agentInstructionsInput) {
                    agentInstructionsInput.value = agent.instructions;
                }
                
                // Initialize LLM provider change handler
                handleLLMProviderChange();
                
                // Initialize selectedTools with agent's tools
                selectedTools = new Set(agent.tools || []);
                
                // Initialize selectedAdvancedTools with agent's advanced tools
                selectedAdvancedTools = new Set();
                if (Array.isArray(agent.advanced_tools)) {
                    agent.advanced_tools.forEach(toolId => selectedAdvancedTools.add(toolId));
                } else if (Array.isArray(agent.advancedTools)) {
                    agent.advancedTools.forEach(toolId => selectedAdvancedTools.add(toolId));
                }
                
                // Load agent tools (combine both normal and advanced tools)
                const allToolIds = [...selectedTools, ...selectedAdvancedTools];
                loadAgentTools(allToolIds);
            }, 100); // Keeping the timeout for now, but the checks add safety
        });
} 