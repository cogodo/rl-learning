
## **Complete RL Framework Implementation Spec**

### **Phase 4: Utilities**

#### **8. Visualization**
```python
# src/utils/visualization.py
class Plotter:
    def plot_learning_curves(self, rewards, losses)
    def plot_algorithm_comparison(self, results)
    def plot_hyperparameter_sensitivity(self, results)
    def save_plots(self, path)
```

### **Phase 5: Testing Infrastructure**

#### **10. Unit Tests**
```python
# tests/test_env_wrappers.py
# tests/test_agents.py
# tests/test_buffers.py
# tests/test_networks.py
```

### **Implementation Order:**

1. **Start with ConfigManager** - Everything else depends on config
2. **EnvironmentFactory + BaseWrapper** - Core environment handling
3. **Logger + MetricsTracker** - Essential for debugging
4. **ReplayBuffer** - Needed for most algorithms
5. **BaseAgent** - Foundation for all agents
6. **Trainer** - Training loop logic
7. **ExperimentRunner** - Orchestrates everything

### **Key Learning Points:**

- **Environment Wrappers**: Learn how to modify environments without changing core logic
- **Experience Buffers**: Understand how to store and sample experience
- **Network Architectures**: Learn how to design networks for different action/observation spaces
- **Training Loops**: Understand the core training process
- **Configuration Management**: Learn how to make experiments reproducible

### **Before Implementing Algorithms:**

1. **Test with Random Agent**: Make sure your framework works with random actions
2. **Test with Simple Policy**: Implement a basic policy that always takes action 0
3. **Validate Metrics**: Ensure your logging and evaluation work correctly
4. **Test Wrappers**: Make sure normalization and other wrappers work as expected

This foundation will make implementing algorithms much easier and help you understand the "why" behind each component!