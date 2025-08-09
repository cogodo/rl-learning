import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class Logger:
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """Initialize logger with directory and experiment name."""
        self.dir = log_dir
        self.exp_name = experiment_name
        self.logs = {
            "episodes": [],
            "steps": [],
            "metrics": []
        }
    
    def log_episode(self, episode: int, reward: float, length: int, loss: Optional[float] = None):
        """Log episode-level metrics."""
        entry = self._create_log_entry({
            "episode": episode,
            "reward": reward,
            "length": length,
            "loss": loss
        })
        self.logs["episodes"].append(entry)
        
    
    def log_step(self, step: int, obs: Any, action: Any, reward: float, done: bool):
        """Log step-level data."""
        entry = self._create_log_entry({
            "step": step,
            "observation": obs,
            "action": action,
            "reward": reward,
            "done": done
        })
        self.logs["steps"].append(entry)
    
    def save_logs(self, path: Optional[str] = None):
        """Save all logged data to file."""
        if path is None:
            path = os.path.join(self.dir, f"{self.exp_name or 'experiment'}_logs.json")
    
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def _create_log_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a log entry with timestamp."""
        entry = data.copy()
        entry["timestamp"] = datetime.now().isoformat()
        return entry