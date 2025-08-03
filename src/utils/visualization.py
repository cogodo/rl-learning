class Plotter:
    def plot_learning_curves(self, rewards, losses):
        return NotImplementedError
    
    def plot_algorithm_comparison(self, results):
        return NotImplementedError
    
    def plot_hyperparam_sensitivity(self, results):
        return NotImplementedError
    
    def save_plots(slef, path):
        return NotImplementedError