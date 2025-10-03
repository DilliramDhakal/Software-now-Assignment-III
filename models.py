from transformers import pipeline

class BaseModel:
    """Base class to enforce interface for models."""
    def __init__(self, model_name, task):
        self.model_name = model_name
        self.task = task
        self.pipe = None

    def load_model(self):
        raise NotImplementedError("Subclasses must override load_model")

    def run(self, input_data):
        raise NotImplementedError("Subclasses must override run")

    
