import tkinter as tk
from tkinter import filedialog, messagebox
from oop_explanations import explanations
from models import SentimentAnalysisModel, ImageClassificationModel

# Decorator for logging
def log_execution(func):
    def wrapper(*args, **kwargs):
        print(f"Running: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HIT137 AI Integration GUI")
        self.geometry("800x650")

        # Prepare model wrappers (lazy-load on demand)
        self.sentiment = SentimentAnalysisModel(
            "distilbert-base-uncased-finetuned-sst-2-english",
            "sentiment-analysis"
        )
        self.imgclf = ImageClassificationModel(
            "google/vit-base-patch16-224",
            "image-classification"
        )
