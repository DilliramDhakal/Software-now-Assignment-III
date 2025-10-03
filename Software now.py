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

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
        )
        if path:
            self.selected_image_path = path
            self.image_label.config(text=path)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", ".png *.jpg *.jpeg *.bmp *.gif"), ("All files", ".*")]
        )
        if path:
            self.selected_image_path = path
            self.image_label.config(text=path)

    @log_execution
    def run_sentiment(self):
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter text.")
            return
        try:
            result = self.sentiment.run(text)
            self.output_box.insert(tk.END, f"\nSentiment Analysis:\n{result}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Sentiment failed: {e}")
