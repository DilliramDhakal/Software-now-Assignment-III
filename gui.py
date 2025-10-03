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
        self.selected_image_path = None
        self.create_widgets()
        
    def create_widgets(self):
        # Input for text
        tk.Label(self, text="Enter text input:").pack(anchor="w", padx=10, pady=(10,0))
        self.text_entry = tk.Entry(self, width=70)
        self.text_entry.pack(padx=10, pady=5)

        # Image chooser
        img_frame = tk.Frame(self)
        img_frame.pack(fill="x", padx=10, pady=(10,0))
        tk.Button(img_frame, text="Choose Image...", command=self.choose_image).pack(side="left")
        self.image_label = tk.Label(img_frame, text="No image selected")
        self.image_label.pack(side="left", padx=10)

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)
        tk.Button(btn_frame, text="Run Sentiment Analysis", command=self.run_sentiment).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Classify Image", command=self.run_image_classification).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Show OOP Explanations", command=self.show_explanations).pack(side="left", padx=5)

        # Output Section
        tk.Label(self, text="Output:").pack(anchor="w", padx=10)
        self.output_box = tk.Text(self, height=22, width=100)
        self.output_box.pack(padx=10, pady=(0,10))

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
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
            
    @log_execution
    def run_image_classification(self):
        if not self.selected_image_path:
            messagebox.showwarning("Input Error", "Please select an image file first.")
            return
        try:
            result = self.imgclf.run(self.selected_image_path)
            self.output_box.insert(tk.END, f"\nImage Classification ({self.selected_image_path}):\n{result}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {e}")

    def show_explanations(self):
        self.output_box.insert(tk.END, "\n--- OOP Explanations ---\n")
        for key, value in explanations.items():
            self.output_box.insert(tk.END, f"{key.capitalize()}: {value}\n")

