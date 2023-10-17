import tkinter as tk
from threading import Thread
from subprocess import run
from tabulate import tabulate
import pandas as pd

class RecSysApp:

    def __init__(self, root):
        self.root = root
    
    def run_app(self):
        self.create_graphics()
        self.display_recommended()

    def create_graphics(self):
        self.label = tk.Label(self.root, text="Enter your text:")
        self.label.pack()

        self.entry = tk.Entry(self.root)
        self.entry.pack()
        self.entry.focus()
        
        self.display = tk.Label(self.root, text="")
        self.display.pack()

        self.submit_button = tk.Button(self.root, text="Submit", command=self.handle_entry)
        self.submit_button.pack()

        self.run_button = tk.Button(self.root, text="Run recommender", command=self.start_run_rec_sys)
        self.run_button.pack()
    
    def display_recommended(self):
        self.text = tk.Text(self.root, state="disabled")
        self.text.pack(fill="both", expand=True)

    def load_csv(self):
        file_path = "recommended.csv"
        try:
            df = pd.read_csv(file_path)
            df = df.rename(columns={"Title": "Exercise Name", "similarity": "Similarity Score"})
            column_order = ["Exercise Name", "Similarity Score"]
            df = df[column_order]
            data = df.to_dict(orient='records')
            table = tabulate(data, headers='keys', tablefmt='fancy_grid')
            self.text.config(state="normal") 
            self.text.delete(1.0, tk.END) 
            self.text.insert(tk.END, table)
            self.text.config(state="disabled") 
        except Exception as e:
            self.text.config(state="normal") 
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, f"Error: No exercise named {self.entry.get()} in dataset. Please enter a different exercise.")
            self.text.config(state="disabled") 
    
    def handle_entry(self):
        entered = self.entry.get()
        with open('entries.txt', 'w') as file:
            file.write(entered)
        self.display.config(text=f"You entered: {entered}")

    def run_rec_sys(self):
        self.label = tk.Label(self.root, text="Running recommendation...")
        self.label.pack()
        try:
           run(["python", "notebook_to_script.py"])
        except Exception as e:
            self.text.config(state="normal") 
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, f"Error: {str(e)}")
            self.text.config(state="disabled")
        self.load_csv() 
        self.label.destroy()
    
    def start_run_rec_sys(self):
        if self.entry.get() == '':
            self.text.config(state="normal") 
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, "%sPlease enter an exercise" % (" "*45))
            self.text.config(state="disabled")
            return
        Thread(target=self.run_rec_sys).start()

def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    root.geometry(f"{width}x{height}+{x}+{y}")
    

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Exercise Recommender System')
    center_window(root, 800, 650)
    root.resizable(width=False, height=False)
    app = RecSysApp(root)
    app.run_app()
    root.mainloop()