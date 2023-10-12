import tkinter as tk
import threading
import subprocess
import pandas as pd

class RecSysApp:

    def __init__(self, root):
        self.root = root
        self.create_graphics()
        self.display_recommended()

    def create_graphics(self):
        self.label = tk.Label(self.root, text="Enter your text:")
        self.label.pack()

        self.entry = tk.Entry(self.root)
        self.entry.pack()

        self.display = tk.Label(self.root, text="")
        self.display.pack()

        self.update_display = tk.Button(self.root, text="Submit", command=self.handle_entry)
        self.update_display.pack()

        self.run_button = tk.Button(self.root, text="Run recommender", command=self.start_run_rec_sys)
        self.run_button.pack()
    
    def display_recommended(self):
        self.text = tk.Text(self.root, state="disabled")
        self.text.pack(fill="both", expand=True)

    def load_csv(self):
        file_path = "recommended.csv"
        try:
            df = pd.read_csv(file_path)
            column_order = ["Title", "similarity"]
            df = df[column_order]
            csv_text = df.to_string(index=False, header=True, justify='center', formatters={
                "Title": "{:<50}".format,
                "similarity": "{:0}".format
            })
            self.text.config(state="normal") 
            self.text.delete(1.0, tk.END) 
            self.text.insert(tk.END, csv_text)
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
            subprocess.run(["python", "notebook_to_script.py"])
        except Exception as e:
            self.text.config(state="normal") 
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, f"Error: {str(e)}")
            self.text.config(state="disabled")
        self.load_csv() 
        self.label.destroy()
    
    def start_run_rec_sys(self):
        if self.entry.get() == '':
            return
        threading.Thread(target=self.run_rec_sys).start()

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Exercise Recommender System')
    root.geometry("800x600")
    root.resizable(width=False, height=False)
    app = RecSysApp(root)
    root.mainloop()
