import tkinter as tk
import threading

class RecSysApp:

    def __init__(self, root):
        self.root = root
        self.create_graphics()

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

    def handle_entry(self):
        entered = self.entry.get()
        with open('entries.txt', 'w') as file:
            file.write(entered)
        self.display.config(text=f"You entered: {entered}")

    def run_rec_sys(self):
        import subprocess
        subprocess.run(["python", "notebook_to_script.py"])
    
    def start_run_rec_sys(self):
         threading.Thread(target=self.run_rec_sys).start()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = RecSysApp(root)
    root.mainloop()
