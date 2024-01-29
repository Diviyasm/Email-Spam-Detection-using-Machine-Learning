import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import joblib

# Load the saved model
loaded_model = joblib.load('naive_bayes_model.sav')

def classify_email():
    email_text = email_entry.get()
    if not email_text:
        messagebox.showerror("Error", "Please enter an email text.")
        return

    # Make a prediction using the loaded model
    prediction = loaded_model.predict([email_text])
    if prediction[0] == 0:
        result_label.config(text="Result: HAM")
    else:
        result_label.config(text="Result: SPAM")

# Create the main application window
app = tk.Tk()
app.title("Email Ham/Spam Detection")

# Label and Entry for entering email text
email_label = ttk.Label(app, text="Enter Email Text:")
email_label.pack()
email_entry = ttk.Entry(app, width=50)
email_entry.pack()

# Button for classification
classify_button = ttk.Button(app, text="Classify", command=classify_email)
classify_button.pack()

# Label to display the result
result_label = ttk.Label(app, text="Result: ")
result_label.pack()

# Start the Tkinter main loop
app.mainloop()
