#%%
import tkinter as tk

#%%
root = tk.Tk()
root.title("NLP-TCM")
root.geometry("600x400")
root.state("zoomed")
root.configure(bg='#9aa7b1')
#root.iconbitmap('mystar.ico')
label = tk.Label(root, text="NLP-TCM", font=("Arial", 20), bg='#f5f3f2',anchor='nw')
label.pack(side='left')

root.mainloop()
