import tkinter as tk

def main():
    """
    یک برنامه دسکتاپ ساده با استفاده از Tkinter.
    """
    window = tk.Tk()
    window.title("برنامه دسکتاپ آلفا")

    label = tk.Label(window, text="سلام از طرف پروژه آلفا!")
    label.pack(padx=20, pady=20)

    button = tk.Button(window, text="خروج", command=window.destroy)
    button.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    main()
