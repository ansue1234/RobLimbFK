import tkinter as tk
import csv

# Create the main window
root = tk.Tk()
root.title("Drawing App")

# Initialize variables to store drawing coordinates
coordinates = []

# Define the function to start drawing
def start_drawing(event):
    global coordinates
    coordinates = [(event.x, event.y)]  # Start recording at the initial point

# Define the function to draw and record coordinates
def draw(event):
    global coordinates
    x, y = event.x, event.y
    coordinates.append((x, y))
    canvas.create_line(coordinates[-2], (x, y), fill="black", width=2)

# Define the function to save coordinates to CSV
def save_to_csv(event):
    global coordinates
    with open("trajectory.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["goal_x", "goal_y"])  # Header row
        writer.writerows(coordinates)
    print("Coordinates saved to trajectory.csv")

# Define the function to exit the application on pressing Escape
def exit_app(event):
    print("Exiting application...")
    root.destroy()

# Create a canvas widget
canvas = tk.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

# Bind mouse events to canvas
canvas.bind("<ButtonPress-1>", start_drawing)  # Left mouse button press
canvas.bind("<B1-Motion>", draw)               # Mouse movement with button pressed
canvas.bind("<ButtonRelease-1>", save_to_csv)   # Left mouse button release

# Bind Escape key to exit the application
root.bind("<Escape>", exit_app)

# Run the application
root.mainloop()
