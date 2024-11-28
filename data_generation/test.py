import pygame
import tkinter as tk

'''
Detecting buttons, axes and D-pads of controllers or steering wheels & pedals
'''

# Initialize pygame
pygame.init()
pygame.joystick.init()

# Initialize the first joystick
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected. Please check the connection!")
    exit()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Tkinter setup
root = tk.Tk()
root.title("Controller / Wheel Input")
root.geometry("400x600")

# Labels for displaying joystick states
button_labels = []
axis_labels = []
hat_labels = []

num_buttons = joystick.get_numbuttons()
num_axes = joystick.get_numaxes()
num_hats = joystick.get_numhats()

# Create and grid labels for buttons
tk.Label(root, text="Buttons").grid(row=0, column=0, columnspan=2)
for i in range(num_buttons):
    label = tk.Label(root, text=f"Button {i}: 0")
    label.grid(row=i+1, column=0, sticky="w")
    button_labels.append(label)

# Create and grid labels for axes
tk.Label(root, text="Axes").grid(row=num_buttons + 1, column=0, columnspan=2)
for i in range(num_axes):
    label = tk.Label(root, text=f"Axis {i}: 0.00")
    label.grid(row=num_buttons + 2 + i, column=0, sticky="w")
    axis_labels.append(label)

# Create and grid labels for hats (D-Pad)
tk.Label(root, text="D-Pad (Hats)").grid(row=num_buttons + num_axes + 2, column=0, columnspan=2)
for i in range(num_hats):
    label = tk.Label(root, text=f"Hat {i}: (0, 0)")
    label.grid(row=num_buttons + num_axes + 3 + i, column=0, sticky="w")
    hat_labels.append(label)

def update_input():
    pygame.event.pump()  # Update pygame events

    # Update button states
    for i in range(num_buttons):
        button = joystick.get_button(i)
        button_labels[i].config(text=f"Button {i}: {button}")

    # Update axis states
    for i in range(num_axes):
        axis = joystick.get_axis(i)
        axis_labels[i].config(text=f"Axis {i}: {axis:.2f}")

    # Update hat states (D-Pad)
    for i in range(num_hats):
        hat = joystick.get_hat(i)
        hat_labels[i].config(text=f"Hat {i}: {hat}")

    # Schedule the next update
    root.after(100, update_input)

# Start the update loop
root.after(100, update_input)
root.mainloop()

# Cleanup on exit
pygame.joystick.quit()
pygame.quit()
