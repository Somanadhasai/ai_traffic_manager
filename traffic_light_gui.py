# traffic_light_gui.py

import tkinter as tk

class TrafficLightApp:
    def __init__(self, root):
        self.root = root
        self.light_label = tk.Label(root, text="Traffic Light", font=("Arial", 20))
        self.light_label.pack()

    def set_light(self, state):
        """
        Updates the traffic light color based on the traffic density state.
        state can be 'Low', 'Medium', or 'High'.
        """
        color_map = {"Low": "green", "Medium": "yellow", "High": "red"}
        color = color_map.get(state, "green")
        self.light_label.config(text=f"Traffic Light: {state}", fg=color)
    
    def update_gui(self):
        # This function updates the GUI and keeps it responsive.
        self.root.update_idletasks()
        self.root.update()

