import tkinter as tk


class Piano:
    def __init__(self):
        self.keysNames = [
            "0a", "0a#", "0h",
            "1c", "1c#", "1d", "1d#", "1e", "1f", "1f#", "1g", "1g#", "1a", "1a#", "1h",
            "2c", "2c#", "2d", "2d#", "2e", "2f", "2f#", "2g", "2g#", "2a", "2a#", "2h",
            "3c", "3c#", "3d", "3d#", "3e", "3f", "3f#", "3g", "3g#", "3a", "3a#", "3h",
            "4c", "4c#", "4d", "4d#", "4e", "4f", "4f#", "4g", "4g#", "4a", "4a#", "4h",
            "5c", "5c#", "5d", "5d#", "5e", "5f", "5f#", "5g", "5g#", "5a", "5a#", "5h",
            "6c", "6c#", "6d", "6d#", "6e", "6f", "6f#", "6g", "6g#", "6a", "6a#", "6h",
            "7c", "7c#", "7d", "7d#", "7e", "7f", "7f#", "7g", "7g#", "7a", "7a#", "7h", "8c"
        ]
        self.keyLabels = []
        m = tk.Tk()
        m.title("Piano")
        m.geometry("1070x300")

        for i in range(0, len(self.keysNames)):
            y = 50
            x = 20*(i - self.countBlacksBefore(i))
            height = 100
            bg = "white"
            fg = "black"
            if self.isBlack(i):
                x -= 10
                bg = "black"
                fg = "white"
                height = 70

            l = tk.Label(m, bg=bg, fg=fg, font=4, highlightbackground="black",
                         highlightcolor="black", highlightthickness=1)
            self.keyLabels.append(l)
            l.place(x=x, y=y, width=20, height=height)

    def isBlack(i):
        if (i + 9) % 12 in (1, 3, 6, 8, 10):
            return True
        return False

    def countBlacksBefore(i):
        octave = (i - 3) // 12
        remainder = (i + 9) % 12
        result = octave * 5
        if remainder > 1:
            result += 1
        if remainder > 3:
            result += 1
        if remainder > 6:
            result += 1
        if remainder > 8:
            result += 1
        if remainder > 10:
            result += 1
        return result

    def green(self, i):
        self.keyLabels[i].config(bg="green")
