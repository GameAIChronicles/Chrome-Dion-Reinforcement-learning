import ctypes
from ctypes import wintypes
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import win32gui
import win32ui

# User32 and DWM C API bindings
user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi


class WindowCapturer:
    def __init__(self):
        self.last_error = ""

    def get_window_list(self):
        """Returns a list of active, visible top-level windows."""
        windows = []

        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.strip() and title not in ["Program Manager", "Settings"]:
                    windows.append((title, hwnd))

        win32gui.EnumWindows(enum_windows_callback, None)
        return windows

    def capture_frame(self, hwnd):
        """Captures a single snapshot of the window and returns it as a BGR numpy array."""
        if not win32gui.IsWindow(hwnd):
            self.last_error = "Invalid Window Handle."
            return None

        try:
            # Use DWM to get exact window bounds including extended shadow frames
            rect = wintypes.RECT()
            # DWMWA_EXTENDED_FRAME_BOUNDS = 9
            if dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
                user32.GetWindowRect(hwnd, ctypes.byref(rect))

            width = rect.right - rect.left
            height = rect.bottom - rect.top

            if width <= 0 or height <= 0:
                self.last_error = f"Invalid window size: {width}x{height}"
                return None

            # Create Compatible DC and Bitmap
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)

            # PW_RENDERFULLCONTENT = 2 (Forces hardware-accelerated windows to render)
            result = user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

            # Fall back to standard PrintWindow (0) if PW_RENDERFULLCONTENT fails
            if result != 1:
                result = user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype="uint8")
            img.shape = (height, width, 4)

            # Clean up Win32 objects immediately
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            if result == 1:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            self.last_error = "PrintWindow failed to capture window context."
            return None

        except Exception as e:
            self.last_error = str(e)
            return None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Window Latch Selector")
        self.root.geometry("480x200")
        self.root.resizable(False, False)

        self.capturer = WindowCapturer()
        self.window_map = {}

        # Result of the selection (filled in when the user clicks "Select")
        self.selected_title = None
        self.selected_hwnd = None

        self.create_widgets()
        self.refresh_windows()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        lbl = ttk.Label(frame, text="Select Target Window:", font=("Arial", 10, "bold"))
        lbl.pack(anchor=tk.W, pady=(0, 5))

        self.combo = ttk.Combobox(frame, state="readonly")
        self.combo.pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        self.btn_refresh = ttk.Button(btn_frame, text="Refresh List", command=self.refresh_windows)
        self.btn_refresh.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_select = ttk.Button(btn_frame, text="Select", command=self.confirm_selection)
        self.btn_select.pack(side=tk.LEFT, padx=(5, 0))

        self.lbl_status = ttk.Label(frame, text="Status: Ready", font=("Arial", 8, "italic"), foreground="gray")
        self.lbl_status.pack(anchor=tk.W, pady=(15, 0))

    def refresh_windows(self):
        windows = self.capturer.get_window_list()
        self.window_map = {title: hwnd for title, hwnd in windows}
        self.combo["values"] = list(self.window_map.keys())
        if self.window_map:
            self.combo.current(0)

    def confirm_selection(self):
        selected_title = self.combo.get()
        if not selected_title or selected_title not in self.window_map:
            messagebox.showerror("Error", "Please select a valid window.")
            return

        # Store the choice, then close the UI so control returns to __main__
        self.selected_title = selected_title
        self.selected_hwnd = self.window_map[selected_title]
        self.root.destroy()


def select_window():
    """Opens the UI and blocks until the user picks a window.
    Returns (title, hwnd), or (None, None) if the UI was closed without selecting."""
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    return app.selected_title, app.selected_hwnd


if __name__ == "__main__":
    # 1) UI opens, user picks a window -> stored in these two variables
    selected_title, selected_hwnd = select_window()

    if selected_hwnd is None:
        print("No window selected.")
    else:
        print(f"Selected: {selected_title} (hwnd={selected_hwnd})")

        # 2) Send the selected window to the capture function -> numpy array back
        capturer = WindowCapturer()
        snapshot = capturer.capture_frame(selected_hwnd)

        if snapshot is None:
            print("Capture failed:", capturer.last_error)
        else:
            print("Snapshot shape:", snapshot.shape)  # (height, width, 3) BGR

            # Any time later, call this again to get a fresh snapshot of the same window:
            # snapshot = capturer.capture_frame(selected_hwnd)

            cv2.imshow("Snapshot", snapshot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
