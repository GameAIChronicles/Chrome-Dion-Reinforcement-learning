import time
import numpy as np
from gymnasium.spaces import Box, Discrete
import pydirectinput
import cv2 as cv
from gymnasium import Env
import screen_capture


class ENVChromeDino(Env):
    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8)  # Observation
        # 0 = jump, 1 = duck, 2 = nothing
        self.action_space = Discrete(3)  # Action

        self.selected_title, self.selected_hwnd = screen_capture.select_window()

        if self.selected_hwnd is None:
            print("No window selected.")
            raise ValueError("No window selected.")

        self.game_location = {'top': 300, 'left': 20, 'width': 1350, 'height': 260}  # Position of the Character
        self.done_loc = {'top': 330, 'left': 470, 'width': 200, 'height': 60}  # Position of Game Over

    # Taking an action in the Env
    def step(self, action):
        Action_map = {0 : 'space', 1 : 'down', 2 : 'nothing'}
        if action != 2:
            pydirectinput.press(Action_map[action])

        reward = 1
        raw = self.capture()
        done = self.GAME_OVER(raw)
        new_obs = self.get_obs(raw)


        return new_obs, reward, done, False, {}

    # Render the Env
    def render(self):
        pass

    # Resetting the Env
    def reset(self, **kwargs):
        time.sleep(0.2)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        raw = self.capture()
        return self.get_obs(raw), {}

    # The image of the Agent in the Game
    def get_obs(self, raw):
        ymin = self.game_location["top"]
        xmin = self.game_location["left"]
        xmax = xmin + self.game_location["width"]  # X start + Width
        ymax = ymin + self.game_location["height"]  # Y start + Height
        # raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        raw= raw[ymin:ymax, xmin:xmax]
        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (100, 100))
        channel = np.reshape(resized, (1, 100, 100))
        return channel

    # Game over Indication
    def GAME_OVER(self, raw):
        ymin = self.done_loc["top"]
        xmin = self.done_loc["left"]
        xmax = xmin + self.done_loc["width"]
        ymax = ymin + self.done_loc["height"]
        done_region = raw[ymin:ymax, xmin:xmax]
        gray = cv.cvtColor(done_region, cv.COLOR_BGR2GRAY)
        if np.mean(gray) < 100:
            gray = cv.bitwise_not(gray)

        _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_text_shapes = 0

        for contour in contours:
            if cv.contourArea(contour) < 15:
                continue

            x, y, w, h = cv.boundingRect(contour)
            if 10 < h < 45 and 5 < w < 40:
                valid_text_shapes += 1
        if valid_text_shapes >= 4:
            # print(f"[SYSTEM] GAME OVER Confirmed! Detected {valid_text_shapes} aligned letter shapes.")
            return True

        return False

    def capture(self):
        #Send the selected window to the capture function -> numpy array back
        capturer = screen_capture.WindowCapturer()
        snapshot = capturer.capture_frame(self.selected_hwnd)
        if snapshot is None:
            print("Capture failed:", capturer.last_error)
            print("The Window might be minimized.")
        return snapshot

    # Closing the window
    def close(self):
        pass


if '__main__' == __name__:
    env = ENVChromeDino()
    raw = env.capture()
    env.GAME_OVER(raw)