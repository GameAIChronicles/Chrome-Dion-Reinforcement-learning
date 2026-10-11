import mss
import time
import numpy as np
from gymnasium.spaces import Box, Discrete
import pydirectinput
import cv2 as cv
from gymnasium import Env
import pytesseract
import screen_capture
import cv2

time.sleep(1)


class ENVChromeDino(Env):
    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)  # Observation
        # 0 = jump, 1 = duck, 2 = nothing
        self.action_space = Discrete(3)  # Action

        self.selected_title, self.selected_hwnd = screen_capture.select_window()

        if self.selected_hwnd is None:
            print("No window selected.")
            raise ValueError("No window selected.")

        self.cap = mss.MSS()
        self.game_location = {'top': 300, 'left': 20, 'width': 1350, 'height': 260}  # Position of the Character
        self.done_loc = {'top': 300, 'left': 440, 'width': 240, 'height': 60}  # Position of Game Over

    # Taking an action in the Env
    def step(self, action):
        Action_map = {0 : 'space', 1 : 'down', 2 : 'nothing'}
        if action != 2:
            pydirectinput.press(Action_map[action])

        reward = 1
        done, done_map = self.GAME_OVER()
        new_obs = self.get_obs()


        return new_obs, reward, done, False, {}

    # Render the Env
    def render(self):
        pass

    # Resetting the Env
    def reset(self, **kwargs):
        time.sleep(0.2)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_obs(), {}

    # The image of the Agent in the Game
    def get_obs(self):
        raw = self.capture()
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
    def GAME_OVER(self):
        Done_cap = np.array(self.cap.grab(self.done_loc))[:, :, :3]
        x = pytesseract.image_to_string(Done_cap, lang='eng')
        DONE = False
        if 'G' in x:
            DONE = True
        return DONE, Done_cap

    def capture(self):
        #Send the selected window to the capture function -> numpy array back
        capturer = screen_capture.WindowCapturer()
        snapshot = capturer.capture_frame(self.selected_hwnd)
        return snapshot

    # Closing the window
    def close(self):
        pass


if '__main__' == __name__:
    env = ENVChromeDino()
    env.get_obs()