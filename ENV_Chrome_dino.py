import time
import numpy as np
from gymnasium.spaces import Box, Discrete
import cv2 as cv
from gymnasium import Env
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException

class ENVChromeDino(Env):
    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8)  # Observation
        # 0 = jump, 1 = duck, 2 = nothing
        self.action_space = Discrete(3)  # Action

        options = webdriver.ChromeOptions()
        options.add_argument('--window-size=1200,600')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-background-timer-throttling')
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_cdp_cmd("Emulation.setFocusEmulationEnabled", {"enabled": True})
        self.driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
            "width": 1200, "height": 400, "deviceScaleFactor": 1, "mobile": False
        })
        try:
            self.driver.get("chrome://dino")
        except WebDriverException:
            pass
        self.game_body = self.driver.find_element(By.TAG_NAME, "body")

        self.game_location = {'top': 90, 'left': 0, 'width': 800, 'height': 250}  # Position of the Character
        self.done_loc = {'top': 130, 'left': 380, 'width': 200, 'height': 50}  # Position of Game Over

    # Taking an action in the Env
    def step(self, action):
        # Action_map = {0 : 'space', 1 : 'down', 2 : 'nothing'}
        # Action_map = {0: Keys.SPACE, 1: Keys.ARROW_DOWN, 2: 'nothing'}
        if action == 0:
            self.game_body.send_keys(Keys.SPACE)
        elif action == 1:
            self.game_body.send_keys(Keys.ARROW_DOWN)

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
        time.sleep(1.5)  # game ignores restart for a while after a crash
        for _ in range(5):
            self.game_body.send_keys(Keys.SPACE)
            time.sleep(0.3)
            raw = self.capture()
            if not self.GAME_OVER(raw):  # restart worked
                break
            time.sleep(0.5)  # still Game Over, so wait and try again
        time.sleep(1.0)
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
        png = self.driver.get_screenshot_as_png()
        return cv.imdecode(np.frombuffer(png, np.uint8), cv.IMREAD_COLOR)

    # Closing the window
    def close(self):
        self.driver.quit()




if '__main__' == __name__:
    env = ENVChromeDino()
    env.reset()
    steps = 0
    for i in range(300):
        _, _, done, _, _ = env.step(2)  # do nothing
        steps += 1
        if done:
            if steps == 1:
                time.sleep(1.0)
                still_over = env.GAME_OVER(env.capture())
                print("episode length: 1 | still Game Over after 1 s:", still_over)
            else:
                print("episode length:", steps)
            steps = 0
            env.reset()