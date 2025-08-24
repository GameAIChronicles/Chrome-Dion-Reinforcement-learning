# Chrome-Dion-Reinforcement-learning

---

🦖 Chrome Dino AI (Reinforcement Learning)

📌 Overview

This project implements an AI agent that learns to play the Google Chrome Dino Game using Reinforcement Learning.
The AI is trained with a custom environment and models, then tested directly in the Chrome browser.

👉 Demo video: AI Beat Me in Chrome Dino Python (Reinforcement Learning) https://youtu.be/za6_gqYyku4?si=6SQKCXa3YUQiYJfQ

[![Watch the video](https://img.youtube.com/vi/za6_gqYyku4/maxresdefault.jpg)](https://youtu.be/za6_gqYyku4)

### [Watch this video on YouTube](https://youtu.be/za6_gqYyku4)

---

📂 Project Structure

ChromeDino/

│

├── DinoModeling.py          # Main training script (train the agent)

├── DinoModelingProceed.py   # Continue training from a saved model

├── DinoPredictionMain.py    # Load model & test it in Chrome

├── EnvChromeDino.py         # Custom Gymnasium environment for Dino game

├── EnvChecker.py            # Check if the environment works with Gymnasium/Stable Baselines

├── GameOverPrediction.py    # Detects 'Game Over' text using screenshots

│

├── data/

│   ├── logs/                # Training logs (TensorBoard etc.)

│   └── models/              # Trained models

│         └── dino_models/     # Multiple trained model versions

README.md


---

📜 File Descriptions

DinoModeling.py → The training script where the agent is trained from scratch.

DinoModelingProceed.py → Continue training an already saved model.

DinoPredictionMain.py → This script is used to test the trained AI model in the actual Chrome Dino game. 
    It works by first moving the mouse cursor to the predefined Chrome icon position on the taskbar, 
    clicking it to make sure the game window is active. Once the game starts, 
    the script continuously takes screenshots of the Chrome window and uses PyTesseract (OCR) to process them. 
    Based on the game environment detected from these screenshots, the trained model decides the next action (e.g., jump or stay). 
    This way, the AI can play the Dino game automatically, without direct game API integration, purely through screen capture and model prediction.

It simulates mouse movements to directly open Chrome at a predefined position.


EnvChromeDino.py → Custom Gymnasium environment for Chrome Dino.

EnvChecker.py → Checks if the environment is working correctly with Gymnasium + Stable Baselines.

GameOverPrediction.py → Captures a full-screen Chrome screenshot and checks if "Game Over" text is visible. Returns True or False accordingly.



---

🖼 Screenshots

![image alt](https://github.com/GameAIChronicles/Chrome-Dion-Reinforcement-learning/blob/cff9d264c64e291e98e0f0e99400d9b8705c817c/Screenshot%202025-08-24%20070353.png)




---

▶ Usage

Train a new model

python DinoModeling.py

Continue training from saved model

python DinoModelingProceed.py

Test the trained model in Chrome

python DinoPredictionMain.py

Make sure Chrome is open in the expected position on screen (the script will move the mouse to it automatically).


---

📊 Logs & Models

Logs → Stored inside data/logs/ for TensorBoard visualization.

Models → Saved models inside data/models/dino_models/.



---

🙏 Acknowledgement

This project was built as a learning project for Reinforcement Learning, Gymnasium, and AI automation.
Thanks to open-source resources and tutorials that inspired me to build this!


---
