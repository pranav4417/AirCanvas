
---

# 🖐️ Gesture Drawing App

A real-time hand gesture–based drawing application built using **OpenCV**, **MediaPipe**, and **playsound**. Control the canvas using just your fingers and draw in mid-air — no mouse or pen needed!

---

## ✨ Features

- 👆 **Draw with your index finger** like it’s magic
- 🟢 **Color selection** with your thumb (stay in a color box for a few seconds to select)
- ✊ **Clear canvas** by showing fists with both hands
- 🎵 **Sound effect** when you change colors
- 🔍 **Smoothed drawing** with a deque history buffer

---

## 🚀 Tech Stack

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- playsound
- threading + deque for responsiveness

---

## 🔧 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/gesture-drawing-app.git
   cd gesture-drawing-app
   ```

2. **Install dependencies**

   ```bash
   pip install opencv-python mediapipe numpy playsound
   ```

3. **Place your ding.wav**

   Put your preferred ding sound file at:

   ```
   /Users/pranavkandakurthi/Downloads/Labsheets/ding.wav
   ```

   Or modify the path in the `play_ding()` function.

4. **Run the app**

   ```bash
   python app.py
   ```

---

## 🕹️ Controls

| Action       | Gesture                                |
| ------------ | -------------------------------------- |
| Draw         | Raise only **index finger**            |
| Select Color | Point **thumb** inside top color boxes |
| Clear Canvas | Make a **fist with both hands**        |

---

## 📁 File Structure

```
gesture-drawing-app/
├── app.py
├── ding.wav
└── README.md
```

---

## 💡 Ideas for Future

- Add undo/redo gesture
- Save drawing as image
- Add gesture recognition for shapes

---

## 👨‍💻 Author

**Pranav Kandakurthi**  
🔗 [My Portfolio](https://pranav4417.github.io)

---


## 📜 License

MIT License — feel free to use and modify.

---
