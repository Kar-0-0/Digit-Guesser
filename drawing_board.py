import tkinter as tk

import cv2
import pyautogui as pag
import pygetwindow as gw

import torch
from neural_python import DigitClassifier


def model_pass(img):
    model = torch.load("FFC")
    output = model(torch.Tensor(img))
    print(torch.argmax(output).item())


def draw(event):
    mousex, mousey = event.x, event.y
    brush_size = 10
    drawing_board.create_rectangle(
        mousex, mousey, mousex + brush_size, mousey + brush_size, fill="Black"
    )


def take_screenshot():
    window_loc = gw.getWindowsWithTitle("Digit Classification")[0]

    img = pag.screenshot(
        "number.png",
        region=(
            window_loc.left + 150,
            window_loc.top + 50,
            window_loc.width - 303,
            window_loc.height - 375,
        ),
    )
    img = cv2.imread("number.png")
    img_gray = cv2.cvtColor(img, 6)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    img_flattened = img_normalized.flatten()

    model_pass(img_flattened)


def clear_board():
    drawing_board.delete("all")


window = tk.Tk()
window.geometry("1080x920")
window.title("Digit Classification")

drawing_board = tk.Canvas(
    window,
    height=600,
    width=800,
    bg="White",
    border=5,
    cursor="spraycan",
    relief="solid",
)
drawing_board.place(x=130, y=0)
click1 = drawing_board.bind("<B1-Motion>", draw)

submit_button = tk.Button(
    window, width=15, height=5, text="Submit", command=take_screenshot
)
submit_button.place(x=350, y=700)

clear_button = tk.Button(
    window, width=15, height=5, text="Clear Board", command=clear_board
)
clear_button.place(x=600, y=700)


window.mainloop()
