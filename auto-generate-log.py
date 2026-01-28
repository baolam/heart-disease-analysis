PATH = "E:\\_university_\\5.SP2026\\ADY201m\\project\\logs"

import os
print(os.listdir(PATH))

folders = os.listdir(PATH)[1:]
print("Cấu trúc đường dẫn:", folders)

import fitz