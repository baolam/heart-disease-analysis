PATH = "E:\\_university_\\5.SP2026\\ADY201m\\project\\logs"

import os
print(os.listdir(PATH))

folders = os.listdir(PATH)[1:]
print("Cấu trúc đường dẫn:", folders)

import fitz
from typing import List
filename = "Group1_AI_log.pdf"

def generate_file():
    doc = fitz.open()
    return doc

def generate_first_page(doc, heading_page="heading_page.pdf"):
    tempo = fitz.open(heading_page)
    doc.insert_pdf(tempo)
    tempo.close()


def add_file(doc, filename):
    tempo = fitz.open(filename)
    print(tempo.page_count)
    doc.insert_pdf(tempo)
    tempo.close()

def get_pdf(files):
    for file in files:
        if file.split('.')[1] == 'pdf':
            return file
    return None

doc = generate_file()
generate_first_page(doc)
for folder_name in folders:
    interact = os.path.join(PATH, folder_name)
    file = get_pdf(os.listdir(interact))
    _filename = os.path.join(interact, file) if file is not None else None
    if _filename:
        print("Xử lý:", _filename)
        add_file(doc, os.path.join(interact, _filename))
doc.save(filename)
doc.close()