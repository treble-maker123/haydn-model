from music21 import converter

if __name__ == "__main__":
    file_path = "./outputs/6C0C1E_12-10_04-29-21.pgz"
    score = converter.thaw(file_path)
    score.show()
