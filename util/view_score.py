from music21 import converter

if __name__ == "__main__":
    file_path = "./samples/D0F52E_12-09_04-47-50.pgz"
    score = converter.thaw(file_path)
    score.show()
