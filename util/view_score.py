from music21 import converter

if __name__ == "__main__":
    file_path = "./outputs/4159EF_12-10_04-50-43.pgz"
    score = converter.thaw(file_path)
    score.show()
