from music21 import converter

if __name__ == "__main__":
    file_path = "./samples/EBC9FA_12-10_01-07-55.pgz"
    score = converter.thaw(file_path)
    score.show()
