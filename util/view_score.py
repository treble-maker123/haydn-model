import os
from music21 import converter

if __name__ == "__main__":
    # file_path = "./outputs/8C50D6_12-13_21-52-07.pgz"
    cdir = os.getcwd()
    file_path = cdir + "/data/5795802993.pgz"
    score = converter.thaw(file_path)

    # score = score.transpose(3)
    # converter.freeze(score, fp=file_path)
    score.show()
