from music21 import converter as conv
from load_data import list_downloaded_data as list_data

def __get_scores__(quiet=False):
    '''
    Returns:
        list: A list containing the music21.stream.Score objects.
    '''
    scores = []
    for filename in list_data():
        try:
            scores.append(conv.parse('data/' + filename))
        except Exception as error:
            if not quiet:
                print("Encountered error {} with {}.".format(error, filename))
    return scores
