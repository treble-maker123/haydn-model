from preprocess import DataToTensor16thMin
from dataset import HaydnQuartetRawDataset
from analysis import MusicAnalysis

dataset = HaydnQuartetRawDataset(
            transform=DataToTensor16thMin,
            analysis=MusicAnalysis)
dataset[0]
