import cv2
import unittest

from insight_face.deploy.interface import FaceSearcher

class TestInterface(unittest.TestCase):

    def setUp(self):
        self.searcher = FaceSearcher("Resnet", num_layers=50)
        self.searcher.load_state('./output/res50/model_ir_se50.pth', 50)
        self.test_face = cv2.imread('./tests/assets/reba.jpg')

    def test_embedding(self):
        self.searcher.get_embedding([self.test_face])


if __name__ == "__main__":
    unittest.main()