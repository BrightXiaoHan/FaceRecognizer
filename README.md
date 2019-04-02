# Face Recognition Project 


## Introduction
- This repo is a reimplementation of Arcface[(paper)](https://arxiv.org/abs/1801.07698), or Insightface[(github)](https://github.com/deepinsight/insightface)
- This repo is get inspiration from [insightface](https://github.com/deepinsight/insightface), [Insightface_Pytorh](https://github.com/TreB1eN/InsightFace_Pytorch)

## SetUp
#### Install MTCNN FaceDetector
Follow this instruction [FaceDetector Installation](./FaceDetector/README.md).

#### Install
```
python setup.py install
```

## Basic Usage
#### Download the pre-trained model.
Download from [pre-trained model](https://github.com/faciallab/Models/tree/master/InsightFace) and merge the part files

```bash
cat model_ir_se50.pth* > model_ir_se50.pth
mv model_ir_se50.pth /path/to/FaceRecognizer/output/res50
```

### FaceVerify
Verify if they are the same person.

![reba](./tests/assets/reba/3.jpg)![reba2](./tests/assets/reba/2.jpg)
```py
from insight_face import FaceSearcher

# Load pre-trained res50 model
searcher.load_state('./output/res50/model_ir_se50.pth', 50)

# Load face images
face1 = cv2.imread('./tests/assets/reba/2.jpg')
face2 = cv2.imread('./tests/assets/reba/3.jpg')

result = searcher.verify(face1, face2)

assert result is True
```

### Search
Find who they are. 
![reba](./tests/assets/multi_face.jpg)
```py
face_bank_dir = 'tests/assets/'

searcher.add_face_bank(face_bank_dir, force_reload=True, bank_name='test')
faces, names, best_sim, _, _ = searcher.search(self.multi_face_img, 'test')

print(names)
```
result
```
>> reba
```