import torch

from collections import OrderedDict

from mtcnn import FaceDetector, get_net_caffe
from insight_face.network import get_by_name
from insight_face.utils.wrapper import no_grad
from insight_face.utils.exception import EmptyTensorException
from insight_face.utils.func import img_transform

class FaceSearcher(object):

    def __init__(self, backbone, device='cpu', **kwargs):

        self.net = get_by_name(backbone, **kwargs)
        self.device = torch.device(device)
        self.net.to(device)
        self.net.eval()

    def load_state(self, model_path, from_muti_gpu=False):

        state_dict = torch.load(model_path)
        if from_muti_gpu:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v

            state_dict = new_state_dict
        
        self.net.load_state_dict(state_dict)


    @no_grad
    def get_embedding(self, faces):
        """Get embedding tensors for given faces
        
        Args:
            faces (np.ndarray): Image matrix with RGB format
        
        Raises:
            EmptyTensorException: Rasied when "faces" is a empty array.
        
        Returns:
            torch.Tensor: Embedding tensor with shape [num_img, emb_dim]
        """


        if len(faces) == 0:
            raise EmptyTensorException("No image in parameter 'faces'.")
        embs = []
        tensors = []
        for img in faces:
            t = img_transform(img).to(self.device) 
            tensors.append(t)

        tensors = torch.stack(tensors)
        source_embs = self.net(tensors)
        return source_embs

    