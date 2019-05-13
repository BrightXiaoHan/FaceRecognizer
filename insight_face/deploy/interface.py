import os
import glob

import torch
import easydict
import cv2
import numpy as np

from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

from mtcnn import FaceDetector, BatchImageDetector, get_net_caffe, align_multi
from insight_face.network import get_by_name
from insight_face.utils.wrapper import no_grad
from insight_face.utils.exception import EmptyTensorException, MultiFaceException, NoSuchNameException
from insight_face.utils.func import img_transform


class FaceSearcher(object):

    def __init__(self, backbone, caffe_model_path, device='cpu', **kwargs):

        self.net = get_by_name(backbone, **kwargs)
        self.device = torch.device(device)
        self.net.to(device)
        self.net.eval()

        pnet, rnet, onet = get_net_caffe(caffe_model_path)
        self.detector = FaceDetector(pnet, rnet, onet, device=device)
        self.batch_detector = BatchImageDetector(pnet, rnet, onet, device=device)

        self.banks = easydict.EasyDict()

        self.single_face_detect_params = easydict.EasyDict(
            threshold=[0.5, 0.6, 0.7],
            factor=0.7,
            minsize=48,
            nms_threshold=[0.7, 0.7, 0]
        )

        self.multi_face_detect_params = easydict.EasyDict(
            threshold=[0.6, 0.7, 0.85],
            factor=0.7,
            minsize=24,
            nms_threshold=[0.7, 0.7, 0.3]
        )

        self.recog_params = easydict.EasyDict(
            verify_threshold=0.5,
            one2many_threshold=0.5,
            many2many_threshold=0.5,
            stranger_threshold=0.3
        )

    def update_params(self, name, **kwargs):
        params = getattr(self, name, None)

        if params is None:
            raise NoSuchNameException("No parameter set named %s." % name)

        params.update(kwargs)

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
            faces (np.ndarray or list): Image matrix with BGR(opencv default format) format.
        
        Raises:
            EmptyTensorException: Rasied when "faces" is a empty array.
        
        Returns:
            torch.Tensor: Embedding tensor with shape [num_img, emb_dim]
        """
        if len(faces) == 0:
            raise EmptyTensorException("No image in parameter 'faces'.")

        if isinstance(faces, np.ndarray) and len(faces.shape) == 3:
            faces = [faces]

        tensors = []
        for img in faces:
            t = img_transform(img).to(self.device) 
            tensors.append(t)

        tensors = torch.stack(tensors)
        source_embs = self.net(tensors)
        return source_embs

    def cosine_sim(self, source_embs, target_embs, _reduce_max=True):
        """

        Args:
            source_embs (torch.Tensor): shape [n, emb_dim]
            target_embs (torch.Tensor): shape [m, emb_dim]
            _reduce_max (bool): Internal parameter, don't use it.

        Returns:
            (torch.Tensor, torch.Tensor): nearest cosine distance and index of target.
        """

        source_shape = source_embs.shape[0]
        target_shape = target_embs.shape[0]

        source_embs = torch.stack([source_embs] * target_shape, 1)
        target_embs = torch.stack([target_embs] * source_shape, 0)
        sim = torch.nn.functional.cosine_similarity(source_embs, target_embs, 2)
        
        if _reduce_max:
            best_sim, sim_index = sim.max(1)

            return best_sim, sim_index
        
        else:
            return sim

    @no_grad
    def verify(self, source, target, aligned=False):
        """Face verification by compute the cosine value between source face and target face.
        
        Args:
            source (np.ndarray): Image matrix with BGR(opencv default format) format.
            target (np.ndarray): Image matrix with BGR(opencv default format) format.
            threshold (float): If the cosine value between two embedded vectors is more than 'threshold', they will be considered as a same person.
            aligned (bool, optional): Defaults to False. If False, detect and align faces before verification.
        
        Returns:
            list: Verification result. True means that they are the same person. 
        """

        detect = lambda x: self.detector.detect(x, **self.single_face_detect_params)
        if not aligned:
            source_boxes, source_landmarks = detect(source)
            _, source = align_multi(source, source_boxes, source_landmarks)

            target_boxes, target_landmarks = detect(target)
            _, target = align_multi(target, target_boxes, target_landmarks)

        if len(source) > 1:
            raise MultiFaceException("More than one face are detected in 'source' image.")

        if len(target) > 1:
            raise MultiFaceException("More than one face are detected in 'target' image.")

        if len(source) < 1:
            raise EmptyTensorException("No face is detected in 'source' image.")

        if len(target) < 1:
            raise EmptyTensorException("No face is detected in 'target' image.")

        source_embs = self.get_embedding(source)
        target_embs = self.get_embedding(target)

        sims = torch.nn.functional.cosine_similarity(source_embs, target_embs, 1)
        result = (sims >= self.recog_params.verify_threshold).cpu().numpy().astype(bool).tolist()

        return result[0]

    def add_face_bank(self, path, force_reload=False, save_intermediate_result=True, suffix='jpg', bank_name='default'):
        """从数据文件夹中加载数据

        Args:
            path (str): 人脸库文件夹的绝对路径。路径下每个人的照片保存到以名字命名的文件夹中, 图片以jpg的格式存储
            force_reload (bool, optional): Defaults to False. 是否强制重新加载
            save_intermediate_result (bool, optional): Defaults to False. 是否保存生成的向量到embedding_matrix.npy。保存生成的向量可以提高下次加载模型的速度
            suffix (str, optional): Defaults to "jpg": Image format.
            name (str, optional): Defaults to "default": Label of this face bank.
        """

        database = {}
        error_list = []  # Record the path of error image.
        error_type = []  # What's the problem with error file. No face is detected (type 0) or multiple faces are detected(tpye 1).

        categories = os.listdir(path)
        for category in categories:
            if not os.path.isdir(os.path.join(path, category)):
                continue
            intermediate_result = os.path.join(
                path, category, 'embedding_matrix.npy')
            error_file = os.path.join(path, category, 'error_list.txt')

            # 从中间结果加载
            if os.path.exists(intermediate_result) and not force_reload and os.path.exists(error_file):
                embedding_matrix = np.load(intermediate_result)
                with open(error_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        fn, t = line.split(',')
                        error_list.append(os.path.join(path, category, fn))
                        error_type.append(int(t))
                # end if
            # 从原始图片加载
            elif os.path.exists(error_file) and not force_reload:
                with open(error_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        fn, t = line.split(',')
                        error_list.append(os.path.join(path, category, fn))
                        error_type.append(int(t))
                continue
            else:
                image_files = glob.iglob(
                    os.path.join(path, category, '*.' + suffix))
                image_list = []
                error_this_category = []
                error_type_category = []

                aligned_faces_folder = os.path.join(path, category, 'aligned')
                if not os.path.isdir(aligned_faces_folder):
                    os.makedirs(aligned_faces_folder)

                for image_file in image_files:  # 遍历类别下的所有照片
                    img = cv2.imread(image_file)
                    boxes, landmarks = self.detector.detect(img, **self.single_face_detect_params)
                    boxes, faces = align_multi(img, boxes, landmarks, crop_size=(112, 112))

                    # 照片中没有检测到人脸，或者检测到两张人脸，跳过该照片并记录错误信息
                    if len(boxes) == 0:
                        print("Image has no face detected. %s." % image_file)
                        error_this_category.append(os.path.basename(image_file))
                        error_list.append(image_file)
                        error_type.append(0)
                        error_type_category.append(0)
                        continue

                    # 检测到多张人脸
                    elif len(boxes) > 1:

                        print("Detect more than one face in this image %s." % image_file)
                        error_this_category.append(os.path.basename(image_file))
                        error_list.append(image_file)
                        error_type.append(1)
                        error_type_category.append(1)
                        continue

                    else:
                        image_list.append(faces[0])
                        cv2.imwrite(os.path.join(aligned_faces_folder, os.path.basename(image_file)), faces[0])
                    # end for loop

                # Record path of error image.
                with open(error_file, 'w') as f:
                    for error, err_t in zip(error_this_category, error_type_category):
                        f.write(error + ',' + str(err_t) + '\n')

                if len(image_list) == 0:  # 如果整个文件夹都没检测到人脸
                    print("Invalid category %s, no face has been detect." % category)
                    continue

                embedding_matrix = self.get_embedding(image_list).cpu().numpy()

                if save_intermediate_result:
                    np.save(intermediate_result, embedding_matrix)
                # end else

            database[category] = embedding_matrix

        # Map index to name
        index2name = list()

        feature_list = list()
        for name, images in database.items():
            feature_list.append(images)
            index2name.extend([name] * images.shape[0])

        if len(feature_list) == 0:
            raise Exception("There is no face in this folder. Please check your datasets.")

        feature_matrix = np.concatenate(feature_list)

        self.banks[bank_name] = easydict.EasyDict(
            feature_matrix=torch.tensor(feature_matrix, device=self.device),
            index2name=index2name,
            error_list=error_list,
            error_type=error_type
        )

    def search(self, image, face_bank='default'):
        """Search all faces detected in given image.
        
        Args:
            image (str or np.ndarray): Image path or numpy array returned by cv2.imread.
            face_bank (str, optional): Search targets added by "add_face_bank" method. Defaults to 'default'.
        
        Raises:
            NoSuchNameException: Quote wbefore specifying
        
        Returns:
            tuple: faces list of (112, 112, 3), names (list), best_sim (list), boxes(list), landmarks(list)
        """
        if face_bank not in self.banks:
            raise NoSuchNameException("No face bank named %s. You can add face bank by add_face_bank method." % face_bank)

        bank = self.banks[face_bank]

        detect = lambda x: self.detector.detect(x, **self.multi_face_detect_params)
        source_boxes, source_landmarks = detect(image)

        if len(source_boxes) == 0:
            return [], [], [], [], []

        _, face_img = align_multi(image, source_boxes, source_landmarks)

        emb = self.get_embedding(face_img)
        target_emb = bank.feature_matrix

        best_sim, sim_index = self.cosine_sim(emb, target_emb)

        threshold_mask = best_sim >= self.recog_params.one2many_threshold
        best_sim = best_sim[threshold_mask].cpu().numpy().tolist()
        sim_index = sim_index[threshold_mask].cpu().numpy().tolist()
        landmarks = source_landmarks[threshold_mask].cpu().numpy().tolist()
        boxes = source_boxes[threshold_mask].cpu().numpy().tolist()
        names = [bank.index2name[i] for i in sim_index]

        face_image_index = threshold_mask.nonzero().squeeze().cpu().numpy().tolist()
        if isinstance(face_image_index, int):
            face_image_index = [face_image_index]
        faces = [face_img[i] for i in face_image_index]

        return faces, names, best_sim, boxes, landmarks

    def search_aligned_faces(self, face_img, face_bank="default"):
        bank = self.banks[face_bank]

        emb = self.get_embedding(face_img)
        target_emb = bank.feature_matrix

        best_sim, sim_index = self.cosine_sim(emb, target_emb)

        threshold_mask = best_sim >= self.recog_params.one2many_threshold
        best_sim = best_sim[threshold_mask].cpu().numpy().tolist()
        sim_index = sim_index[threshold_mask].cpu().numpy().tolist()
        names = [bank.index2name[i] for i in sim_index]

        face_image_index = threshold_mask.nonzero().squeeze().cpu().numpy().tolist()
        if isinstance(face_image_index, int):
            face_image_index = [face_image_index]
        faces = [face_img[i] for i in face_image_index]

        return faces, names, best_sim

    def recognize_and_identify_strangers(self, image, face_bank='default', stranger_identifier='stranger'):
        """Search all faces detected in given image. Recognize acquaintance and identify strangers.
        
        Args:
            image (str or np.ndarray): Image path or numpy array returned by cv2.imread.
            face_bank (str, optional): Search targets added by "add_face_bank" method. Defaults to 'default'.
        
        Raises:
            NoSuchNameException: Quote wbefore specifying
        
        Returns:
            acquaintance: faces list of (112, 112, 3), names (list), best_sim (list), boxes(list), landmarks(list)
            stranger: faces list of (112, 112, 3), names (list), best_sim (list), boxes(list), landmarks(list)
        """
        if face_bank not in self.banks:
            raise NoSuchNameException("No face bank named %s. You can add face bank by add_face_bank method." % face_bank)

        bank = self.banks[face_bank]

        detect = lambda x: self.detector.detect(x, **self.multi_face_detect_params)
        source_boxes, source_landmarks = detect(image)

        if len(source_boxes) == 0:
            return [], [], [], [], []

        _, face_img = align_multi(image, source_boxes, source_landmarks)

        emb = self.get_embedding(face_img)
        target_emb = bank.feature_matrix

        sims, indexes = self.cosine_sim(emb, target_emb)

        # acquaintace 
        threshold_mask = sims >= self.recog_params.one2many_threshold
        best_sim = sims[threshold_mask].cpu().numpy().tolist()
        sim_index = indexes[threshold_mask].cpu().numpy().tolist()
        landmarks = source_landmarks[threshold_mask].cpu().numpy().tolist()
        boxes = source_boxes[threshold_mask].cpu().numpy().tolist()
        names = [bank.index2name[i] for i in sim_index]

        face_image_index = threshold_mask.nonzero().squeeze().cpu().numpy().tolist()
        if isinstance(face_image_index, int):
            face_image_index = [face_image_index]
        faces = [face_img[i] for i in face_image_index]

        acquaintance = [faces, names, best_sim, boxes, landmarks]

        # stranger
        threshold_mask = sims <= self.recog_params.stranger_threshold
        best_sim = sims[threshold_mask].cpu().numpy().tolist()
        sim_index = indexes[threshold_mask].cpu().numpy().tolist()
        landmarks = source_landmarks[threshold_mask].cpu().numpy().tolist()
        boxes = source_boxes[threshold_mask].cpu().numpy().tolist()
        names = [stranger_identifier] * len(sim_index)

        face_image_index = threshold_mask.nonzero().squeeze().cpu().numpy().tolist()
        if isinstance(face_image_index, int):
            face_image_index = [face_image_index]
        faces = [face_img[i] for i in face_image_index]

        stranger = [faces, names, best_sim, boxes, landmarks]

        return acquaintance, stranger

         
    def match(self, source, target):
        """Match the most similar person between two images. (hungarian algorithm)
        
        Args:
            source (np.ndarray): Image matrix returned by cv2.imread.
            target (np.ndarray): Image matrix returned by cv2.imread.

        Returns:
            Match list [source_face, target_face, similarity]  
        """
        detect = lambda x: self.detector.detect(x, **self.multi_face_detect_params)
        source_boxes, source_landmarks = detect(source)
        target_boxes, target_landmarks = detect(target)

        if len(source_boxes) == 0 or len(target_boxes) == 0:
            return []

        _, source_faces = align_multi(source, source_boxes, source_landmarks)
        _, target_faces = align_multi(target, target_boxes, target_landmarks)


        source_emb = self.get_embedding(source_faces)
        target_emb = self.get_embedding(target_faces)

        
        sims = self.cosine_sim(source_emb, target_emb, _reduce_max=False).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-sims)
        match_sim = sims[row_ind, col_ind]

        source_faces = source_faces[row_ind]
        target_faces = target_faces[col_ind]

        # filter 
        mask = match_sim >= self.recog_params.many2many_threshold
        match_sim = match_sim[mask]
        source_faces = source_faces[mask]
        target_faces = target_faces[mask]

        return source_faces, target_faces, match_sim

    def embedding_faces_in_the_wild(self, img):
        detect = lambda x: self.detector.detect(x, **self.multi_face_detect_params)
        source_boxes, source_landmarks = detect(img)

        if len(source_boxes) == 0:
            return torch.empty((0, 512))

        _, face_img = align_multi(img, source_boxes, source_landmarks)

        emb = self.get_embedding(face_img)

        return emb, source_boxes, source_landmarks