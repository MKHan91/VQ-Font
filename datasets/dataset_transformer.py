from itertools import chain
import copy
from PIL import Image, ImageFile
import numpy as np
import random
import os
import json
# import paddle
import torch
from torch.utils.data import Dataset, DataLoader
from .lmdbutils import read_data_from_lmdb
import threading
import torchvision.transforms as T

import cv2
from skimage import morphology
ImageFile.LOAD_TRUNCATED_IMAGES = True


# region - Train용 
class CombTrainDataset(Dataset):
    """
    CombTrainDataset
    """
    def __init__(self, env, env_get, train_font_dict, content_reference_json, content_font, 
                 transform=None):
        self.env = env
        self.env_get = env_get
                
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)
        
        self.brush_font = "reference_images_v2"
        # # ----- train_font_dict에서 "reference_images_v2" 제외 -----
        # self.brush_font = "reference_images_v2"
        # self.brush_unis = train_font_dict.get(self.brush_font, [])
        # excluded_fonts = {self.brush_font}
        # train_font_dict = {k: v for k, v in train_font_dict.items() if k not in excluded_fonts}
        # # ----------------------------------------------------------
        
        self.train_font_dict = train_font_dict
        self.content_chars = sorted(list(self.cr_mapping.keys()))
        self.n_content_chars = len(self.content_chars)
        self.train_font_names = list(self.train_font_dict)
        self.n_fonts = len(self.train_font_names)
        self.transform = transform
        self.content_font_name = content_font
        
        print ('#'*30 + f' number of content_chars: {self.n_content_chars} ' + '#'*30)
        print ('#'*30 + f' number of train fonts: {self.n_fonts} ' + '#'*30)
        
        # self.augment = T.Compose([
        #     T.RandomApply([T.RandomAffine(degrees=10, 
        #                                   translate=(0.1,0.1), 
        #                                   scale=(0.9,1.1), 
        #                                   shear=10)],
        #                   p=0.5),
        #     T.RandomPerspective(distortion_scale=0.2, p=0.3)
        # ])
        # ✅ 2단계: 붓글씨 특성에 맞게 augmentation 강화
        self.augment = T.Compose([
            T.RandomApply([T.RandomAffine(degrees=15,
                                        translate=(0.1, 0.1),
                                        scale=(0.85, 1.15),
                                        shear=15)], p=0.7),
            T.RandomPerspective(distortion_scale=0.3, p=0.5),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.4),
            T.RandomApply([T.ElasticTransform(alpha=30.0)], p=0.3),
        ])
        self.n_aug = 1  # augmentation 횟수
        
    # region font cr_mapping
    def sample_pair_style(self, font_name, intersec_train_uni, train_unis):
        trg_uni = intersec_train_uni[0] 
        style_unis = self.cr_mapping[trg_uni]

        try:           
            imgs_ske = [np.asarray(read_data_from_lmdb(self.env, f'{font_name}_{uni}')['img']) for uni in style_unis]
            for i in range(len(imgs_ske)):
                # _, sample = cv2.threshold(imgs_ske[i], 127, 255, cv2.THRESH_BINARY_INV)
                _, binary = cv2.threshold(imgs_ske[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                binary[binary == 255] = 1
                skeleton0 = morphology.skeletonize(binary)
                imgs_ske[i] = (255 - skeleton0.astype(np.uint8)*255)
                
            imgs_ske = torch.cat([self.transform(Image.fromarray(img)) for img in imgs_ske])
            imgs = torch.cat([self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis]) 

        except:
            return None, None
        # imgs = paddle.concat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])

        # return imgs, imgs_ske, list(style_unis)
        return imgs, imgs_ske, style_unis
        
    
    #random return a character both in train font and cr_mapping.keys()
    def get_random_trg(self, train_unis):
        target_list = list(set.intersection(set(train_unis), 
                                            set(self.content_chars)))
        intersec_train_uni = random.choice(target_list)
        
        return [intersec_train_uni]
    
    
        
    def __getitem__(self, index):
        # TODO: 다양한 비율로 해보기.
        # 75% 확률로 붓글씨 폰트 강제 선택, 나머지 25%는 일반 폰트
        if random.random() < 0.75 and self.brush_font in self.train_font_dict:
            train_font_name = self.brush_font
            font_idx = self.train_font_names.index(train_font_name)
        else:
            font_idx = index % self.n_fonts
            train_font_name = self.train_font_names[font_idx]
        
        train_unis = self.train_font_dict[train_font_name]
        
        sample_index = torch.tensor([index])

        while True:
            intersec_train_uni = self.get_random_trg(train_unis)
            # """ 스타일 글자 이미지는 학습 글자 이미지에서 맵핑되는 글자 이미지를 가져옴. (cr_mapping에서 가져옴)"""
            style_imgs, style_imgs_ske, _ = self.sample_pair_style(train_font_name, intersec_train_uni, train_unis)

            if style_imgs is None: 
                print('!!!!!!!!!!!!!!!!!!!!!!!! style image is None !!!!!!!!!!!!!!!!!!!!!!!!')
                continue

            # -------------------- augmentation 적용 --------------------
            style_imgs_aug = []
            style_imgs_ske_aug = []
            for img, ske in zip(style_imgs, style_imgs_ske):
                img, ske = img.unsqueeze(0), ske.unsqueeze(0)
                combined = torch.cat([img, ske], dim=0)
                combined = self.augment(combined)
                
                for _ in range(self.n_aug):
                    style_imgs_aug.append(combined[0:1])
                    style_imgs_ske_aug.append(combined[1:2])
            # ----------------------------------------------------------
            
            
            style_imgs = torch.cat(style_imgs_aug)
            style_imgs_ske = torch.cat(style_imgs_ske_aug)
            
            trg_imgs = torch.cat([self.env_get(self.env, train_font_name, uni, self.transform)
                                  for uni in intersec_train_uni])
            
            trg_uni_ids = [self.content_chars.index(uni) for uni in intersec_train_uni]
            font_idx = torch.tensor([font_idx])
            
            content_imgs = torch.cat([self.env_get(self.env, self.content_font_name, uni, self.transform)
                                      for uni in intersec_train_uni]).unsqueeze_(1)
           
            content_imgs_ske = [np.asarray(read_data_from_lmdb(self.env, f'{self.content_font_name}_{uni}')['img']) for uni in intersec_train_uni]
            for i in range(len(content_imgs_ske)):
                _,binary = cv2.threshold(content_imgs_ske[i],127,255,cv2.THRESH_BINARY_INV)
                binary[binary==255] = 1
                skeleton0 = morphology.skeletonize(binary)
                content_imgs_ske[i] = (255-skeleton0.astype(np.uint8)*255)
            
            content_imgs_ske = torch.cat([self.transform(Image.fromarray(img)) for img in content_imgs_ske])

            ret = (
                torch.repeat_interleave(font_idx, len(style_imgs)), # font_idx의 값을 len(style_imgs)) 번
                style_imgs,
                style_imgs_ske,
                torch.repeat_interleave(font_idx, len(trg_imgs)),
                torch.tensor(trg_uni_ids),
                trg_imgs,
                content_imgs,
                content_imgs_ske,
                intersec_train_uni,
                torch.repeat_interleave(sample_index, len(style_imgs)),
                sample_index 
            )
            
            return ret


    def __len__(self):
        return sum([len(v) for v in self.train_font_dict.values()])


    @staticmethod
    def collate_fn(batch):
        (style_ids, style_imgs,style_imgs_ske,
         trg_ids, trg_uni_ids, trg_imgs, content_imgs,content_imgs_ske, trg_unis, style_sample_index, trg_sample_index) = zip(*batch) 
        
        #print (style_comp_ids)

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(style_imgs_ske).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_uni_ids),
            torch.cat(trg_imgs).unsqueeze_(1),
            torch.cat(content_imgs),
            torch.cat(content_imgs_ske).unsqueeze_(1),
            trg_unis,
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index)
        )
        
        return ret


# region - Test 용
class CombTestDataset(Dataset):
    """
    CombTestDataset
    """
    def __init__(self, env, env_get, target_fu, avails, content_reference_json, content_font, language="chn",
                 transform=None, ret_targets=True):

        self.fonts = list(target_fu)
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]])
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis]
        self.unis = sorted(set.union(*map(set, avails.values())))
        self.env = env
        self.env_get = env_get
        self.avails = avails
            
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)
            
        self.train_unis = sorted(set.union(*map(set, self.cr_mapping.values())))
        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font_name = content_font

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: int(x, 16),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]
        
    def sample_pair_style(self, trg_uni, avail_unis):
        if trg_uni not in self.cr_mapping:
            style_unis = random.sample(avail_unis, 3)
        else:
            style_ref = self.cr_mapping[trg_uni]
            style_unis = []
            for ref in style_ref:
                if ref in avail_unis:
                    style_unis.append(ref)
                else:
                    style_unis.append(random.choice(avail_unis))
        return list(style_unis)
    

    def __getitem__(self, index):
        font_name, trg_uni = self.fus[index]
        font_idx = self.fonts.index(font_name)
        sample_index = torch.tensor([index])
        
        avail_unis = self.avails[font_name]
        style_unis = self.sample_pair_style(trg_uni, avail_unis)
        
        try:
            imgs_ske = [np.asarray(read_data_from_lmdb(self.env,f'{font_name}_{uni}')['img']) for uni in style_unis]
            for i in range(len(imgs_ske)):
                _,binary = cv2.threshold(imgs_ske[i],127,255,cv2.THRESH_BINARY_INV)
                binary[binary==255] = 1
                skeleton0 = morphology.skeletonize(binary)
                imgs_ske[i] = (255-skeleton0.astype(np.uint8)*255)
                
            b = [self.transform(Image.fromarray(img)) for img in imgs_ske]
            
            # a = [self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis]
            
            a = [self.env_get(self.env, "reference_images_v2", uni, self.transform) for uni in style_unis]
            
        except:
            print (font_name, style_unis)

        style_imgs = torch.stack(a)
        # st = style_imgs[0][0].cpu().numpy()
        style_imgs_ske =  torch.stack(b)
        
        font_idx = torch.tensor([font_idx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        
        content_img = self.env_get(self.env, self.content_font_name, trg_uni, self.transform)
        
        content_imgs_ske = np.asarray(read_data_from_lmdb(self.env,f'{self.content_font_name}_{trg_uni}')['img'])
        _,binary = cv2.threshold(content_imgs_ske,127,255,cv2.THRESH_BINARY_INV)
        binary[binary==255] = 1
        skeleton0 = morphology.skeletonize(binary)
        content_imgs_ske = (255-skeleton0.astype(np.uint8)*255)
        content_imgs_ske = self.transform(Image.fromarray(content_imgs_ske))

        
        ret = (
            torch.repeat_interleave(font_idx, len(style_imgs)),
            style_imgs,
            style_imgs_ske,
            font_idx,
            trg_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)), #style sample index
            sample_index, #trg sample index
            content_img,
            content_imgs_ske,
            
        )
        
        if self.ret_targets:
            try:
                trg_img = self.env_get(self.env, font_name, trg_uni, self.transform)
            except:
                trg_img = torch.ones(1, 128, 128)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs,style_imgs_ske, trg_ids, trg_unis, style_sample_index, trg_sample_index, content_imgs,content_imgs_ske, *left = list(zip(*batch))
        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs),
            torch.cat(style_imgs_ske),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            torch.cat(content_imgs_ske).unsqueeze_(1),
            
        )

        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret
    
    
class FixedRefDataset(Dataset):
    '''
    FixedRefDataset
    '''
    def __init__(self, env, env_get, target_dict, ref_unis, k_shot, content_reference_json, content_font, language="chn",  transform=None, ret_targets=True):
        '''
        ref_unis: target unis
        target_dict: {style_font: [uni1, uni2, uni3]} gen_fonts:gen_unis
        '''
        self.target_dict = target_dict
        self.ref_unis = sorted(ref_unis)
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]
        self.k_shot = k_shot
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)
            
        self.content_font_name = content_font
        self.fonts = list(target_dict)

        self.env = env
        self.env_get = env_get
        
        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]
        
    def sample_pair_style(self, font, trg_uni):
        assert trg_uni in self.cr_mapping, "infer uni is not in your content reference map"
        style_unis = self.cr_mapping[trg_uni]
        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])
        return imgs, list(style_unis)


    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])
        
        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))
        style_imgs, style_unis = self.sample_pair_style(fname, trg_uni)
        
        fidces = torch.tensor([fidx])
        
        # 내가 수정한 부분
        # --------------------------------------------------------------------------------
        trg_uni_char = chr(int(trg_uni, 16))
        style_unis_char = [chr(int(style_uni, 16)) for style_uni in style_unis]
        # --------------------------------------------------------------------------------
        
        trg_dec_uni =torch.tensor([self.to_int(trg_uni_char)])
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis_char])
        
        content_img = self.env_get(self.env, self.content_font_name, trg_uni, self.transform)
        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)), #fidces,
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img
        )

        if self.ret_targets:
            trg_img = self.env_user_get(self.env_user, fname, trg_uni, self.transform)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs, trg_ids, trg_unis, style_unis, style_sample_index, trg_sample_index, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1)
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret


class FixedRefDataset_random(Dataset):
    '''
    FixedRefDataset
    '''

    def __init__(self, env, env_get, target_dict, ref_unis, k_shot, content_reference_json, content_font,
                 language="chn", transform=None, ret_targets=True):
        '''
        ref_unis: target unis
        target_dict: {style_font: [uni1, uni2, uni3]} gen_fonts:gen_unis
        '''
        self.target_dict = target_dict
        self.ref_unis = sorted(ref_unis)
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]
        self.k_shot = k_shot
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)

        self.content_font_name = content_font
        self.fonts = list(target_dict)

        self.env = env
        self.env_get = env_get

        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }

        self.to_int = to_int_dict[language.lower()]

    def sample_pair_style(self, font, trg_uni):
        assert trg_uni in self.cr_mapping, "infer uni is not in your content reference map"
        style_unis = self.ref_unis
        print('self.ref_unis:',self.ref_unis)
        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])
        return imgs, list(style_unis)

    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])

        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))
        style_imgs, style_unis = self.sample_pair_style(fname, trg_uni)

        fidces = torch.tensor([fidx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis])

        content_img = self.env_get(self.env, self.content_font_name, trg_uni, self.transform)
        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)),  # fidces,
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img
        )

        if self.ret_targets:
            trg_img = self.env_user_get(self.env_user, fname, trg_uni, self.transform)
            ret += (trg_img,)

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs, trg_ids, trg_unis, style_unis, style_sample_index, trg_sample_index, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1)
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret

