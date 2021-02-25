import os
import cv2
import time
import argparse
#import torch
import warnings
import numpy as np
import sys
import glob
from insightface.app.face_analysis import FaceAnalysis
print('path=',os.path)
from insightface.model_zoo import model_zoo, face_recognition
#from retinaface import RetinaFace

DIST_THRESHOLD = 0.7 #2.0
class VideoTracker(object):
    def __init__(self, args, video_path=''):
        self.args = args
        self.video_path = video_path
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.thresh = 0.8
        self.scales = [640,640]
        self.fa = FaceAnalysis()
        #use CPU 
        self.fa.prepare(-1)
        self.init_name="unknown"
        #self.recModel = self.fa.getRecModel()
        self.known_normed_embeddings = []
        self.known_embeddings = []
        self.known_embeddings_norms = []
        self.known_names = []
        for entry in os.listdir(args.known_people_folder):
            dire = os.path.join(args.known_people_folder, entry)
            if os.path.isdir(dire):
                for sub_entry in os.listdir(dire):
                    fl = os.path.join(dire,sub_entry)
                    print(f'Reading {fl}...')
                    if os.path.isfile(fl):
                        im_known = cv2.imread(fl)
                        #just rec, we can add detection if needed
                        #im_embeddings = self.recModel.get_embedding(im_known)
                        #detect & recog face from a perosn
                        rets = self.fa.get(im_known,self.thresh)
                        bbox_im, landmark_im, det_score_im, embedding_im, gender_im, age_im,embedding_norm_im, normed_embedding_im=rets[0]
                        self.known_normed_embeddings.append(normed_embedding_im)
                        self.known_embeddings_norms.append(embedding_norm_im)
                        self.known_embeddings.append(embedding_im)
                        self.known_names.append(entry)
        '''
        #retina face detection
        self.thresh = 0.8
        self.scales = [320,320] #[640,640] #[1024, 1980]

        self.count = 1

        gpuid = -1
        self.detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
        '''
        '''
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        '''
    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
            self.im_shape = frame.shape
        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
            ret, frame = self.vdo.read()
            self.im_shape = frame.shape

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def match_face(self,normed_embeddings,embeddings,embedding_norms):        
        face_names = []
        face_probs = []
        #dists = [[math.fabs((e1 - e2).norm().item()) for e2 in self.known_embeddings] for e1 in embeddings]
        #print('known_embedding=',type(self.known_embeddings),self.known_embeddings)
        #print('embedding=',embeddings)
        dists1 = [[ np.dot(e1,e2)/(np.linalg.norm(e2)*np.linalg.norm(e1))  for e2 in self.known_normed_embeddings] for e1 in normed_embeddings]
        dists = [[ np.dot(e1,e2)/(self.known_embeddings_norms[j]*embedding_norms[i])  for j, e2 in enumerate(self.known_embeddings)] for i,e1 in enumerate(embeddings)]

        #dists = [[np.sum(np.square(e2-e1)) for e2 in self.known_embeddings] for e1 in embeddings] 
        #dists = [[np.linalg.norm(e1-e2) for e2 in self.known_embeddings] for e1 in embeddings]
        print('dists1,dist',dists1,dists)
        min_dists = [min(d) for d in dists]
        min_dists_thres = [(DIST_THRESHOLD if t>=DIST_THRESHOLD else t) for t in min_dists]
        max_dists = max(min_dists_thres)
        num_dists = len(min_dists)
        print("face min_dist",min_dists)
        for k,md in enumerate(min_dists):
            if md < DIST_THRESHOLD:
                min_index = dists[k].index(md)
                face_names.append(self.known_names[min_index])
                if num_dists ==1 :
                    prob = 1.0
                else:
                    prob = 1.0 - md/max_dists #1.0/(num_dists-1)-md/((num_dists-1)*sum_dists_thres)
                face_probs.append(prob)
            else:
                face_names.append(self.init_name)
                face_probs.append(0.0)
        print('face_names=',face_names)
        return face_names,face_probs

    def run(self):
        results = []
        idx_frame = 0
        #print("###im w,h=",self.im_width,self.im_height)
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(self.im_shape[0:2])
        im_size_max = np.max(self.im_shape[0:2])
        #im_scale = 1.0
        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        #im_scale = 1.0
        print('im_scale', im_scale)
        flip = False

        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            s = time.time()
            rets = self.fa.get(ori_im,self.thresh,im_scale)
             
            #for c in range(self.count):
            #    faces, landmarks = self.detector.detect(ori_im,self.thresh,scales=scales,do_flip=flip)
            e = time.time()
            print("##### face analysis= ",e-s)
            if len(rets) == 0:
                if self.args.save_path:
                    self.writer.write(ori_im)
                continue
            normed_embeddings = []
            embeddings = []
            embedding_norms = []
            for ret in rets:                
                normed_embeddings.append(ret[7])
                embeddings.append(ret[3])
                embedding_norms.append(ret[6])
            #print(' normed_embeddings =',type( normed_embeddings),len( normed_embeddings))
            face_names,face_probs = self.match_face( normed_embeddings,embeddings,embedding_norms)
            for i,ret in enumerate(rets):
                bbox, landmark, det_score, embedding, gender, age,embedding_norm, normed_embedding = ret
                box = bbox.astype(np.int)
                color = (0, 0, 255)
                cv2.rectangle(ori_im, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                landmark5 = landmark.astype(np.int)
                for l in range(landmark5.shape[0]):
                     color = (0, 0, 255)
                     if l == 0 or l == 3:
                         color = (0, 255, 0)
                         cv2.circle(ori_im, (landmark5[l][0], landmark5[l][1]), 1, color,2)
            
                cv2.putText(ori_im,face_names[i],(box[0],box[3]-5), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
            end = time.time()
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            #write_results(self.save_results_path, results, 'mot')

            print("time: {:.03f}s, fps: {:.03f} " \
                             .format(end - start, 1 / (end - start)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default='')
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=False)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="0")
    parser.add_argument("--known_people_folder", type=str, default="./user_photos/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()
