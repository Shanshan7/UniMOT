from tracker.base_tracker import BaseTrack


# class MOTR(BaseTrack):
#     def update(self, dt_instances: Instances):
#         ret = []
#         for i in range(len(dt_instances)):
#             label = dt_instances.labels[i]
#             if label == 0:
#                 id = dt_instances.obj_idxes[i]
#                 box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
#                 ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 6))
#
# class Detector(object):
#     def __init__(self, args):
#
#         self.args = args
#
#         # build model and load weights
#         self.model, _, _ = build_model(args)
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         self.model = load_model(self.model, args.resume)
#         self.model = self.model.cuda()
#         self.model.eval()
#
#         # mkidr save_dir
#         vid_name, prefix = args.input_video.split('/')[-1].split('.')
#         self.save_root = os.path.join(args.output_dir, 'results', vid_name)
#         Path(self.save_root).mkdir(parents=True, exist_ok=True)
#         self.save_img_root = os.path.join(self.save_root, 'imgs')
#         Path(self.save_img_root).mkdir(parents=True, exist_ok=True)
#         self.txt_root = os.path.join(self.save_root, f'{vid_name}.txt')
#         self.vid_root = os.path.join(self.save_root, args.input_video.split('/')[-1])
#
#         # build dataloader and tracker
#         self.dataloader = LoadVideo(args.input_video)
#         self.tr_tracker = MOTR()
#
#     @staticmethod
#     def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
#         keep = dt_instances.scores > prob_threshold
#         return dt_instances[keep]
#
#     @staticmethod
#     def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
#         wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
#         areas = wh[:, 0] * wh[:, 1]
#         keep = areas > area_threshold
#         return dt_instances[keep]
#
#     @staticmethod
#     def write_results(txt_path, frame_id, bbox_xyxy, identities):
#         save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
#         with open(txt_path, 'a') as f:
#             for xyxy, track_id in zip(bbox_xyxy, identities):
#                 if track_id < 0 or track_id is None:
#                     continue
#                 x1, y1, x2, y2 = xyxy
#                 w, h = x2 - x1, y2 - y1
#                 line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
#                 f.write(line)
#
#     @staticmethod
#     def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         if dt_instances.has('scores'):
#             img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
#         else:
#             img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
#         if ref_pts is not None:
#             img_show = draw_points(img_show, ref_pts)
#         if gt_boxes is not None:
#             img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
#         cv2.imwrite(img_path, img_show)
#         return img_show
#
#     def run(self, prob_threshold=0.7, area_threshold=100, vis=True, dump=True):
#         # save as video
#         fps = self.dataloader.frame_rate
#         videowriter = cv2.VideoWriter(self.vid_root, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.dataloader.seq_w, self.dataloader.seq_h))
#         track_instances = None
#         fid = 0
#         for _, cur_img, ori_img in tqdm(self.dataloader):
#             if track_instances is not None:
#                 track_instances.remove('boxes')
#                 track_instances.remove('labels')
#
#             res = self.model.inference_single_image(cur_img.cuda().float(), (self.dataloader.seq_h, self.dataloader.seq_w), track_instances)
#             track_instances = res['track_instances']
#             dt_instances = track_instances.to(torch.device('cpu'))
#
#             # filter det instances by score.
#             dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
#             dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
#
#             if vis:
#                 vis_img_path = os.path.join(self.save_img_root, '{:06d}.jpg'.format(fid))
#                 vis_img = self.visualize_img_with_bbox(vis_img_path, ori_img, dt_instances)
#                 videowriter.write(vis_img)
#
#             if dump:
#                 tracker_outputs = self.tr_tracker.update(dt_instances)
#                 self.write_results(txt_path=self.txt_root,
#                                 frame_id=(fid+1),
#                                 bbox_xyxy=tracker_outputs[:, :4],
#                                 identities=tracker_outputs[:, 5])
#             fid += 1
#         videowriter.release()