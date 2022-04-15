            # Apply NMS
            results = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.class_filter,
                                       agnostic=self.agnostic_nms)[0]
            results[:, :4] = scale_coords(input_data.shape[2:], results[:, :4], origin_data.shape).round()
            results_numpy = results.cpu().numpy()
            for res in results_numpy:
                det_result = DetResult()

                det_result.current_frame = frame_idx
                det_result.class_id = res[5]
                det_result.confidence = res[4]
                det_result.head_location = []
                det_result.pedestrian_location = res[0:4]
                self.det_result_info.det_results_vector.append(det_result)
