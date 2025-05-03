import numpy as np

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # List of dictionaries storing track info
        self.frame_count = 0
        self.next_id = 1
    
    def _iou(self, bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        area = w * h
        
        # Compute IOU
        area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        iou = area / (area1 + area2 - area) if (area1 + area2 - area) > 0 else 0
        
        return iou
    
    def _predict_next_bbox(self, bbox, velocity=None):
        if velocity is None:
            # No velocity information, return same bbox
            return bbox.copy()
        
        # Apply velocity to get predicted bbox
        pred_bbox = bbox.copy()
        pred_bbox[0] += velocity[0]  # x1 += vx
        pred_bbox[1] += velocity[1]  # y1 += vy
        pred_bbox[2] += velocity[0]  # x2 += vx
        pred_bbox[3] += velocity[1]  # y2 += vy
        
        return pred_bbox
    
    def _calculate_velocity(self, old_bbox, new_bbox):
        old_cx = (old_bbox[0] + old_bbox[2]) / 2
        old_cy = (old_bbox[1] + old_bbox[3]) / 2
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        vx = new_cx - old_cx
        vy = new_cy - old_cy
        damping = 0.5
        return [vx * damping, vy * damping]
    
    def update(self, dets=None):
        self.frame_count += 1
        trks = np.empty((0, 5))
        if dets is None or len(dets) == 0:
            for tracker in self.trackers:
                tracker['age'] += 1
                if tracker['age'] > self.max_age:
                    continue
                bbox = self._predict_next_bbox(tracker['bbox'], tracker.get('velocity'))
                tracker['bbox'] = bbox
                trks = np.append(trks, [np.append(bbox, tracker['id'])], axis=0)
            self.trackers = [t for t in self.trackers if t['age'] <= self.max_age]
            return trks
        
        predicted_bboxes = []
        for tracker in self.trackers:
            pred_bbox = self._predict_next_bbox(tracker['bbox'], tracker.get('velocity'))
            predicted_bboxes.append(pred_bbox)

        matched_indices = []
        unmatched_detections = list(range(len(dets)))
        unmatched_trackers = list(range(len(self.trackers)))
        
        if len(predicted_bboxes) > 0 and len(dets) > 0:
            for t in range(len(predicted_bboxes)):
                best_iou = self.iou_threshold
                best_detection = -1
                for d in unmatched_detections.copy():
                    iou_val = self._iou(predicted_bboxes[t], dets[d, :4])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_detection = d

                if best_detection >= 0:
                    matched_indices.append((t, best_detection))
                    unmatched_detections.remove(best_detection)
                    if t in unmatched_trackers:
                        unmatched_trackers.remove(t)
        
        # Update matched trackers
        for t, d in matched_indices:
            # Get tracker and detection
            tracker = self.trackers[t]
            det_bbox = dets[d, :4]
            velocity = self._calculate_velocity(tracker['bbox'], det_bbox)
            tracker['bbox'] = det_bbox
            tracker['age'] = 0
            tracker['hits'] += 1
            tracker['velocity'] = velocity
            if tracker['hits'] >= self.min_hits or self.frame_count <= self.min_hits:
                trks = np.append(trks, [np.append(det_bbox, tracker['id'])], axis=0)
        
        for t in unmatched_trackers:
            tracker = self.trackers[t]
            tracker['age'] += 1
            if tracker['age'] > self.max_age:
                continue
            tracker['bbox'] = self._predict_next_bbox(tracker['bbox'], tracker.get('velocity'))
            if tracker['hits'] >= self.min_hits:
                trks = np.append(trks, [np.append(tracker['bbox'], tracker['id'])], axis=0)
        
        for d in unmatched_detections:
            # Create new tracker
            new_tracker = {
                'id': self.next_id,
                'bbox': dets[d, :4],
                'age': 0,
                'hits': 1,
                'velocity': [0, 0]
            }
            self.trackers.append(new_tracker)
            self.next_id += 1
        
        self.trackers = [t for t in self.trackers if t['age'] <= self.max_age]
        return trks