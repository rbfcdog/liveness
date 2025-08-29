import mediapipe as mp 


class FaceDetector:
    
    def __init__(self, min_confidence: float = 0.5):
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)

    def __call__(self, image):
        face_obj = {}
        # fazer alinhamento e normalizacao de cor dps (lbp tb), ver se impacta tanto assim, senao so adiciona camadas iniciais e fds
        results = self.detector.process(image)

        if not results.detections:
            return None
        
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

        face_obj['face_arr'] = image[y:y+h_box, x:x+w_box]
        face_obj['bbox'] = (x, y, x+w_box, y+h_box)

        return face_obj 

        

    def close(self):
        self.detector.close()
