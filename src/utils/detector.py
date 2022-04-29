import cv2
from config import get_vrd_cfg

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

cv2.setNumThreads(0)

class Detector(object):
    """
        object bounding-box detector that uses detectron2 library
    """
    def __init__(self):
        self.cfg=get_vrd_cfg()

        #load model config and pretrained model
        self.predictor = DefaultPredictor(self.cfg)
    
    def onImage(self, image_path):
        image = cv2.imread(image_path)
        predictions = self.predictor(image)
        
        viz = Visualizer(
            image[:,:,::-1],
            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode=ColorMode.IMAGE_BW
        )

        output = viz.draw_instance_predictions(predictions["instances".to("cpu")])

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        #cv2.imwrite(os.path.join(ROOT_DIR, "tmp/img.jpg"), output.get_image()[:,:,::-1])
        cv2.waitKey(0)

