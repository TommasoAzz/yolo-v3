from __future__ import annotations


class TrainingConfig:
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_NUM_WORKERS = 0
    DEFAULT_EPOCH = 1
    DEFAULT_CONF_THRESHOLD = 0.5
    DEFAULT_NMS_IOU_THRESHOLD = 0.5
    DEFAULT_MAP_IOU_THRESHOLD = 0.5
    DEFAULT_SAVE_MODEL = True
    DEFAULT_CHECKPOINT_FILE = "checkpoint.pth.tar"
    DEFAULT_LOAD_MODEL = False
    DEFAULT_TRAINING_SET_SIZE = 1
    DEFAULT_VALIDATION_SET_SIZE = 1
    DEFAULT_TEST_SET_SIZE = 1
    DEFAULT_LEARNING_RATE = 1e-2
    DEFAULT_WEIGHT_DECAY = 1e-2
    DEFAULT_EARLY_STOPPING = False
    DEFAULT_EARLY_STOPPING_BAD_EPOCHS = 3

    def __init__(self, **kwargs):
        self.batch_size = int(kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE))
        self.num_workers = int(kwargs.get("num_workers", self.DEFAULT_NUM_WORKERS))
        self.epoch = int(kwargs.get("epoch", self.DEFAULT_EPOCH))
        self.conf_threshold = float(kwargs.get("conf_threshold", self.DEFAULT_CONF_THRESHOLD))
        self.nms_iou_threshold = float(kwargs.get("nms_iou_threshold", self.DEFAULT_NMS_IOU_THRESHOLD))
        self.map_iou_threshold = float(kwargs.get("map_iou_threshold", self.DEFAULT_MAP_IOU_THRESHOLD))
        self.save_model = bool(kwargs.get("save_model", self.DEFAULT_SAVE_MODEL))
        self.checkpoint_file = str(kwargs.get("checkpoint_file", self.DEFAULT_CHECKPOINT_FILE))
        self.load_model = bool(kwargs.get("load_model", self.DEFAULT_LOAD_MODEL))
        self.training_set_size = int(kwargs.get("training_set_size", self.DEFAULT_TRAINING_SET_SIZE))
        self.validation_set_size = int(kwargs.get("validation_set_size", self.DEFAULT_VALIDATION_SET_SIZE))
        self.test_set_size = int(kwargs.get("test_set_size", self.DEFAULT_TEST_SET_SIZE))
        self.learning_rate = float(kwargs.get("learning_rate", self.DEFAULT_LEARNING_RATE))
        self.weight_decay = float(kwargs.get("weight_decay", self.DEFAULT_WEIGHT_DECAY))
        self.early_stopping = bool(kwargs.get("early_stopping", self.DEFAULT_EARLY_STOPPING))
        self.early_stopping_bad_epochs = int(kwargs.get("early_stopping_bad_epochs",
                                                        self.DEFAULT_EARLY_STOPPING_BAD_EPOCHS))

    @staticmethod
    def from_dict(config: dict) -> TrainingConfig | None:
        cfg_dictionary = config.get("training")

        if cfg_dictionary is None:
            print("FAILURE: configuration file is not well formed.")
            return None

        return TrainingConfig(**cfg_dictionary)

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "conf_threshold": self.conf_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "map_iou_threshold": self.map_iou_threshold,
            "training_set_size": self.training_set_size,
            "validation_set_size": self.validation_set_size,
            "test_set_size": self.test_set_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping": self.early_stopping,
            "early_stopping_bad_epochs": self.early_stopping_bad_epochs
        }

    def __repr__(self):
        return f"""TrainingConfig:
              batch_size: {self.batch_size}
              num_workers: {self.num_workers}
              epoch: {self.epoch}
              conf_threshold: {self.conf_threshold}
              nms_iou_threshold: {self.nms_iou_threshold}
              map_iou_threshold: {self.map_iou_threshold}
              save_model: {self.save_model}
              checkpoint_file: {self.checkpoint_file}
              load_model: {self.load_model}
              training_set_size: {self.training_set_size}
              validation_set_size: {self.validation_set_size}
              test_set_size: {self.test_set_size}
              learning_rate: {self.learning_rate}
              weight_decay: {self.weight_decay}
              early_stopping: {self.early_stopping}
              early_stopping_bad_epochs: {self.early_stopping_bad_epochs}"""
