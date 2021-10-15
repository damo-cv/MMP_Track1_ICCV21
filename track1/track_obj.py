class TrackData():
    def __init__(self):

        self.trail_data = None    # common dictionary, which is mutable
        self.camera_id = None
        self.trail_num = 0
        self.file_name = None
        self.feat_list = []
        self.mean_feat = None
        self.height = None
        self.map_time_stamp = []
        self.target_image_path = None
        self.record_id = None

    def set_record_id(self, record_id):
        """Set record id."""

        self.record_id = record_id

    def get_record_id(self):
        """Get record id."""

        return self.record_id

    def set_image_path(self, image_path):
        """Set image path."""

        self.target_image_path = image_path

    def get_image_path(self):
        """Get image path."""

        return self.target_image_path

    def get_map_time_stamp(self):
        """Get map time stamp."""

        return list(set(self.map_time_stamp))

    def add_map_time_stamp(self, map_time_stamp):
        """Add map time stamp."""

        self.map_time_stamp.append(map_time_stamp)

    def get_trail_stamp_list(self):
        """Get trail stamp list."""

        return list(self.trail_data.keys())

    def set_stamp_data(self, time_stamp, new_stamp_data):
        """Set stamp data."""

        self.trail_data[time_stamp] = new_stamp_data

    def get_stamp_data(self, time_stamp):
        """Get stamp data."""

        try:
            stamp_data = self.trail_data[time_stamp]
        except KeyError:
            stamp_data = None
        return stamp_data

    def set_height(self, height):
        """Set height."""

        self.height = height

    def get_height(self):
        """Get height."""

        return self.height

    def set_trail(self, trail_data):
        """Set trail data."""

        self.trail_data = trail_data

    def get_trail(self):
        """Get trail data."""

        return self.trail_data

    def set_camera_id(self, camera_id):
        """Set camera id."""

        self.camera_id = camera_id

    def get_camera_id(self):
        """Get camera id."""

        return self.camera_id

    def set_trail_num(self, trail_num):
        """Set trail num."""

        self.trail_num = trail_num

    def get_trail_num(self):
        """Get trail num."""

        return self.trail_num

    def set_file_name(self, file_name):
        """Set file_name."""

        self.file_name = file_name

    def get_file_name(self):
        """Get file_name."""

        return self.file_name

    def put_feat_list(self, feat):
        """Put feat list."""

        self.feat_list.append(feat)

    def get_feat_list(self):
        """Get feat list."""

        return self.feat_list

    def set_mean_feat(self, mean_feat):
        """Put mean_feat."""

        self.mean_feat = mean_feat

    def get_mean_feat(self):
        """Get mean_feat."""

        return self.mean_feat


class FrameData():
    """To save info for each stamp of a track."""

    def __init__(self):
        self.bbox = None
        self.head = None
        self.feat = None
        self.predict_flag = 0
        self.world = None
        self.flag = None
        self.temp_world_dict = {}
        self.camera_bbox = {}

    def set_camera_bbox(self, camera, bbox):
        """Set camera bbox."""

        self.camera_bbox[camera] = bbox

    def get_camera_bbox(self, camera):
        """Get camera bbox."""

        try:
            bbox = self.camera_bbox[camera]
            return bbox
        except KeyError:
            return None

    def set_bbox(self, bbox):
        """Set bbox."""

        self.bbox = bbox

    def get_bbox(self):
        """Get bbox."""

        return self.bbox

    def set_head(self, head):
        """Set bbox."""

        self.head = head

    def get_head(self):
        """Get bbox."""

        return self.head

    def set_feat(self, feat):
        self.feat = feat

    def get_feat(self):
        return self.feat

    def set_flag(self, flag):
        self.flag = flag

    def get_flag(self):
        return self.flag

    def set_predict_flag(self, flag):
        """Set bbox."""

        self.predict_flag = flag


    def set_world(self, world):
        """Set world."""

        self.world = world

    def get_world(self):
        """Get world."""

        return self.world

    def put_temp_world_dict(self, record_id_a, record_id_b, world):
        """Get world temp."""

        self.temp_world_dict[(record_id_a, record_id_b)] = world

    def get_temp_world_dict(self):
        """Get world temp."""

        return self.temp_world_dict



class FrameResultData(FrameData):
    """To save info for each stamp of global id."""

    def __init__(self):
        super(FrameResultData, self).__init__()
        self.camera_bbox = {}
        self.camera_head = {}
        self.camera_feat = {}
        self.footpoint = None

    def set_camera_bbox(self, camera, bbox):
        """Set camera bbox."""

        self.camera_bbox[camera] = bbox

    def get_camera_bbox(self, camera):
        """Get camera bbox."""

        try:
            bbox = self.camera_bbox[camera]
            return bbox
        except KeyError:
            return None

    def set_camera_head(self, camera, head):
        """Set camera head."""

        self.camera_head[camera] = head

    def get_camera_head(self, camera):
        """Get camera head."""

        try:
            head = self.camera_head[camera]
            return head
        except KeyError:
            return None

    def set_footpoint(self, footpoint):
        """Set room_map."""

        self.footpoint = footpoint

    def get_footpoint(self):
        """Get room_map."""

        return self.footpoint

    def set_camera_feat(self, camera, feat):
        self.camera_feat[camera] = feat
  
    def get_camera_feat(self, camera):
        try:
            feat = self.camera_feat[camera]
            return feat
        except:
            return None
    
