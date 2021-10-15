# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import numpy as np

from application_util import visualization
import torch
from torchvision.ops import nms
from fm_tracker.multitracker import JDETracker

def gather_sequence_info(detections):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())
    seq_info = {
        "detections": detections,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }
    return seq_info


def create_detections(detection_mat, frame_idx):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_mask = detection_mat[mask]
    dets = np.array([row[2:7] for row in detection_mask])
    feats = np.array([row[7:] for row in detection_mask])
    return dets, feats


def run(detections, **kwargs):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    min_confidence = kwargs['min_confidence']
    nms_max_overlap = kwargs['nms_max_overlap']
    max_cosine_distance = kwargs['max_cosine_distance']
    display=False
    seq_info = gather_sequence_info(detections)
    tracker = JDETracker(min_confidence, max_cosine_distance, 30)
    results = []

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)

        # Load image and generate detections. Default is 0-indexed det files. Jiwang's ones are 1-indexed.
        dets, feats = create_detections(seq_info["detections"], frame_idx)
        if len(dets) > 0:
            dets[:, 2:4] += dets[:, 0:2]
            if nms_max_overlap >= 0:
                nms_keep = nms(torch.from_numpy(dets[:, :4]),
                                torch.from_numpy(dets[:, 4]),
                                iou_threshold=nms_max_overlap).numpy().astype(np.int64)
                dets = dets[nms_keep]
                feats = feats[nms_keep]

        # Update tracker.
        online_targets = tracker.update(dets, feats, frame_idx)
        # Store results.
        for t in online_targets:
            tlwh = t.det_tlwh
            tid = t.track_id
            score = t.score
            feature = t.features[-1]
            out_list = [frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score]
            out_list.extend(feature.tolist())
            results.append(out_list)
    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    return results
