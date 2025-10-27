#!/usr/bin/env python3
"""
Quick script to inspect Waymo tfrecord segmentation structure
"""

import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

# Read first tfrecord file
tfrecord_path = "/home/cvlab/hsy/Datasets/waymo_segment/waymo/val/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"

dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

# Get first frame
for data in dataset.take(1):
    frame = open_dataset.Frame()
    frame.ParseFromString(data.numpy())

    print("Frame timestamp:", frame.timestamp_micros)

    # List all available fields
    print("\n=== Available Frame Fields ===")
    available_fields = [field[0].name for field in frame.ListFields()]
    print(f"Fields: {available_fields}")

    print("\n=== Camera Images ===")
    for img in frame.images:
        camera_names = {
            1: 'FRONT',
            2: 'FRONT_LEFT',
            3: 'FRONT_RIGHT',
            4: 'SIDE_LEFT',
            5: 'SIDE_RIGHT'
        }
        print(f"Camera {camera_names.get(img.name, img.name)}")

        # Check if image has segmentation field
        image_fields = [field[0].name for field in img.ListFields()]
        print(f"  Image fields: {image_fields}")

    print("\n=== Camera Labels ===")
    if len(frame.camera_labels) > 0:
        for cam_label in frame.camera_labels:
            camera_names = {
                1: 'FRONT',
                2: 'FRONT_LEFT',
                3: 'FRONT_RIGHT',
                4: 'SIDE_LEFT',
                5: 'SIDE_RIGHT'
            }
            print(f"\nCamera: {camera_names.get(cam_label.name, cam_label.name)}")

            # List all fields in camera label
            label_fields = [field[0].name for field in cam_label.ListFields()]
            print(f"  Camera label fields: {label_fields}")

            # Check for panoptic label
            if hasattr(cam_label, 'panoptic_label') and len(cam_label.panoptic_label) > 0:
                panoptic_label = tf.io.decode_png(cam_label.panoptic_label, channels=1)
                print(f"  Panoptic label shape: {panoptic_label.shape}")
                print(f"  Panoptic label dtype: {panoptic_label.dtype}")
                unique_vals = tf.unique(tf.reshape(panoptic_label, [-1]))[0]
                print(f"  Panoptic label unique values: {len(unique_vals)} unique IDs")
                print(f"  Sample values: {unique_vals[:10].numpy()}")

            if hasattr(cam_label, 'panoptic_label_divisor'):
                print(f"  Panoptic label divisor: {cam_label.panoptic_label_divisor}")

            if hasattr(cam_label, 'instance_id_to_global_id_mapping'):
                mappings = cam_label.instance_id_to_global_id_mapping
                print(f"  Instance ID mappings: {len(mappings)}")
                if len(mappings) > 0:
                    print(f"  Sample mapping: local_instance_id={mappings[0].local_instance_id}, global_instance_id={mappings[0].global_instance_id}, is_tracked={mappings[0].is_tracked}")

            if len(cam_label.labels) > 0:
                print(f"  2D box labels: {len(cam_label.labels)}")
                # Check first label structure
                first_label = cam_label.labels[0]
                label_detail_fields = [field[0].name for field in first_label.ListFields()]
                print(f"  First label fields: {label_detail_fields}")

    break

print("\n" + "="*60)
print("Inspection complete!")
