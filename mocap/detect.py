# import pyrealsense2 as rs
# import numpy as np
# import cv2

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from grip import GripPipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# drawing figures
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_tips = np.array([])
y_tips = np.array([])
z_tips = np.array([])
x_traj = np.array([])
y_traj = np.array([])
z_traj = np.array([])
line, = ax.plot(x_traj, y_traj, z_traj, 'g-')
tips = ax.scatter(x_tips, y_tips, z_tips, c='r', marker='o')
ax.set_xlim(0.0, 0.4)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.show(block=False)
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
process_pipeline = GripPipeline()
# Streaming loop
#EMA on last depth to reduce effects of glitches
last_depth_top, last_depth_bot = None, None
lower_pt, upper_pt = None, None
alpha = 0.5
try:
    c = 0
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))
        image = color_image
        process_pipeline.process(image)
        hulls = process_pipeline.convex_hulls_output
        # print(hulls)
        # print(len(hulls))
        # creating hull mask at where the reflective tapes are
        # the average depth value of the reflective tapes are the "depth" for the limb
        canvas = np.zeros_like(depth_image_3d)
        centers = []
        for hull in hulls:
            cv2.drawContours(image, [hull], 0, (255, 0 , 0), 3)
            cv2.fillPoly(canvas,[hull],(255,255,255))
            masked = cv2.bitwise_and(depth_image_3d, canvas)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
                cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
            # print(image.shape)
            # print(depth_image_3d.shape)
        # for pt in centers:
        #     print(depth_image[pt[1], pt[0]])
        x_tips, y_tips, z_tips = [], [], []
        for cx, cy in centers:
            # avg_d = 0
            # for i in range(cx-5, cx+5):
            #     for j in range(cy-5, cy+5):
            #         avg_d += aligned_depth_frame.get_distance(i, j)
            depth = aligned_depth_frame.get_distance(cx, cy)
            if cy <= color_image.shape[0]//3:
                if last_depth_top is None:
                    last_depth_top = depth
                depth = alpha*last_depth_top + (1 - alpha)*depth
                # print(depth, last_depth_top)
                # block sudden jumps
                if abs(depth - last_depth_top) > 0.01:
                    depth = last_depth_top
                last_depth_top = depth
            else:
                if last_depth_bot is None:
                    last_depth_bot = depth
                depth = alpha*last_depth_bot + (1 - alpha)*depth
                if abs(depth - last_depth_bot) > 0.01:
                    depth = last_depth_bot
                last_depth_bot = depth
            # depth = aligned_depth_frame.get_distance(cx, cy)
            # depth = d/1000
            depth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [cx, cy], depth)
            msg = "%.2lf, %.2lf, %.2lf\n" % (depth_point[0], depth_point[1], depth_point[2])
            
            # x_tips.append(depth_point[0])
            # y_tips.append(depth_point[1])
            # z_tips.append(depth_point[2])
            R = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]])
            x_tips.append(depth_point[2])
            y_tips.append(-depth_point[0])
            z_tips.append(-depth_point[1])
            if cy >= color_image.shape[0]//4:
                # np.append(x_traj, depth_point[0])
                # np.append(y_traj, depth_point[1])
                # np.append(z_traj, depth_point[2])
                if c > 150:
                    x_traj = np.append(x_traj, depth_point[2])
                    y_traj = np.append(y_traj, -depth_point[0])
                    z_traj = np.append(z_traj, -depth_point[1])
                    c = 151
                lower_pt = depth_point
                cv2.putText(image, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(image, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                upper_pt = depth_point
        # calculate the bending angle of the two points
        # assuming the angles are in the correct coord frame
        if lower_pt is not None and upper_pt is not None:
            pt_diff = lower_pt - upper_pt
            # x bending angle x-z plane
            x_bend_ang = np.arctan2(pt_diff[2], pt_diff[0])
            # y bending angle y-z plane
            y_bend_ang = np.arctan2(pt_diff[2], pt_diff[1])
            # vec2 = np.array([0, 0, 1])
            # cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            # theta = np.arccos(cos_theta)
            cv2.putText(image, "Bending Angle x: %.2lf" % (x_bend_ang*180/np.pi), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Bending Angle x: %.2lf" % (y_bend_ang*180/np.pi), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            print("Bending Angle x: %.2lf" % (x_bend_ang*180/np.pi))
            print("Bending Angle y: %.2lf" % (y_bend_ang*180/np.pi))
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', image)
        key = cv2.waitKey(1)

        # Update plot data
        # print(line)
        line.set_data(x_traj, y_traj)
        line.set_3d_properties(z_traj)

        tips._offsets3d = (x_tips, y_tips, z_tips)

        ax.draw_artist(ax.patch)
        ax.draw_artist(line)
        # ax.draw_artist(tips)
        fig.canvas.draw_idle()
        fig.show()
        fig.canvas.flush_events()
        c += 1
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()