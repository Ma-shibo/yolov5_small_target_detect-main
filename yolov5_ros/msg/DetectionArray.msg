# Array of detections message (optimized - no image transmission)
std_msgs/Header header
Detection[] detections         # array of detections
int32 detection_count         # number of detections
float64 inference_time        # inference time in seconds
int32 image_width            # original image width
int32 image_height           # original image height