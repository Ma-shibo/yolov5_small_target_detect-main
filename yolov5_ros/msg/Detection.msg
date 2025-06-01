# Single detection message
std_msgs/Header header
string class_name     # detected object class name
int32 class_id       # detected object class id
float64 confidence   # detection confidence score
BoundingBox bbox     # bounding box coordinates
geometry_msgs/Point center  # center point of detection