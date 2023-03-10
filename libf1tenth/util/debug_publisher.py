import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
import numpy as np


class DebugPublisher(Node):
    
    def __init__(self):
        super().__init__('debug_publisher')
        self.get_logger().info("Debug Publisher initialized")
        self.debug_publishers = {}
    
    def publish_debug(self, data, topic, type):
        try:
            if topic not in self.debug_publishers:
                self.debug_publishers[topic] = self.create_publisher(type, topic, 10)
                self.get_logger().info("created publisher for topic: " + topic)
            msg = self._to_msg(data, type)
            self.debug_publishers[topic].publish(msg)
            
        except Exception as e:
            self.get_logger().error("Error publishing to topic: " + topic)
            self.get_logger().error(str(e))
            
    def _to_msg(self, data, type):
        if not (isinstance(data, float) or (isinstance(data, np.ndarray) and data.dtype == float)):
            self.get_logger().error("Input variable must be a float or a numpy array of floats")
            return
        
        if type == Float32:
            msg = Float32()
            msg.data = float(data)
        elif type == Float32MultiArray:
            msg = Float32MultiArray()
            msg.data = list(data.astype(float))
            
        return msg
            
        
def main(args=None):
    try:
        rclpy.init(args=args)
        node = DebugPublisher()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()