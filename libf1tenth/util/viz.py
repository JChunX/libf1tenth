import numpy as np
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def make_delete_all_markers_msg():
    '''
    Creates a MarkerArray message with a single marker that deletes all markers
    
    Returns:
    - delete_all_msg: MarkerArray message
    '''
    delete_all_msg = MarkerArray()
    marker = Marker()
    marker.action = marker.DELETEALL
    delete_all_msg.markers.append(marker)
    return delete_all_msg

def to_waypoint_target_viz_msg(waypoint_position):
    '''
    Creates a Marker message for visualization of waypoint target
    
    Args:
    - waypoint_position: ndarray of shape (2,)
    
    Returns:
    - waypoint_target_visualize_message: Marker message
    '''
    marker = Marker()
    marker.header.frame_id = "/map"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.r = 0.
    marker.color.g = 255.
    marker.color.b = 0.
    marker.color.a = 255.
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = float(waypoint_position[0])
    marker.pose.position.y =  float(waypoint_position[1])
    marker.pose.position.z = 0.   
    marker.lifetime = Duration(seconds=0.1).to_msg()
    
    return marker

def to_waypoints_viz_msg(waypoints, waypoint_visualize_message=None, color='r', type='sphere'):
    '''
    Creates a MarkerArray message for visualization of waypoints
    
    Args:
    - waypoints: Waypoints object
    - color: color of the waypoints
    
    Returns:
    - waypoint_visualize_message: MarkerArray message
    '''
    
    if waypoint_visualize_message is None:
        waypoint_visualize_message = MarkerArray()
    for i in range(0, len(waypoints)):
        marker = Marker()
        marker.id = i
        marker.header.frame_id = "/map"
        marker.action = marker.ADD
        if type == 'sphere':
            marker.type = marker.SPHERE
        elif type == 'line_strip':
            marker.type = marker.LINE_STRIP
            
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        if color == 'r':
            marker.color.r = 1.0
        elif color == 'g':
            marker.color.g = 1.0
        elif color == 'b':
            marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(waypoints.x[i])
        marker.pose.position.y =  float(waypoints.y[i])
        marker.pose.position.z = 0.     
        waypoint_visualize_message.markers.append(marker)
        
    return waypoint_visualize_message

def to_graph_viz_msg(graph, frame_id):
    '''
    Creates a MarkerArray message for visualization of graph
    
    Args:
    - graph: Graph object
    - frame_id: string, frame id of the graph
    
    Returns:
    - graph_visualize_message: MarkerArray message
    '''
    graph_visualize_message = MarkerArray()
    # Add nodes
    for i in range(len(graph.nodes)):
        marker = Marker()
        marker.id = i
        marker.header.frame_id = frame_id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 255.
        marker.color.g = 0.
        marker.color.b = 0.
        marker.color.a = 255.
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(graph.nodes[i].position[0])
        marker.pose.position.y =  float(graph.nodes[i].position[1])
        marker.pose.position.z = 0.     
        graph_visualize_message.markers.append(marker)
    # add edges
    for i in range(len(graph.edges)):
        marker = Marker()
        marker.id = i + len(graph.nodes)
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 255.
        marker.color.g = 0.
        marker.color.b = 0.
        marker.color.a = 255.
        marker.pose.orientation.w = 1.0
        
        marker_point_a = Point()
        node_a_position = graph.nodes[graph.edges[i][0]].position
        marker_point_a.x = float(node_a_position[0])
        marker_point_a.y = float(node_a_position[1])
        
        marker_point_b = Point()
        node_b_position = graph.nodes[graph.edges[i][1]].position
        marker_point_b.x = float(node_b_position[0])
        marker_point_b.y = float(node_b_position[1])

        marker.points.append(marker_point_a)
        marker.points.append(marker_point_b)
        
        graph_visualize_message.markers.append(marker)
        
    return graph_visualize_message