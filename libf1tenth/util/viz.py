import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


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
    
    return marker

def to_waypoints_viz_msg(waypoints, waypoint_visualize_message=None, color='r'):
    '''
    Creates a MarkerArray message for visualization of waypoints
    
    Args:
    - waypoints: Waypoints object
    - waypoint_visualize_message: MarkerArray message to update
    - color: color of the waypoints
    
    Returns:
    - waypoint_visualize_message: MarkerArray message
    '''
    if waypoint_visualize_message is None:
        waypoint_visualize_message = MarkerArray()
    for i in range(0, len(waypoints), 20):
        marker = Marker()
        # id is a random number between 0 and 2^32
        marker.id = np.random.randint(0, 2**15)
        marker.header.frame_id = "/map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        if color == 'r':
            marker.color.r = 2*((waypoints.velocity[i]) / 15.0)# * 255.
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif color == 'g':
            marker.color.r = 0.0
            marker.color.g = 2*((waypoints.velocity[i]) / 15.0)# * 255.
            marker.color.b = 0.0
        elif color == 'b':
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 2*((waypoints.velocity[i]) / 15.0)# * 255.
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