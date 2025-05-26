import folium
import streamlit as st
import streamlit_folium as st_folium
import networkx as nx
from graph import Graph
from collections import defaultdict
import heapq
import math
import time
import random
from queue import PriorityQueue

# Configure Streamlit page
st.set_page_config(page_title="Cairo Transportation Optimizer", layout="wide")
st.title("🚦 Cairo Smart City Transportation")

# Initialize session state
if 'path_found' not in st.session_state:
    st.session_state.path_found = False
if 'path_details' not in st.session_state:
    st.session_state.path_details = ""
if 'transport_mode' not in st.session_state:
    st.session_state.transport_mode = "All"
if 'traffic_lights' not in st.session_state:
    st.session_state.traffic_lights = {}
if 'current_time' not in st.session_state:
    st.session_state.current_time = 8 * 3600  # Default to 8am
if 'time_of_day' not in st.session_state:
    st.session_state.time_of_day = "morning_peak"
if 'mst_computed' not in st.session_state:
    st.session_state.mst_computed = False
if 'mst_details' not in st.session_state:
    st.session_state.mst_details = ""
if 'mst_edges' not in st.session_state:
    st.session_state.mst_edges = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'manual_control' not in st.session_state:
    st.session_state.manual_control = {}

# Initialize Graph
graph = Graph()

# Load data
try:
    graph.load_nodes()
    graph.load_existing_roads(weight_type='distance')
    graph.load_metro_lines()
    graph.load_bus_routes()
    graph.load_potential_roads(weight_type='distance')
    graph.load_traffic_data()
    graph.load_transport_demand()
    
    # Initialize traffic lights at intersections
    for node in graph.G.nodes():
        if graph.G.degree(node) >= 3:
            st.session_state.traffic_lights[node] = {
                'state': 'green',
                'timer': 0,
                'cycle': 60,
                'green_time': 30,
                'yellow_time': 5,
                'red_time': 25,
                'last_update': time.time(),
                'manual_control': False
            }
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Update traffic light states
current_time = time.time()
time_diff = current_time - st.session_state.last_update
st.session_state.last_update = current_time

for node, light in st.session_state.traffic_lights.items():
    if not light.get('manual_control', False):
        # Update timer based on real time
        light['timer'] = (light['timer'] + time_diff) % light['cycle']
        
        # Update state based on timer
        if light['timer'] < light['green_time']:
            light['state'] = 'green'
        elif light['timer'] < light['green_time'] + light['yellow_time']:
            light['state'] = 'yellow'
        else:
            light['state'] = 'red'
    
    # Calculate time left in current state
    if light['state'] == 'green':
        light['time_left'] = light['green_time'] - light['timer']
    elif light['state'] == 'yellow':
        light['time_left'] = (light['green_time'] + light['yellow_time']) - light['timer']
    else:  # red
        light['time_left'] = light['cycle'] - light['timer']

# ====================== Algorithm Implementations ======================

def dijkstra_route(start, end, time_of_day, transport_mode):
    """Dijkstra's algorithm considering traffic conditions and lights"""
    distances = {node: float('inf') for node in graph.G.nodes()}
    distances[start] = 0
    previous = {node: None for node in graph.G.nodes()}
    queue = [(0, start)]
    
    while queue:
        current_dist, current_node = heapq.heappop(queue)
        
        if current_node == end:
            break
        
        if current_dist > distances[current_node]:
            continue
        
        for neighbor, data in graph.G[current_node].items():
            if transport_mode == "Metro Only" and data.get('type') != 'metro':
                continue
            if transport_mode == "Bus Only" and data.get('type') != 'bus':
                continue
            if transport_mode == "Roads Only" and data.get('type') not in ['existing_road', 'potential_road']:
                continue
            if transport_mode == "Emergency Vehicle" and data.get('type') not in ['existing_road', 'potential_road', 'metro']:
                continue
            
            if data.get('type') == 'metro':
                speed = 80
            elif data.get('type') == 'bus':
                speed = 30
            else:
                speed = 40
            
            base_time = (data.get('distance', 1) / speed) * 3600
            traffic_factor = data.get(time_of_day, 1) / max(1, data.get('capacity', 1))
            
            light_wait = 0
            if neighbor in st.session_state.traffic_lights and transport_mode != "Emergency Vehicle":
                light = st.session_state.traffic_lights[neighbor]
                if light['state'] == 'red':
                    light_wait = light['time_left']
                elif light['state'] == 'yellow':
                    light_wait = light['time_left'] + light['red_time']
            
            segment_time = base_time * (1 + traffic_factor) + light_wait
            distance = current_dist + segment_time
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return path, distances[end] if end in distances else float('inf')

def astar_route(start, end, transport_mode):
    """A* algorithm for emergency vehicle fastest route ignoring traffic lights and congestion"""
    def heuristic(u, v):
        pos_u = graph.G.nodes[u]['pos']
        pos_v = graph.G.nodes[v]['pos']
        return math.sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2) * 50
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in graph.G.nodes()}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.G.nodes()}
    f_score[start] = heuristic(start, end)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[end]
        
        for neighbor, data in graph.G[current].items():
            if transport_mode == "Emergency Vehicle" and data.get('type') not in ['existing_road', 'potential_road', 'metro']:
                continue
            
            speed = 90
            base_time = (data.get('distance', 1) / speed) * 3600
            
            segment_time = base_time
            
            tentative_g_score = g_score[current] + segment_time
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, float('inf')

def dynamic_programming_route(start, end, time_of_day, transport_mode):
    """Dynamic programming (Bellman-Ford) route considering traffic patterns but no real-time updates"""
    distances = {node: float('inf') for node in graph.G.nodes()}
    distances[start] = 0
    previous = {node: None for node in graph.G.nodes()}
    
    for _ in range(len(graph.G.nodes()) - 1):
        updated = False
        for u, v, data in graph.G.edges(data=True):
            if transport_mode == "Metro Only" and data.get('type') != 'metro':
                continue
            if transport_mode == "Bus Only" and data.get('type') != 'bus':
                continue
            if transport_mode == "Roads Only" and data.get('type') not in ['existing_road', 'potential_road']:
                continue
            if transport_mode == "Emergency Vehicle" and data.get('type') not in ['existing_road', 'potential_road', 'metro']:
                continue
            
            if data.get('type') == 'metro':
                speed = 80
            elif data.get('type') == 'bus':
                speed = 30
            else:
                speed = 40
                
            base_time = (data.get('distance', 1) / speed) * 3600
            traffic_factor = data.get(time_of_day, 1) / max(1, data.get('capacity', 1))
            
            light_wait = 0
            if v in st.session_state.traffic_lights and transport_mode != "Emergency Vehicle":
                light = st.session_state.traffic_lights[v]
                light_progress = (st.session_state.current_time + distances[u]) % light['cycle']
                if light_progress > light['green_time']:
                    if light_progress < light['green_time'] + light['yellow_time']:
                        light_wait = (light['green_time'] + light['yellow_time'] - light_progress) * 0.5
                    else:
                        light_wait = light['cycle'] - light_progress
            
            edge_time = base_time * (1 + traffic_factor) + light_wait
            
            if distances[u] + edge_time < distances[v]:
                distances[v] = distances[u] + edge_time
                previous[v] = u
                updated = True
        
        if not updated:
            break
    
    if distances[end] == float('inf'):
        return None, None
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return path, distances[end]

def greedy_best_first_route(start, end, transport_mode):
    """Greedy best-first search: always chooses the neighbor closest to the goal."""
    visited = set()
    previous = {node: None for node in graph.G.nodes()}
    pq = PriorityQueue()
    pq.put((0, start))
    
    def heuristic(u, v):
        pos_u = graph.G.nodes[u]['pos']
        pos_v = graph.G.nodes[v]['pos']
        return math.sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2)

    while not pq.empty():
        _, current = pq.get()
        if current == end:
            break
        if current in visited:
            continue
        visited.add(current)
        for neighbor, data in graph.G[current].items():
            if transport_mode == "Metro Only" and data.get('type') != 'metro':
                continue
            if transport_mode == "Bus Only" and data.get('type') != 'bus':
                continue
            if transport_mode == "Roads Only" and data.get('type') not in ['existing_road', 'potential_road']:
                continue
            if transport_mode == "Emergency Vehicle" and data.get('type') not in ['existing_road', 'potential_road', 'metro']:
                continue
            if neighbor not in visited:
                previous[neighbor] = current
                pq.put((heuristic(neighbor, end), neighbor))
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    if path[0] != start:
        return None, float('inf')
    total_distance = 0
    for i in range(1, len(path)):
        u = path[i-1]
        v = path[i]
        total_distance += graph.G[u][v].get('distance', 0)
    avg_speed = 40
    total_seconds = (total_distance / avg_speed) * 3600 if avg_speed > 0 else 0
    return path, total_seconds

# ====================== UI Components ======================

with st.sidebar:
    st.header("Transportation Options")
    
    st.subheader("Time of Day")
    time_col1, time_col2 = st.columns(2)
    with time_col1:
        if st.button("Morning (7am-10am)"):
            st.session_state.current_time = 8 * 3600
            st.session_state.time_of_day = "morning_peak"
        if st.button("Day (10am-4pm)"):
            st.session_state.current_time = 12 * 3600
            st.session_state.time_of_day = "afternoon"
    with time_col2:
        if st.button("Evening (4pm-7pm)"):
            st.session_state.current_time = 17 * 3600
            st.session_state.time_of_day = "evening_peak"
        if st.button("Night (7pm-7am)"):
            st.session_state.current_time = 22 * 3600
            st.session_state.time_of_day = "night"
    
    st.subheader("Transport Mode")
    transport_mode = st.radio(
        "Select Mode:",
        ["All", "Metro Only", "Bus Only", "Roads Only", "Emergency Vehicle"],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.transport_mode = transport_mode
    
    st.subheader("Traffic Light Controls")
    if st.session_state.traffic_lights:
        selected_light = st.selectbox(
            "Select Traffic Light",
            options=list(st.session_state.traffic_lights.keys()),
            format_func=lambda x: f"{graph.G.nodes[x]['name']} ({x})"
        )
        
        if selected_light:
            light = st.session_state.traffic_lights[selected_light]
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Force Green"):
                    light['state'] = 'green'
                    light['timer'] = 0
                    light['time_left'] = light['green_time']
                    light['manual_control'] = True
            with col2:
                if st.button("Force Red"):
                    light['state'] = 'red'
                    light['timer'] = light['green_time'] + light['yellow_time']
                    light['time_left'] = light['red_time']
                    light['manual_control'] = True
            with col3:
                if st.button("Auto Mode"):
                    light['manual_control'] = False
                    light['timer'] = 0
            
            st.markdown(f"""
            **Current State:** {light['state'].title()}  
            **Time Left:** {int(light['time_left'])}s  
            **Cycle:** {light['cycle']}s  
            **Mode:** {'Manual' if light['manual_control'] else 'Automatic'}
            """)
    
    st.subheader("Routing Strategy")
    algorithm = st.radio(
        "Select Strategy:",
        [
            "Dijkstra (Shortest Path)", 
            "A* (Emergency Fastest)", 
            "Dynamic Programming (Bellman-Ford)",
            "Greedy Best-First",
            "Kruskal's MST"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    if algorithm != "Kruskal's MST":
        st.subheader("Route Planning")
        node_name_to_id = {f"{attrs['name']} ({node})": node for node, attrs in graph.get_nodes()}
        start_node_name = st.selectbox(
            "Start Location",
            sorted(node_name_to_id.keys()),
            index=list(node_name_to_id.keys()).index("Cairo International Airport,Airport (F1)") if "Cairo International Airport,Airport (F1)" in node_name_to_id else 0
        )
        end_node_name = st.selectbox(
            "End Location",
            sorted(node_name_to_id.keys()),
            index=list(node_name_to_id.keys()).index("Egyptian Museum (F5)") if "Egyptian Museum (F5)" in node_name_to_id else 1
        )
        start_node = node_name_to_id[start_node_name]
        end_node = node_name_to_id[end_node_name]
    else:
        st.subheader("MST Parameters")
        weight_attr = st.radio(
            "Select Weight:",
            ["Distance", "Cost"],
            index=0,
            label_visibility="collapsed"
        )
    
    if st.button("Find Optimal Route"):
        find_path = True
    else:
        find_path = False

# ====================== Map Visualization ======================

m = folium.Map(
    location=[30.0444, 31.2357],
    zoom_start=12,
    tiles='cartodbpositron',
    min_lat=29.8,
    max_lat=30.2,
    min_lon=30.9,
    max_lon=31.5,
    zoom_control=False
)

line_colors = {
    'metro': '#FF0000',
    'bus': '#00AA00',
    'road': '#1E90FF',
    'mst': '#800080'  # Purple for MST
}

show_lines = {
    'metro': transport_mode in ["All", "Metro Only", "Emergency Vehicle"],
    'bus': transport_mode in ["All", "Bus Only"],
    'road': transport_mode in ["All", "Roads Only", "Emergency Vehicle"]
}

for u, v, attrs in graph.get_edges():
    if u not in graph.G.nodes or v not in graph.G.nodes:
        continue
    
    edge_type = attrs.get('type', '')
    coords = [graph.G.nodes[u]['pos'], graph.G.nodes[v]['pos']]
    
    if edge_type == 'metro' and show_lines['metro'] and (transport_mode != "Emergency Vehicle" or transport_mode == "All"):
        folium.PolyLine(
            locations=coords,
            color=line_colors['metro'],
            weight=4,
            opacity=0.8,
            popup=f"Metro Line: {graph.G.nodes[u]['name']} to {graph.G.nodes[v]['name']}"
        ).add_to(m)
    elif edge_type == 'bus' and show_lines['bus']:
        folium.PolyLine(
            locations=coords,
            color=line_colors['bus'],
            weight=3,
            dash_array='5, 3',
            opacity=0.8,
            popup=f"Bus Route: {graph.G.nodes[u]['name']} to {graph.G.nodes[v]['name']}"
        ).add_to(m)
    elif edge_type in ['existing_road', 'potential_road'] and show_lines['road']:
        folium.PolyLine(
            locations=coords,
            color=line_colors['road'],
            weight=3,
            opacity=0.8,
            popup=f"Road: {graph.G.nodes[u]['name']} to {graph.G.nodes[v]['name']}"
        ).add_to(m)

if st.session_state.mst_computed and st.session_state.mst_edges and algorithm == "Kruskal's MST":
    for u, v, attrs in st.session_state.mst_edges:
        coords = [graph.G.nodes[u]['pos'], graph.G.nodes[v]['pos']]
        weight = attrs.get('distance', 'N/A') if weight_attr == "Distance" else attrs.get('cost', 'N/A')
        folium.PolyLine(
            locations=coords,
            color=line_colors['mst'],
            weight=5,
            opacity=0.9,
            popup=f"MST Edge: {graph.G.nodes[u]['name']} to {graph.G.nodes[v]['name']} ({weight_attr}: {weight})"
        ).add_to(m)

for node, attrs in graph.G.nodes(data=True):
    lat, lon = attrs['pos']
    
    popup_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 250px;">
        <h4 style="margin-bottom: 5px;">{attrs['name']}</h4>
        <p style="margin: 2px 0;"><strong>Node ID:</strong> {node}</p>
    """
    
    if node in st.session_state.traffic_lights:
        light = st.session_state.traffic_lights[node]
        popup_content += f"""
        <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 8px;">
            <h5 style="margin-bottom: 5px;">Traffic Light</h5>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="width: 20px; height: 50px; background: #333; border-radius: 4px;
                    display: flex; flex-direction: column; justify-content: space-around; padding: 5px 0;">
                    <div style="width: 16px; height: 16px; margin: 0 auto;
                        background: {'#f00' if light['state'] == 'red' else '#333'};
                        border-radius: 50%; border: 1px solid #666;"></div>
                    <div style="width: 16px; height: 16px; margin: 0 auto;
                        background: {'#ff0' if light['state'] == 'yellow' else '#333'};
                        border-radius: 50%; border: 1px solid #666;"></div>
                    <div style="width: 16px; height: 16px; margin: 0 auto;
                        background: {'#0f0' if light['state'] == 'green' else '#333'};
                        border-radius: 50%; border: 1px solid #666;"></div>
                </div>
                <div>
                    <p style="margin: 3px 0;"><strong>State:</strong> {light['state'].title()}</p>
                    <p style="margin: 3px 0;"><strong>Time left:</strong> {int(light['time_left'])}s</p>
                    <p style="margin: 3px 0;"><strong>Cycle:</strong> {light['cycle']}s</p>
                </div>
            </div>
        </div>
        """
    
    popup_content += "</div>"
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_content, max_width=300),
        icon=folium.Icon(color='blue' if node in st.session_state.traffic_lights else 'gray', 
                        icon='info-sign', prefix='fa')
    ).add_to(m)

# ====================== Pathfinding and MST Computation ======================

if find_path:
    time_of_day = st.session_state.time_of_day
    transport_mode = st.session_state.transport_mode
    
    # Reset previous results
    st.session_state.path_found = False
    st.session_state.path_details = ""
    if algorithm != "Kruskal's MST":
        st.session_state.mst_computed = False
        st.session_state.mst_edges = []
        st.session_state.mst_details = ""
    
    if algorithm == "Kruskal's MST":
        start_time = time.time()
        criticals = [node for node, attrs in graph.G.nodes(data=True) if attrs.get('type') in ['Medical', 'Government']]
        mst_edges = graph.kruskal_mst(weight_attr=weight_attr.lower(), must_have_degree2=criticals)
        
        if mst_edges:
            st.session_state.mst_computed = True
            st.session_state.mst_edges = mst_edges
            
            total_weight = sum(attrs.get(weight_attr.lower(), 0) for _, _, attrs in mst_edges)
            weight_unit = "km" if weight_attr == "Distance" else "units"
            
            mst_details = f"""
**Minimum Spanning Tree Summary**

🌐 **Algorithm:** Kruskal's  
📏 **Weight Attribute:** {weight_attr}  
🔗 **Total Edges:** {len(mst_edges)}  
⚖️ **Total {weight_attr}:** {total_weight:.1f} {weight_unit}  
⏱️ **Computation Time:** {(time.time() - start_time) * 1000:.2f} ms  

**Edges in MST:**
"""
            for u, v, attrs in mst_edges:
                weight = attrs.get(weight_attr.lower(), 'N/A')
                mst_details += f"- {graph.G.nodes[u]['name']} to {graph.G.nodes[v]['name']} ({weight_attr}: {weight})\n"
            
            st.session_state.mst_details = mst_details
        else:
            st.sidebar.error("Failed to compute MST")
    else:
        valid_edge_types = []
        if transport_mode == "Metro Only":
            valid_edge_types = ['metro']
        elif transport_mode == "Bus Only":
            valid_edge_types = ['bus']
        elif transport_mode == "Roads Only":
            valid_edge_types = ['existing_road', 'potential_road']
        elif transport_mode == "Emergency Vehicle":
            valid_edge_types = ['existing_road', 'potential_road', 'metro']
        else:
            valid_edge_types = ['metro', 'bus', 'existing_road', 'potential_road']
        
        temp_graph = nx.Graph()
        for node, attrs in graph.G.nodes(data=True):
            temp_graph.add_node(node, **attrs)
        
        for u, v, attrs in graph.G.edges(data=True):
            if attrs.get('type') in valid_edge_types:
                temp_graph.add_edge(u, v, **attrs)
        
        if not nx.has_path(temp_graph, start_node, end_node):
            st.sidebar.error("No path exists with current selections")
        else:
            start_time = time.time()
            
            if algorithm == "Dijkstra (Shortest Path)":
                path, total_seconds = dijkstra_route(start_node, end_node, time_of_day, transport_mode)
            elif algorithm == "A* (Emergency Fastest)":
                path, total_seconds = astar_route(start_node, end_node, transport_mode)
            elif algorithm == "Dynamic Programming (Bellman-Ford)":
                path, total_seconds = dynamic_programming_route(start_node, end_node, time_of_day, transport_mode)
            elif algorithm == "Greedy Best-First":
                path, total_seconds = greedy_best_first_route(start_node, end_node, transport_mode)
            
            computation_time = (time.time() - start_time) * 1000
            
            if path:
                st.session_state.path_found = True
                
                path_coords = [graph.G.nodes[node]['pos'] for node in path]
                folium.PolyLine(
                    locations=path_coords,
                    color='#FF00FF',
                    weight=6,
                    opacity=1,
                    popup=f"Optimal Path ({algorithm})"
                ).add_to(m)
                
                folium.Marker(
                    location=graph.G.nodes[path[0]]['pos'],
                    icon=folium.Icon(color='green', icon='circle'),
                    popup=f"Start: {graph.G.nodes[path[0]]['name']}"
                ).add_to(m)
                
                folium.Marker(
                    location=graph.G.nodes[path[-1]]['pos'],
                    icon=folium.Icon(color='red', icon='flag'),
                    popup=f"End: {graph.G.nodes[path[-1]]['name']}"
                ).add_to(m)
                
                total_distance = 0
                for i in range(1, len(path)):
                    u = path[i-1]
                    v = path[i]
                    total_distance += graph.G[u][v].get('distance', 0)
                
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                
                route_details = f"""
**Route Summary - {algorithm.split(' (')[0]}**

📍 **From:** {graph.G.nodes[path[0]]['name']}  
🏁 **To:** {graph.G.nodes[path[-1]]['name']}  
🚦 **Transport Mode:** {transport_mode}  
⏰ **Time of Day:** {time_of_day.replace('_', ' ').title()}  
📏 **Total Distance:** {total_distance:.1f} km  
⏱️ **Estimated Time:** {hours}h {minutes}m {seconds}s  
⚡ **Computation Time:** {computation_time:.2f} ms  
"""
                
                st.session_state.path_details = route_details

# ====================== Main Display ======================

col1, col2 = st.columns([3, 1])

with col1:
    st_folium.st_folium(
        m,
        width=1200,
        height=800,
        returned_objects=[]
    )

with col2:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#212529;">Route Optimization</h3>
        <p style="color:#212529;">Real-time traffic-aware routing for Cairo</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.path_found and algorithm != "Kruskal's MST":
        st.markdown(st.session_state.path_details)
    
    if st.session_state.mst_computed and algorithm == "Kruskal's MST":
        st.markdown(st.session_state.mst_details)
    
    st.markdown("""
    **Routing Strategies**

    - **Dijkstra:** Conservative route considering all traffic conditions and lights
    - **A*:** Fastest possible route ignoring traffic lights and congestion (90 km/h)
    - **Dynamic Programming:** Considers typical traffic patterns for the selected time (Bellman-Ford)
    - **Greedy Best-First:** Always chooses the next node closest to the destination (may not be optimal)
    - **Kruskal's MST:** Computes a minimum spanning tree of road network, ensuring critical facilities (Medical, Government) have degree >= 2
    """)
    
    st.markdown("""
    **Traffic Light Priority**

    - Emergency vehicles always get green light
    - Other vehicles obey normal traffic light cycles
    - Green: 30s, Yellow: 5s, Red: 25s
    """)