import networkx as nx
import math

class UnionFind:
    def __init__(self, elements):
        # Initialize parent pointer and rank
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
        return True

class Graph:
    def __init__(self):
        self.G = nx.Graph()

    def load_nodes(self):
        try:
            # Hardcoded neighborhoods
            neighborhoods = [
                {'ID': '1', 'Name': 'Maadi', 'Population': 250000, 'Type': 'Residential', 'X': 31.25, 'Y': 29.96},
                {'ID': '2', 'Name': 'Nasr City', 'Population': 500000, 'Type': 'Mixed', 'X': 31.34, 'Y': 30.06},
                {'ID': '3', 'Name': 'Downtown', 'Population': 100000, 'Type': 'Business', 'X': 31.24, 'Y': 30.04},
                {'ID': '4', 'Name': 'New Cairo', 'Population': 300000, 'Type': 'Residential', 'X': 31.47, 'Y': 30.03},
                {'ID': '5', 'Name': 'Heliopolis', 'Population': 200000, 'Type': 'Mixed', 'X': 31.32, 'Y': 30.09},
                {'ID': '6', 'Name': 'Zamalek', 'Population': 50000, 'Type': 'Residential', 'X': 31.22, 'Y': 30.06},
                {'ID': '7', 'Name': '6th October', 'Population': 400000, 'Type': 'Mixed', 'X': 30.98, 'Y': 29.93},
                {'ID': '8', 'Name': 'Giza', 'Population': 550000, 'Type': 'Mixed', 'X': 31.21, 'Y': 29.99},
                {'ID': '9', 'Name': 'Mohandessin', 'Population': 180000, 'Type': 'Business', 'X': 31.20, 'Y': 30.05},
                {'ID': '10', 'Name': 'Dokki', 'Population': 220000, 'Type': 'Mixed', 'X': 31.21, 'Y': 30.03},
                {'ID': '11', 'Name': 'Shubra', 'Population': 450000, 'Type': 'Residential', 'X': 31.24, 'Y': 30.011},
                {'ID': '12', 'Name': 'Helwan', 'Population': 350000, 'Type': 'Industrial', 'X': 31.33, 'Y': 29.85},
                {'ID': '13', 'Name': 'New Administrative Capital', 'Population': 50000, 'Type': 'Government', 'X': 31.8, 'Y': 30.02},
                {'ID': '14', 'Name': 'Al Rehab', 'Population': 120000, 'Type': 'Residential', 'X': 31.49, 'Y': 30.06},
                {'ID': '15', 'Name': 'Sheikh Zayed', 'Population': 150000, 'Type': 'Residential', 'X': 30.94, 'Y': 30.01}
            ]
            for row in neighborhoods:
                node_id = str(row['ID'])
                self.G.add_node(node_id,
                                name=row['Name'],
                                type=row['Type'],
                                population=row['Population'],
                                pos=[row['Y'], row['X']])
            print(f"Loaded {len(neighborhoods)} neighborhoods")

            # Hardcoded facilities
            facilities = [
                {'ID': 'F1', 'Name': 'Cairo International Airport,Airport', 'Type': 'Airport', 'X': 31.41, 'Y': 30.11},
                {'ID': 'F2', 'Name': 'Ramses Railway Station', 'Type': 'Transit Hub', 'X': 31.25, 'Y': 30.06},
                {'ID': 'F3', 'Name': 'Cairo University', 'Type': 'Education', 'X': 31.21, 'Y': 30.03},
                {'ID': 'F4', 'Name': 'Al-Azhar University', 'Type': 'Education', 'X': 31.26, 'Y': 30.05},
                {'ID': 'F5', 'Name': 'Egyptian Museum', 'Type': 'Tourism', 'X': 31.23, 'Y': 30.05},
                {'ID': 'F6', 'Name': 'Cairo International Stadium', 'Type': 'Sports', 'X': 31.3, 'Y': 30.07},
                {'ID': 'F7', 'Name': 'Smart Village', 'Type': 'Business', 'X': 30.97, 'Y': 30.07},
                {'ID': 'F8', 'Name': 'Cairo Festival City', 'Type': 'Commercial', 'X': 31.4, 'Y': 30.03},
                {'ID': 'F9', 'Name': 'Qasr El Aini Hospital', 'Type': 'Medical', 'X': 31.23, 'Y': 30.03},
                {'ID': 'F10', 'Name': 'Maadi Military Hospital', 'Type': 'Medical', 'X': 31.25, 'Y': 29.95}
            ]
            for row in facilities:
                node_id = str(row['ID'])
                self.G.add_node(node_id,
                                name=row['Name'],
                                type=row['Type'],
                                pos=[row['Y'], row['X']])
            print(f"Loaded {len(facilities)} facilities")
        except Exception as e:
            raise ValueError(f"Error loading nodes: {e}")

    def load_existing_roads(self, weight_type='distance'):
        try:
            roads = [
                {'FromID': '1', 'ToID': '3', 'Distance': 8.5, 'Capacity': 3000, 'Condition': 7},
                {'FromID': '1', 'ToID': '8', 'Distance': 6.2, 'Capacity': 2500, 'Condition': 6},
                {'FromID': '2', 'ToID': '3', 'Distance': 5.9, 'Capacity': 2800, 'Condition': 8},
                {'FromID': '2', 'ToID': '5', 'Distance': 4.0, 'Capacity': 3200, 'Condition': 9},
                {'FromID': '3', 'ToID': '5', 'Distance': 6.1, 'Capacity': 3500, 'Condition': 7},
                {'FromID': '3', 'ToID': '6', 'Distance': 3.2, 'Capacity': 2000, 'Condition': 8},
                {'FromID': '3', 'ToID': '9', 'Distance': 4.5, 'Capacity': 2600, 'Condition': 6},
                {'FromID': '3', 'ToID': '10', 'Distance': 3.8, 'Capacity': 2400, 'Condition': 7},
                {'FromID': '4', 'ToID': '2', 'Distance': 15.2, 'Capacity': 3800, 'Condition': 9},
                {'FromID': '4', 'ToID': '14', 'Distance': 5.3, 'Capacity': 3000, 'Condition': 10},
                {'FromID': '5', 'ToID': '11', 'Distance': 7.9, 'Capacity': 3100, 'Condition': 7},
                {'FromID': '6', 'ToID': '9', 'Distance': 2.2, 'Capacity': 1800, 'Condition': 8},
                {'FromID': '7', 'ToID': '8', 'Distance': 24.5, 'Capacity': 3500, 'Condition': 8},
                {'FromID': '7', 'ToID': '15', 'Distance': 9.8, 'Capacity': 3000, 'Condition': 9},
                {'FromID': '8', 'ToID': '10', 'Distance': 3.3, 'Capacity': 2200, 'Condition': 7},
                {'FromID': '8', 'ToID': '12', 'Distance': 14.8, 'Capacity': 2600, 'Condition': 5},
                {'FromID': '9', 'ToID': '10', 'Distance': 2.1, 'Capacity': 1900, 'Condition': 7},
                {'FromID': '10', 'ToID': '11', 'Distance': 8.7, 'Capacity': 2400, 'Condition': 6},
                {'FromID': '11', 'ToID': 'F2', 'Distance': 3.6, 'Capacity': 2200, 'Condition': 7},
                {'FromID': '12', 'ToID': '1', 'Distance': 12.7, 'Capacity': 2800, 'Condition': 6},
                {'FromID': '13', 'ToID': '4', 'Distance': 45.0, 'Capacity': 4000, 'Condition': 10},
                {'FromID': '14', 'ToID': '13', 'Distance': 35.5, 'Capacity': 3800, 'Condition': 9},
                {'FromID': '15', 'ToID': '7', 'Distance': 9.8, 'Capacity': 3500, 'Condition': 9},
                {'FromID': 'F1', 'ToID': '5', 'Distance': 7.5, 'Capacity': 3200, 'Condition': 9},
                {'FromID': 'F1', 'ToID': '2', 'Distance': 9.2, 'Capacity': 2000, 'Condition': 8},
                {'FromID': 'F2', 'ToID': '3', 'Distance': 2.5, 'Capacity': 2000, 'Condition': 7},
                {'FromID': 'F7', 'ToID': '15', 'Distance': 8.3, 'Capacity': 2800, 'Condition': 8},
                {'FromID': 'F8', 'ToID': '4', 'Distance': 6.1, 'Capacity': 3000, 'Condition': 9},
                {'FromID': 'F9', 'ToID': '10', 'Distance': 0.5, 'Capacity': 2000, 'Condition': 7}
            ]
            for row in roads:
                from_id = str(row['FromID'])
                to_id = str(row['ToID'])
                if from_id in self.G and to_id in self.G:
                    attrs = {
                        'distance': row['Distance'],
                        'capacity': row['Capacity'],
                        'condition': row['Condition'],
                        'type': 'existing_road'
                    }
                    if weight_type == 'distance':
                        attrs['weight'] = row['Distance']
                    elif weight_type == 'capacity':
                        attrs['weight'] = row['Capacity']
                    else:
                        raise ValueError("weight_type must be 'distance' or 'capacity'")
                    self.G.add_edge(from_id, to_id, **attrs)
            print(f"Loaded {len(roads)} existing roads")
        except Exception as e:
            raise ValueError(f"Error loading existing roads: {e}")

    def load_metro_lines(self):
        try:
            metro_lines = [
                {
                    'Name': 'Line 1 (Helwan-New Marg)', 
                    'Stations': ['12', '1', '3', 'F2', '11'],
                    'Daily Passengers': 1500000,
                    'Color': 'red'
                },
                {
                    'Name': 'Line 2 (Shubra-Giza)', 
                    'Stations': ['11', 'F2', '3', '10', '8'],
                    'Daily Passengers': 1200000,
                    'Color': 'green'
                },
                {
                    'Name': 'Line 3 (Airport-Imbaba)', 
                    'Stations': ['F1', '5', '2', '3', '9'],
                    'Daily Passengers': 800000,
                    'Color': 'blue'
                }
            ]
            for line in metro_lines:
                stations = line['Stations']
                line_name = line['Name']
                daily_passengers = line['Daily Passengers']
                color = line['Color']
                for i in range(len(stations) - 1):
                    from_id = stations[i]
                    to_id = stations[i + 1]
                    if from_id in self.G and to_id in self.G:
                        self.G.add_edge(from_id, to_id,
                                        type='metro',
                                        line_name=line_name,
                                        daily_passengers=daily_passengers,
                                        color=color,
                                        sequence=i+1)
            print(f"Loaded {len(metro_lines)} metro lines with proper connections")
        except Exception as e:
            raise ValueError(f"Error loading metro lines: {e}")
    
    def load_bus_routes(self):
        try:
            bus_routes = [
                {'RouteID': 'B1', 'Stops': '1,3,6,9','buses_Assigned': 25,'Daily Passengers': 35000},
                {'RouteID': 'B2', 'Stops': '7,15,8,10,3','buses_Assigned': 23,'Daily Passengers': 42000},
                {'RouteID': 'B3', 'Stops': '2,5,F1','buses_Assigned': 20,'Daily Passengers': 28000},
                {'RouteID': 'B4', 'Stops': '4,14,2,3','buses_Assigned': 22,'Daily Passengers': 31000},
                {'RouteID': 'B5', 'Stops': '8,12,1','buses_Assigned': 18,'Daily Passengers':25000},
                {'RouteID': 'B6', 'Stops': '11,5,2','buses_Assigned': 24,'Daily Passengers': 33000},
                {'RouteID': 'B7', 'Stops': '13,4,14','buses_Assigned': 15,'Daily Passengers': 21000},
                {'RouteID': 'B8', 'Stops': 'F7,15,7','buses_Assigned': 12,'Daily Passengers': 17000},
                {'RouteID': 'B9', 'Stops': '1,8,10,9,6', 'buses_Assigned': 28, 'Daily Passengers': 39000},
                {'RouteID': 'B10', 'Stops': 'F8,4,2,5','buses_Assigned': 20,'Daily Passengers': 28000}
            ]
            bus_colors = ['yellow', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'brown', 'violet']
            for idx, row in enumerate(bus_routes):
                stops = row['Stops'].split(',')
                route_name = row['RouteID']
                buses_Assigned=row['buses_Assigned']
                daily_passengers=row['Daily Passengers']
                color = bus_colors[idx % len(bus_colors)]
                for i in range(len(stops) - 1):
                    from_id = stops[i].strip()
                    to_id = stops[i + 1].strip()
                    if from_id in self.G and to_id in self.G:
                        self.G.add_edge(from_id, to_id,
                                        type='bus',
                                        route_name=route_name,
                                        buses_Assigned=buses_Assigned,
                                        daily_passengers=daily_passengers,
                                        color=color)
            print(f"Loaded {len(bus_routes)} bus routes")
        except Exception as e:
            raise ValueError(f"Error loading bus routes: {e}")

    def load_potential_roads(self, weight_type='distance'):
        try:
            potential_roads = [
                {'FromID': '1', 'ToID': '4', 'Distance': 22.8, 'Capacity': 4000, 'Cost': 450},
                {'FromID': '1', 'ToID': '14', 'Distance': 25.3, 'Capacity': 3800, 'Cost': 500},
                {'FromID': '2', 'ToID': '13', 'Distance': 48.2, 'Capacity': 4500, 'Cost': 950},
                {'FromID': '3', 'ToID': '13', 'Distance': 56.7, 'Capacity': 4500, 'Cost': 1100},
                {'FromID': '5', 'ToID': '4', 'Distance': 16.8, 'Capacity': 3500, 'Cost': 320},
                {'FromID': '6', 'ToID': '8', 'Distance': 7.5, 'Capacity': 2500, 'Cost': 150},
                {'FromID': '7', 'ToID': '13', 'Distance': 82.3, 'Capacity': 4000, 'Cost': 1600},
                {'FromID': '9', 'ToID': '11', 'Distance': 6.9, 'Capacity': 2800, 'Cost': 140},
                {'FromID': '10', 'ToID': 'F7', 'Distance': 27.4, 'Capacity': 3200, 'Cost': 550},
                {'FromID': '11', 'ToID': '13', 'Distance': 62.1, 'Capacity': 4200, 'Cost': 1250},
                {'FromID': '12', 'ToID': '14', 'Distance': 30.5, 'Capacity': 3600, 'Cost': 610},
                {'FromID': '14', 'ToID': '5', 'Distance': 18.2, 'Capacity': 3300, 'Cost': 360},
                {'FromID': '15', 'ToID': '9', 'Distance': 22.7, 'Capacity': 3000, 'Cost': 450},
                {'FromID': 'F1', 'ToID': '13', 'Distance': 40.2, 'Capacity': 4000, 'Cost': 800},
                {'FromID': 'F7', 'ToID': '9', 'Distance': 26.8, 'Capacity': 3200, 'Cost': 540}
            ]
            for row in potential_roads:
                from_id = str(row['FromID'])
                to_id = str(row['ToID'])
                if from_id in self.G and to_id in self.G:
                    attrs = {
                        'distance': row['Distance'],
                        'capacity': row['Capacity'],
                        'cost': row['Cost'],
                        'type': 'potential_road'
                    }
                    if weight_type == 'distance':
                        attrs['weight'] = row['Distance']
                    self.G.add_edge(from_id, to_id, **attrs)
            print(f"Loaded {len(potential_roads)} potential roads")
        except Exception as e:
            raise ValueError(f"Error loading potential roads: {e}")

    def load_traffic_data(self):
        try:
            traffic_data = [
                {'RoadID': '1-3', 'Morning': 2800, 'Afternoon': 1500, 'Evening': 2600, 'Night': 800},
                {'RoadID': '1-8', 'Morning': 2200, 'Afternoon': 1200, 'Evening': 2100, 'Night': 600},
                {'RoadID': '2-3', 'Morning': 2700, 'Afternoon': 1400, 'Evening': 2500, 'Night': 700},
                {'RoadID': '2-5', 'Morning': 3000, 'Afternoon': 1600, 'Evening': 2800, 'Night': 650},
                {'RoadID': '3-5', 'Morning': 3200, 'Afternoon': 1700, 'Evening': 3100, 'Night': 800},
                {'RoadID': '3-6', 'Morning': 1800, 'Afternoon': 1400, 'Evening': 1900, 'Night': 500},
                {'RoadID': '3-9', 'Morning': 2400, 'Afternoon': 1300, 'Evening': 2200, 'Night': 550},
                {'RoadID': '3-10', 'Morning': 2300, 'Afternoon': 1200, 'Evening': 2100, 'Night': 500},
                {'RoadID': '4-2', 'Morning': 3600, 'Afternoon': 1800, 'Evening': 3300, 'Night': 750},
                {'RoadID': '4-14', 'Morning': 2800, 'Afternoon': 1600, 'Evening': 2600, 'Night': 600},
                {'RoadID': '5-11', 'Morning': 2900, 'Afternoon': 1500, 'Evening': 2700, 'Night': 650},
                {'RoadID': '6-9', 'Morning': 1700, 'Afternoon': 1300, 'Evening': 1800, 'Night': 450},
                {'RoadID': '7-8', 'Morning': 3200, 'Afternoon': 1700, 'Evening': 3000, 'Night': 700},
                {'RoadID': '7-15', 'Morning': 2800, 'Afternoon': 1500, 'Evening': 2600, 'Night': 600},
                {'RoadID': '8-10', 'Morning': 2000, 'Afternoon': 1100, 'Evening': 1900, 'Night': 450},
                {'RoadID': '8-12', 'Morning': 2400, 'Afternoon': 1300, 'Evening': 2200, 'Night': 500},
                {'RoadID': '9-10', 'Morning': 1800, 'Afternoon': 1200, 'Evening': 1700, 'Night': 400},
                {'RoadID': '10-11', 'Morning': 2200, 'Afternoon': 1300, 'Evening': 2100, 'Night': 500},
                {'RoadID': '11-F2', 'Morning': 2100, 'Afternoon': 1200, 'Evening': 2000, 'Night': 450},
                {'RoadID': '12-1', 'Morning': 2600, 'Afternoon': 1400, 'Evening': 2400, 'Night': 550},
                {'RoadID': '13-4', 'Morning': 3800, 'Afternoon': 2000, 'Evening': 3500, 'Night': 800},
                {'RoadID': '14-13', 'Morning': 3600, 'Afternoon': 1900, 'Evening': 3300, 'Night': 750},
                {'RoadID': '15-7', 'Morning': 2800, 'Afternoon': 1500, 'Evening': 2600, 'Night': 600},
                {'RoadID': 'F1-5', 'Morning': 3300, 'Afternoon': 2200, 'Evening': 3100, 'Night': 1200},
                {'RoadID': 'F1-2', 'Morning': 3000, 'Afternoon': 2000, 'Evening': 2800, 'Night': 1100},
                {'RoadID': 'F2-3', 'Morning': 1900, 'Afternoon': 1600, 'Evening': 1800, 'Night': 900},
                {'RoadID': 'F7-15', 'Morning': 2600, 'Afternoon': 1500, 'Evening': 2400, 'Night': 550},
                {'RoadID': 'F8-4', 'Morning': 2800, 'Afternoon': 1600, 'Evening': 2600, 'Night': 600},
                {'RoadID': 'F9-10', 'Morning': 1800, 'Afternoon': 1200, 'Evening': 1700, 'Night': 400}
            ]
            for row in traffic_data:
                try:
                    u, v = row['RoadID'].split('-')
                    if self.G.has_edge(u, v):
                        self.G[u][v]['morning_peak'] = row['Morning']
                        self.G[u][v]['afternoon'] = row['Afternoon']
                        self.G[u][v]['evening_peak'] = row['Evening']
                        self.G[u][v]['night'] = row['Night']
                except (ValueError, KeyError):
                    continue
            print(f"Loaded traffic data for {len(traffic_data)} edges")
        except Exception as e:
            raise ValueError(f"Error loading traffic data: {e}")

    def load_transport_demand(self):
        try:
            transport_demand = [
                {'FromID': '3', 'ToID': '5', 'DailyPassengers': 15000},
                {'FromID': '1', 'ToID': '3', 'DailyPassengers': 12000},
                {'FromID': '2', 'ToID': '3', 'DailyPassengers': 18000},
                {'FromID': 'F2', 'ToID': '11', 'DailyPassengers': 25000},
                {'FromID': 'F1', 'ToID': '3', 'DailyPassengers': 20000},
                {'FromID': '7', 'ToID': '3', 'DailyPassengers': 14000},
                {'FromID': '4', 'ToID': '3', 'DailyPassengers': 16000},
                {'FromID': '8', 'ToID': '3', 'DailyPassengers': 22000},
                {'FromID': '3', 'ToID': '9', 'DailyPassengers': 13000},
                {'FromID': '5', 'ToID': '2', 'DailyPassengers': 17000},
                {'FromID': '11', 'ToID': '3', 'DailyPassengers': 24000},
                {'FromID': '12', 'ToID': '3', 'DailyPassengers': 11000},
                {'FromID': '1', 'ToID': '8', 'DailyPassengers': 9000},
                {'FromID': '7', 'ToID': 'F7', 'DailyPassengers': 18000},
                {'FromID': '4', 'ToID': 'F8', 'DailyPassengers': 12000},
                {'FromID': '13', 'ToID': '3', 'DailyPassengers': 8000},
                {'FromID': '14', 'ToID': '4', 'DailyPassengers': 7000}
            ]
            for row in transport_demand:
                from_id = str(row['FromID'])
                to_id = str(row['ToID'])
                if self.G.has_edge(from_id, to_id):
                    self.G[from_id][to_id]['daily_passengers'] = row['DailyPassengers']
            print(f"Loaded transport demand for {len(transport_demand)} edges")
        except Exception as e:
            raise ValueError(f"Error loading transport demand: {e}")

    def get_networkx_graph(self):
        return self.G

    def get_nodes(self):
        return self.G.nodes(data=True)

    def get_edges(self):
        return self.G.edges(data=True)

    def dijkstra_path(self, source, target, weight='distance'):
        """Find the shortest path between two nodes using Dijkstra's algorithm"""
        try:
            path = nx.dijkstra_path(self.G, source=source, target=target, weight=weight)
            length = nx.dijkstra_path_length(self.G, source=source, target=target, weight=weight)
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
        except Exception as e:
            raise ValueError(f"Error in Dijkstra's algorithm: {e}")

    def a_star_path(self, source, target, weight='distance'):
        """Find the shortest path between two nodes using A* algorithm"""
        try:
            def heuristic(u, v):
                pos_u = self.G.nodes[u]['pos']
                pos_v = self.G.nodes[v]['pos']
                return math.sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2)
            
            path = nx.astar_path(self.G, source=source, target=target, heuristic=heuristic, weight=weight)
            length = nx.astar_path_length(self.G, source=source, target=target, heuristic=heuristic, weight=weight)
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
        except Exception as e:
            raise ValueError(f"Error in A* algorithm: {e}")

    def greedy_best_first_path(self, source, target):
        """Find a path between two nodes using Greedy Best-First Search"""
        try:
            def heuristic(u, v):
                pos_u = self.G.nodes[u]['pos']
                pos_v = self.G.nodes[v]['pos']
                return math.sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2)
            
            path = [source]
            current = source
            visited = set([source])
            
            while current != target:
                neighbors = list(self.G.neighbors(current))
                if not neighbors:
                    return None, float('inf')
                
                neighbors.sort(key=lambda x: heuristic(x, target))
                next_node = None
                for node in neighbors:
                    if node not in visited:
                        next_node = node
                        break
                
                if next_node is None:
                    return None, float('inf')
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            total_distance = 0
            for i in range(len(path)-1):
                total_distance += self.G[path[i]][path[i+1]].get('distance', 1)
            
            return path, total_distance
        except Exception as e:
            raise ValueError(f"Error in Greedy Best-First Search: {e}")

    def dynamic_programming_path(self, source, target, weight='distance'):
        """Find the shortest path between two nodes using dynamic programming (Bellman-Ford)"""
        try:
            path = nx.bellman_ford_path(self.G, source=source, target=target, weight=weight)
            length = nx.bellman_ford_path_length(self.G, source=source, target=target, weight=weight)
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
        except nx.NetworkXUnbounded:
            return None, float('inf')
        except Exception as e:
            raise ValueError(f"Error in dynamic programming algorithm: {e}")

    def kruskal_mst(self, weight_attr='distance', must_have_degree2=None):
        """
        Compute a Minimum Spanning Forest of the graph using Kruskal's algorithm.
        - weight_attr: attribute to sort edges by ('distance' or 'cost').
        - must_have_degree2: optional list of node IDs that must have degree >= 2.
        Returns a list of edges in the MST (each edge as (u, v, attrs)).
        """
        # Gather all edges without duplication (u < v lexicographically)
        edges = []
        for u, v, attrs in self.G.edges(data=True):
            if attrs.get('type') in ['existing_road', 'potential_road']:  # Only consider roads
                if u < v:
                    edges.append((u, v, attrs))
                else:
                    edges.append((v, u, attrs))
        
        # Choose weight function
        def edge_weight(item):
            _, _, attrs = item
            return attrs.get(weight_attr, float('inf')) or float('inf')
        
        # Sort edges
        edges.sort(key=edge_weight)
        
        # Initialize union-find over all nodes
        uf = UnionFind(self.G.nodes())
        mst_edges = []
        
        # Kruskal main loop
        for u, v, attrs in edges:
            if uf.union(u, v):
                mst_edges.append((u, v, attrs))
        
        # Enforce minimal degree constraint if required
        if must_have_degree2:
            # Count current degrees in MST
            degree = {n: 0 for n in self.G.nodes()}
            for u, v, _ in mst_edges:
                degree[u] += 1
                degree[v] += 1
            # For each required node, add cheapest extra edge if degree < 2
            for node in must_have_degree2:
                if degree.get(node, 0) < 2:
                    # Find candidate edges incident to node not in MST
                    candidates = []
                    for u, v, attrs in self.G.edges(data=True):
                        if attrs.get('type') not in ['existing_road', 'potential_road']:
                            continue
                        if (u == node or v == node):
                            neighbor = v if u == node else u
                            # Skip if edge already in MST
                            if not any((node == u1 and neighbor == v1) or (node == v1 and neighbor == u1) for u1, v1, _ in mst_edges):
                                candidates.append((node, neighbor, attrs))
                    # Pick the minimum weight candidate
                    if candidates:
                        candidates.sort(key=lambda x: x[2].get(weight_attr, float('inf')))
                        u, v, attrs = candidates[0]
                        mst_edges.append((u, v, attrs))
        
        return mst_edges