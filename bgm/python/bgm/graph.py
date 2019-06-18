try :
	import pandas as pd
except ImportError as e :
	print('Attempt to import Pandas failed')
try :
	import numpy as np
except ImportError as e :
	print('Attempt to import Numpy failed')

class Node :
	def __init__(self,
		         name='',
		         value=None) :
		self.name = name
		self.value = value

class Edge :
	def __init__(self,
		         source,
		         target=None,
		         value=None,
		         fn=None,
		         directed=False) :
		self.connection = [source,target] # List of Node Name or Node Object
		self.directed = directed
		self.value = value
		self.fn = fn

	def __getitem__(self,idx) :
		if idx >= 2 :
			raise
		return self.connection[idx]

class Graph :
	def __init__(self,rep):
		self.rep = None
		self.name_to_node = None
		self.load(rep)

	def __repr__(self) :
		s = ''
		for node,edge_list in self.rep.items() :
			s += node + ' --> ['
			for edge in edge_list :
				s += edge[1] + ', '
			if s[-1] == '[' :
				s += ']\n'
			else :
				s = s[:-2] + ']\n'
		return s[:-1]

	def __getitem__(self,node_name) :
		if node_name not in self.name_to_node :
			raise
		return self.name_to_node[node_name]

	def get_neighborhood(self,node_name,return_str=True) :
		nodes = []
		if node_name not in self.rep :
			edges = []
		else :
			edges = self.rep[node_name]
		for edge in edges :
			if return_str :
				nodes.append(edge[1])
			else :
				nodes.append(self.name_to_node[edge[1]])
		return nodes

	def to_undirected(self) :
		new_rep = dict(self.rep)
		for parent,edge_list in self.rep.items() :
			for edge in edge_list :
				edge.directed = False #possibly a bug
				child = edge[1]
				if child not in new_rep :
					new_rep.update({child:[Edge(child,parent)]})
				else :
					add_parent = True
					for grandchild in new_rep[child] :
						if grandchild[1] == parent :
							add_parent = False
							break
					if add_parent :
						new_rep[child].append(Edge(child,parent))
		self.rep = new_rep

	def load(self,graph_rep) :
		# graph rep is either list, dict, pd.Df, np.array
		if type(graph_rep) == dict :
			self.rep = self._construct_from_dict(graph_rep)
		elif type(graph_rep) == str :
			pass #load from file
		elif type(graph_rep) == list :
			self.rep,self.name_to_node,undirect_graph =\
			                                self._construct_from_list(graph_rep)
			if undirect_graph :
				self.to_undirected()
		elif type(graph_rep) == pd.DataFrame :
			self.rep,self.name_to_node =\
			                           self._construct_from_pandas_df(graph_rep)
		elif type(graph_rep) == np.ndarray :
			self.rep,self.name_to_node =\
			                         self._construct_from_numpy_array(graph_rep)

	def _construct_from_pandas_df(self,graph_rep) :
		rep = {}
		name_to_node = {}

		shape = graph_rep.shape
		if len(shape) != 2 :
			return rep,name_to_node
		elif shape[0] != shape[1] :
			return rep,name_to_node

		graph_rep.set_index(graph_rep.columns,inplace=True)
		for node in graph_rep.columns :
			children = graph_rep.columns.values[graph_rep.loc[node,:] == 1]
			if node not in rep :
				rep.update({node:[]})
				name_to_node.update({node:Node(name=node)})
			for child in children :
				rep[node].append(Edge(node,child))

		return rep,name_to_node

	def _construct_from_numpy_array(self,graph_rep) :
		rep = {}
		name_to_node = {}

		shape = graph_rep.shape
		if len(shape) != 2 :
			return rep,name_to_node
		elif shape[0] != shape[1] :
			return rep,name_to_node

		node_names = np.arange(0,shape[0])
		for node in node_names :
			children = node_names[graph_rep[node,:] == 1]
			str_node = str(node)
			if str_node not in rep :
				rep.update({str_node:[]})
				name_to_node.update({str_node:Node(name=str_node)})
			for child in children :
				rep[str_node].append(Edge(str_node,str(child)))

		return rep,name_to_node

	def _construct_from_list(self,graph_rep) :
		#[['A','B'],['B','C'],'D']
		#[['A','B'],('B','C')]
		#[['A','B'],['B','C'],['A','C'],'undirected']
		rep = {}
		name_to_node = {}
		undirect_graph = False

		if len(graph_rep) == 0 :
			return rep,name_to_node,undirect_graph

		for edge_pair in graph_rep :
			if type(edge_pair) == list or type(edge_pair) == tuple :
				if len(edge_pair) != 2 :
					raise
				elif type(edge_pair[0]) != str or type(edge_pair[1]) != str :
					raise # pairs must be strings
				node = edge_pair[0]
				if node not in rep :
					rep.update({node:[Edge(node,edge_pair[1],directed=True)]})
					name_to_node.update({node:Node(name=node)})
				else :
					rep[node].append(Edge(node,edge_pair[1],directed=True))
				if edge_pair[1] not in name_to_node :
						name_to_node.update(\
							             {edge_pair[1]:Node(name=edge_pair[1])})
			elif type(edge_pair) == str : # Nothing is connected
				node = edge_pair
				if edge_pair.lower() == 'undirected' :
					undirect_graph = True
				elif node not in rep :
					rep.update({node:[]})
					name_to_node.update({node:Node(name=node)})
			else :
				raise
		return rep,name_to_node,undirect_graph

	"""
	def _construct_from_dict(self,graph_rep) :
		#{'A': 'B'}
		#{'A': Edge()}
		#{'A': {'EdgeList':'B','fn':lambda x: x+1,'value':7.5}}
		#{'A':{'EdgeList':{'name':'B','value':3.4},'fn':lambda x: x+1} }
		#{'A': ['B','C']}
		rep = {}

		if len(graph_rep) == 0 :
			return rep

		for node_name,edge_list in graph_rep.items() :
			if type(node_name) == str :
				node = Node(name=node_name)
			elif type(node_name) == Node :
				node = node_name
			else :
				raise # Node type must be str or Node

			if type(edge_list) == str :
				rep.update((node,Edge((node.name,edge_list))))
			elif type(edge_list) == Edge :
				rep.update((node,edge_list))
			elif type(edge_list) == dict :
				node_fn = edge_list.get('fn',None)
				node_val = edge_list.get('value',None)
				node.fn = node_fn
				node.value = node_val
				edge_list_val = edge_list.get('EdgeList',None)
				if edge_list_val is None :
					rep.update((node,Edge(())))
				elif type(edge_list_val) == str :
					rep.update((node,Edge((node.name,edge_list_val))))
				elif type(edge_list_val) == Edge :
					rep.update((node,edge_list_val))
				elif type(edge_list_val) == dict :
					pass
				else :
					raise
			elif type(edge_list) == list :
				for edge in edge_list :
					if type(edge) == str :
						pass
					elif type(edge) == Edge :
						pass

	"""
	def get_adj_matrix(self) :
		node_names = list(self.name_to_node.keys())
		adj_matrix = pd.DataFrame(index=node_names,
			                      columns=node_names)
		adj_matrix.fillna(0,inplace=True)
		for node,edge_list in self.rep.items() :
			for edge in edge_list :
				adj_matrix.loc[node,edge[1]] = 1

		return adj_matrix

