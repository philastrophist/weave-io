

class FactorGraph:
	def relation_graph_distance_sum(self, node, given_nodes):
		"""
		Returns the sum of the distances between the node and the given_nodes
		"""

	def uniquely_described_by(node, *others):
		return set(others) in self.relation_graph.nodes[node]['definition_sets']

	def nearest_quickest_filetype(self, node):
		pass

	def resolve(self, directory: Path, parent_ids: Dict[str, str], factors: Dict[str, str], 
				ProductType: str, plural: bool, FileType: str = None):
		"""
		Return the Product(s) referenced by the given information
		Query is to find the path that:
		 	starts at the factor InstanceNodes
			goes through the parent ids InstanceNodes
			ends at the ProductType TypeNode
		instance_graph is a MultiDiGraph with 2-way edges between Hierarchies and files but 1-way edges from factors and ids
		"""
		factor_names, factor_values = zip(*factors.items())
		id_names, id_values = zip(*parent_ids.items())
		if FileType is None:
			possible_product_type_nodes = [p for p in self.product_types if self.relation_graph.nodes[p]['name'] = ProductType]
			possible_file_type_nodes = [f for p in possible_product_type_nodes for f in self.relation_graph.predecessors(p)]
			distances = map(lambda x: self.relation_graph_distance_sum(x, *factors, *parents), possible_file_type_nodes)
			file_type_choices = [f for f, d in zip(possible_file_type_nodes, distances) if d == min(distances)]
			file_type_choices.sort(key=lambda f: self.file_precedence[f])
			FileType = file_type_choices[0]  # only 1 filetype 
		if not self.uniquely_described_by(FileType, factors, parents) and not plural:
			raise KeyError
		product_node = list(self.relation_graph.successors(FileType))[0]
		self.populate_with_schema(schema_files)
		self.populate_with_file_names(directory.glob('*'))  # only with links from fnames
		# required = id_names + factor_names
		# for r in required:
		# 	filetype = self.nearest_quickest_filetype(r)
		# 	for file_instance in self.files[filetype]:
		# 		self.populate_with_file_data(file_instance, r)
		accepted = []
		undirected_instance_graph = self.instance_graph.undirected()
		for file_instance in self.files[FileType]:
			self.populate_with_file_data(file_instance, factors, parents)  # updates relevant relations from inside the file (and irrelevant ones if they come freely)
			for node in parent_values + factor_values:
				if not self.undirected_instance_graph.has_path(node, file_instance):
					break  # not accepted
			else:
				accepted.append(file_instance)
		products = []
		for file_instance in accepted:
			product = file_instance.read(ProductType)  # read into the file_instance, return reference
			products.append(product)
		return UnifiedProduct(products)
