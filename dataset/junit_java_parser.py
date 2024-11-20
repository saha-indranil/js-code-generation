from tree_sitter import Language, Parser
import tree_sitter_utils.utils as tsutils
from typing import List, Dict, Any, Set, Optional
from common import constants

class JavaParser():
	
	def __init__(self):
		self.parser = tsutils.get_parser(constants.JAVA)
		
	def parse_at_file_path(self, file_path):
		with open(file_path, 'r') as content_file:
			try: 
				content = content_file.read()
				return self.parse_content(content)
			except:
				return list()

	@staticmethod
	def parse_get_all_methods_only(content):
		methods = list()
		methods_list = tsutils.query_matching_for_code('java', content, get_nodes=True).get('function_block', [])
		for method_node in methods_list:
			method_metadata = JavaParser.get_function_metadata('', method_node, content)
			methods.append(method_metadata)
		return methods

	@staticmethod
	def get_declarations(content):
		import_query = "(import_declaration) @import"
		package_query = "(package_declaration) @package"
		import_list = "\n".join(tsutils.query_matching_for_code('java', content, patterns_to_match=import_query).get('import', []))
		package_list = "\n".join(tsutils.query_matching_for_code('java', content, patterns_to_match=package_query).get('package', []))
		return "\n".join([package_list, import_list]) 

	@staticmethod
	def process_comment_from_code(code):
		import tree_sitter_utils.utils as tsutils
		for comment in tsutils.query_matching_for_code('java', code).get('comments', []):
			code = code.replace(comment, '')
		return code

	def parse_content(self, content):
		"""
		Parses a java file and extract metadata of all the classes and methods defined
		"""
		declarations = JavaParser.get_declarations(content)

		self.content = content
		tree = tsutils.get_parse_tree(self.parser, self.content)
		content_without_comment = JavaParser.process_comment_from_code(content)

		classes = [node for node in tree.root_node.children if node.type == 'class_declaration']
		#print(tree.root_node.sexp())
		
		#Parsed Classes
		parsed_classes = list()

		#Classes
		all_class_metadata = []
		for _class in classes:

			#Class metadata
			class_identifier = self.match_from_span([child for child in _class.children if child.type == 'identifier'][0], content).strip()
			class_metadata = self.get_class_metadata(_class, content)

			methods = list()

			#Parse methods
			for child in (child for child in _class.children if child.type == 'class_body'):
				for _, node in enumerate(child.children):
					if node.type == 'method_declaration' or node.type == 'constructor_declaration':	
						#Read Method metadata
						method_metadata = JavaParser.get_function_metadata(class_identifier, node, content)
						methods.append(method_metadata)

			class_metadata['methods'] = methods
			class_metadata['file_body'] = content_without_comment
			class_metadata['declarations'] = declarations
			
			all_class_metadata.append(class_metadata)
		
		for metadata in all_class_metadata:
			metadata['number_of_classes_in_file'] = len(all_class_metadata)
			parsed_classes.append(metadata)

		return parsed_classes

	@staticmethod
	def get_class_metadata(class_node, blob: str):
		"""
		Extract class-level metadata 
		"""
		metadata = {
			'modifiers': '',
			'class_body': '',
			'identifier': '',
			'type_parameters': '',
			'superclass': '',
			'interfaces': '',
			'fields': '',
			'argument_list': '',
			'methods':'',
		}
		metadata['class_body'] = class_node.text.decode('utf-8')
		#Modifier
		modifiers = class_node.child_by_field_name('modifiers')
		if modifiers:
			metadata['modifiers'] = JavaParser.match_from_span(modifiers, blob)

		#Modifier
		type_parameters = class_node.child_by_field_name('type_parameters')
		if type_parameters:
			metadata['type_parameters'] = JavaParser.match_from_span(type_parameters, blob)
		
		#Superclass
		superclass = class_node.child_by_field_name('superclass')
		if superclass:
			metadata['superclass'] = JavaParser.match_from_span(superclass, blob)
		
		#Interfaces
		interfaces = class_node.child_by_field_name('interfaces')
		if interfaces:
			metadata['interfaces'] = JavaParser.match_from_span(interfaces, blob)
		
		#Fields
		fields = JavaParser.get_class_fields(class_node, blob)
		metadata['fields'] = fields

		#Identifier and Arguments
		is_header = False
		for n in class_node.children:
			if is_header:
				if n.type == 'identifier':
					metadata['identifier'] = JavaParser.match_from_span(n, blob).strip('(:')
				elif n.type == 'argument_list':
					metadata['argument_list'] = JavaParser.match_from_span(n, blob)
			if n.type == 'class':
				is_header = True
			elif n.type == ':':
				break
		return metadata



	@staticmethod
	def get_class_fields(class_node, blob: str):
		"""
		Extract metadata for all the fields defined in the class
		"""
		
		body_node = class_node.child_by_field_name("body")
		fields = []
		
		for f in JavaParser.children_of_type(body_node, "field_declaration"):
			field_dict = {}

			#Complete field
			field_dict["original_string"] = JavaParser.match_from_span(f, blob)

			#Modifier
			modifiers_node_list = JavaParser.children_of_type(f, "modifiers")
			if len(modifiers_node_list) > 0:
				modifiers_node = modifiers_node_list[0]
				field_dict["modifier"] = JavaParser.match_from_span(modifiers_node, blob)
			else:
				field_dict["modifier"] = ""

			#Type
			type_node = f.child_by_field_name("type")
			field_dict["type"] = JavaParser.match_from_span(type_node, blob)

			#Declarator
			declarator_node = f.child_by_field_name("declarator")
			field_dict["declarator"] = JavaParser.match_from_span(declarator_node, blob)
			
			#Var name
			var_node = declarator_node.child_by_field_name("name")
			field_dict["var_name"] = JavaParser.match_from_span(var_node, blob)

			fields.append(field_dict)

		return fields



	@staticmethod
	def get_function_metadata(class_identifier, function_node, blob: str):
		"""
		Extract method-level metadata 
		"""		
		metadata = {
			'identifier': '',
			'parameters': '',
			'modifiers': '',
			'return' : '',
			'body': '',
			'class': '',
			'signature': '',
			'full_signature': '',
			'class_method_signature': '',
			'testcase': '',
			'constructor': '',
		}

		# Parameters
		declarators = []
		JavaParser.traverse_type(function_node, declarators, '{}_declaration'.format(function_node.type.split('_')[0]))
		parameters = []
		for n in declarators[0].children:
			if n.type == 'identifier':
				metadata['identifier'] = JavaParser.match_from_span(n, blob).strip('(')
			elif n.type == 'formal_parameters':
				parameters.append(JavaParser.match_from_span(n, blob))
		metadata['parameters'] = ' '.join(parameters)

		#Body
		metadata['body'] = JavaParser.match_from_span(function_node, blob)
		metadata['class'] = class_identifier

		#Constructor
		metadata['constructor'] = False
		if "constructor" in function_node.type:
			metadata['constructor'] = True

		#Test Case
		modifiers_node_list = JavaParser.children_of_type(function_node, "modifiers")
		metadata['testcase'] = False
		for m in modifiers_node_list:
			modifier = JavaParser.match_from_span(m, blob)
			if '@Test' in modifier or 'test' in metadata['identifier'].lower():
				metadata['testcase'] = True

		#Method Invocations
		invocation = []
		method_invocations = list()
		JavaParser.traverse_type(function_node, invocation, '{}_invocation'.format(function_node.type.split('_')[0]))
		for inv in invocation:
			name = inv.child_by_field_name('name')
			method_invocation = JavaParser.match_from_span(name, blob)
			method_invocations.append(method_invocation)
		metadata['invocations'] = method_invocations

		#Modifiers and Return Value
		for child in function_node.children:
			if child.type == "modifiers":
				metadata['modifiers']  = ' '.join(JavaParser.match_from_span(child, blob).split())
			if("type" in child.type):
				metadata['return'] = JavaParser.match_from_span(child, blob)
		
		#Signature
		metadata['signature'] = '{} {}{}'.format(metadata['return'], metadata['identifier'], metadata['parameters'])
		metadata['full_signature'] = '{} {} {}{}'.format(metadata['modifiers'], metadata['return'], metadata['identifier'], metadata['parameters'])
		metadata['class_method_signature'] = '{}.{}{}'.format(class_identifier, metadata['identifier'], metadata['parameters'])

		return metadata


	def get_method_names(self, file):
		"""
		Extract the list of method names defined in a file
		"""

		#Build Tree
		with open(file, 'r') as content_file: 
			content = content_file.read()
			self.content = content
		tree = self.parser.parse(bytes(content, "utf8"))
		classes = (node for node in tree.root_node.children if node.type == 'class_declaration')

		#Method names
		method_names = list()

		#Class
		for _class in classes:		
			#Iterate methods
			for child in (child for child in _class.children if child.type == 'class_body'):
				for _, node in enumerate(child.children):
					if node.type == 'method_declaration':
						if not JavaParser.is_method_body_empty(node):
							
							#Method Name
							method_name = JavaParser.get_function_name(node, content)
							method_names.append(method_name)

		return method_names


	@staticmethod
	def get_function_name(function_node, blob: str):
		"""
		Extract method name
		"""
		declarators = []
		JavaParser.traverse_type(function_node, declarators, '{}_declaration'.format(function_node.type.split('_')[0]))
		for n in declarators[0].children:
			if n.type == 'identifier':
				return JavaParser.match_from_span(n, blob).strip('(')


	@staticmethod
	def match_from_span(node, blob: str) -> str:
		"""
		Extract the source code associated with a node of the tree
		"""
		line_start = node.start_point[0]
		line_end = node.end_point[0]
		char_start = node.start_point[1]
		char_end = node.end_point[1]
		lines = blob.split('\n')
		if line_start != line_end:
			return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
		else:
			return lines[line_start][char_start:char_end]


	@staticmethod
	def traverse_type(node, results: List, kind: str) -> None:
		"""
		Traverses nodes of given type and save in results
		"""
		if node.type == kind:
			results.append(node)
		if not node.children:
			return
		for n in node.children:
			JavaParser.traverse_type(n, results, kind)


	@staticmethod
	def is_method_body_empty(node):
		"""
		Check if the body of a method is empty
		"""
		for c in node.children:
			if c.type in {'method_body', 'constructor_body'}:
				if c.start_point[0] == c.end_point[0]:
					return True

	
	@staticmethod
	def children_of_type(node, types):
		"""
		Return children of node of type belonging to types

		Parameters
		----------
		node : tree_sitter.Node
			node whose children are to be searched
		types : str/tuple
			single or tuple of node types to filter

		Return
		------
		result : list[Node]
			list of nodes of type in types
		"""
		if isinstance(types, str):
			return JavaParser.children_of_type(node, (types,))
		return [child for child in node.children if child.type in types]