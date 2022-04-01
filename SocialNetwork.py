"""
Social Network Evolution Engine

Main author: Patrick Shepherd
Developed at the University of Kentucky and Berea College.
"""
import random

import matplotlib.pyplot     as plt
import networkx              as nx
import numpy                 as np
import random                as rnd
import statistics            as stats
import xml.etree.ElementTree as ElementTree

import os
import sys

from copy          import deepcopy
from helper        import *
from tabulate      import tabulate
from xml.dom       import minidom

# Type constants to check against for input verification
BOOL = type( bool()  )
STR  = type( str()   )
INT  = type( int()   )
FLT  = type( float() )
LST  = type( list()  )
DCT  = type( dict()  )

MAXINT_32 = 2147483647

# Admissible keywords for topology parameter
TOPOLOGIES = ['',
              'complete',
              'cycle',
              'random',
              'scale free',
              'small world',
              'star']

# Lists to hold agents in different categories.
CONFORMING = ['default']
REBELLING  = []

HOMOPHILIC   = ['default']
HETEROPHILIC = []
HALFPHILIC   = []

# Dictionary to hold parameter definitions for different agent types.
# Fields are:
#   homophily: determines an agent's satisfaction based on neighbors
#              Options: 'homophilic', 'heterophilic', 'halfphilic', or float representing
#                       optimal agreement percentage
#   conformity: determines an agent's update trajectory
#               Options: 'conforming', 'rebelling', 'gravit_attract', 'gravit_attract_repel'
#   bound: determines whether confidence bound is used in opinion update
#          Options: True, False
#   update_method: determines how an agent updates its opinion
#   num_influencers: determines how many neighbors are considered when updating
#                    Options: integer or 'max'
#                             if integer > number of neighbors, defaults to max
default = {'homophily': 'homophilic',
           'conformity': 'conforming',
           'bound': False,
           'update_method': False,
           'num_influencers': 'max'
           }
AGENT_PROPS = {'default': default}


# The SocialNetwork class
class SocialNetwork:

    def __init__(self, props=None):
        """
        Constructor for SocialNetwork objects.
        :param props: A dict of named properties for the graph.
                      User can provide any key-value pairs they
                      wish.
                      Built-in Properties:
                        n:                  (int) number of nodes
                        directed:           (bool) directed edges or not
                        symmetric:          (bool) symmetric edges or not
                        topology:           (str) name of topology or generator
                        saturation:         (float) average node degree
                        dimensions:         (int) number of diffusion dimensions
                        visibility:         (str) how to initialize masks for privacy
                        weight:             (float or str) how to initialize edge weights
                        weight_mean:        (float) mean weight for Gaussian distribution
                        weight_stdev:       (float) stdev weight for Gaussian distribution
                        friend:             (float) probability of accepting a new edge
                        unfriend:           (float) probability of severing an existing edge
                        update:             (float) probability of updating diffusion dimension
                        resistance_param:   (float, str, or dict) how to initialize
                                            node resistance values
                        uncertainty_param:  (float, str, or dict) how to initialize
                                            node uncertainty values
                        attributes:         (str) continuous or binary diffusion values
                        type_dist:          (dict [type name]->proportion) distribution of
                                            node types by name
                        selfloops:
        """

        if type(props) is not DCT:
            self._log( 'Properties argument must be a dictionary.' )
            return

        # Objects only have two named attributes.
        # self._properties is a dictionary [property name]->[property value]
        # self._graph is the Networkx object underlying the structure.
        self._properties = props
        self._graph      = None
        self._temp_props = dict()

        self._build()

    def _build(self):
        """
        Main driver function to construct the graph.
        Cleans up property dictionary, then initializes all graph
        properties and representations.
        :return: None
        """

        # Create nodes and edges
        self._validate_input()
        self.initialize_graph()
        self.generate_edges()
        self.initialize_attribute_space()
        self.initialize_masks()
        self.initialize_correlations()
        self.initialize_resistance()
        self.initialize_uncertainty()
        self.initialize_confidence_bound()
        self.mix_network()
        self.load_agent_types()

    def _log(self, msg):
        """
        Print param:msg to the console.
        :param msg: The message to be printed
        :return: None
        """
        if type(msg) is not STR:
            pass
        # print(msg)

    def _save(self, filename):
        """
        Writes the current network out to a file.
        :param filename: Filename string to be written to.
        :return:
        """
        # Make sure filename ends in xml
        if '.xml' not in filename: filename += '.xml'
        self._log('Writing network to %s' % os.path.join(os.getcwd(), filename))

        # Set up a dictionary that will store different encodings for
        # attributes of different types.
        attr = {}
        for key in self._properties:
            if key in ['attribute_space', 'correlations']:
                attr[key] = matrix_to_string(self._properties[key])
            elif key in ['types', 'resistance']:
                attr[key] = vector_to_string(self._properties[key])
            # Skip over these because they get written out at the Node level.
            elif key in ['weights', 'normalized_weights', 'masks']:
                continue
            else:
                attr[key] = str(self.prop(key))

        root = ElementTree.Element('Graph', attrib=attr)

        for i in range(self.prop('n')):
            props = {'idx': str(i)}
            props['parents'] = vector_to_string(self.get_neighbors(i))
            props['masks'] = encode_mask(self.prop('masks')[i])
            props['weights'] = vector_to_string([self.prop('weights')[i][j] for j in self.get_neighbors(i)])

            node = ElementTree.SubElement(root, 'Node', attrib=props)

        f = open(filename, 'w')
        xml_raw = ElementTree.tostring(root).decode('utf-8')
        xml_out = minidom.parseString(xml_raw)
        f.write(xml_out.toprettyxml(indent=" "))
        f.close()

        def _read(self, filename):
            """

            :param self:
            :param filename:
            :return:
            """

            if '.xml' not in filename:
                filename += '.xml'
            if not os.path.exists(filename):
                self._log('Cannot find file: %s' % filename)
                return

            print('Reading file: %s' % filename)

            tree = ElementTree.parse(os.path.join(os.getcwd(), filename))
            root = tree.getroot()

            for a in root.attrib:
                # Matrices
                if a in ['attribute_space', 'correlations']:
                    self._properties[a] = string_to_matrix(root.attrib[a])
                # Float lists
                elif a in ['resistance']:
                    self._properties[a] = string_to_vector(root.attrib[a])
                # Bool lists
                elif a in []:
                    self._properties[a] = [bool(i) for i in root.attrib[a].split(',')]
                # Ints
                elif a in ['dimensions', 'seed', 'n']:
                    self._properties[a] = int(root.attrib[a])
                # Floats
                elif a in ['friend', 'unfriend', 'rewire',
                           'saturation', 'weight', 'update',
                           'weight_mean', 'weight_stdev']:
                    try:
                        self._properties[a] = float(root.attrib[a])
                    except:
                        self._properties[a] = root.attrib[a]
                # Bools
                elif a in ['directed', 'symmetric', 'wearing']:
                    self._properties[a] = bool(root.attrib[a])
                # String lists
                elif a in ['types', 'agent_locations', 'businesses',
                           'home_locations']:
                    self._properties[a] = root.attrib[a].split(',')
                # Dictionaries
                elif a in ['type_dist', 'indexes_by_type', 'business_type_dist',
                           'housing_dist', 'agents_by_location', 'locations']:
                    self._properties[a] = eval(root.attrib[a])
                elif a == 'resistance_param':
                    if root.attrib[a] == 'random':
                        self._properties[a] = 'random'
                    else:
                        try:
                            self._properties[a] = float(root.attrib[a])
                        except ArithmeticError:
                            self._properties[a] = eval(root.attrib[a])
                else:
                    self._properties[a] = root.attrib[a]

            # Set up NetworkX graph underneath
            if self.prop('directed'):
                self._graph = nx.DiGraph()
            else:
                self._graph = nx.Graph()

            # Add nodes to graph
            self._graph.add_nodes_from(range(self.prop('n')))

            self.initialize_edge_weights()
            self.initialize_masks()

            for child in root:

                idx = int(child.attrib['idx'])
                if not child.attrib['parents']:
                    parents = []
                else:
                    parents = [int(i) for i in string_to_vector(child.attrib['parents'])]
                for p in parents:
                    self._graph.add_edge(idx, p)

                for a in child.attrib:

                    if a in ['idx', 'parents']:
                        continue
                    elif a == 'weights':
                        weights = string_to_vector(child.attrib[a])
                        for i in range(len(parents)):
                            self._properties['weights'][parents[i]][idx] = weights[i]
                        self.update_weight_column(idx)
                    elif a == 'masks':
                        masks = string_to_matrix(child.attrib['masks'])
                        for line in masks:
                            parent = int(line[0])
                            mask = decode_mask(line[1])
                            for k in range(len(mask)):
                                if mask[k] == 0:
                                    self._properties['masks'][idx][parent][k] = mask[k]
                                else:
                                    self._properties['masks'][idx][parent][k] = self.prop('attribute_space')[parent][k]

    def _write_adj_matrix(self, filename):
        """

        :param filename:
        :return:
        """
        if '.txt' not in filename:
            filename += '.txt'
        with open(filename, 'w') as f:
            edges = list(self._graph.edges())
            for i in range(self.prop('n')):
                mystr = ''
                for j in range(self.prop('n')):
                    if (i, j) in edges:
                        mystr += '1 '
                    else:
                        mystr += '0 '
                mystr = mystr[:-1] + '\n'
                f.write(mystr)

    def _validate_input(self):
        """
        Check self._properties for standard properties, and fill in default values if needed.
        :return: None
        """

        # Number of nodes
        # Default: 0
        if 'n' not in self._properties:
            self._properties['n'] = 0

        # Directed or undirected graph
        # Default: undirected
        if 'directed' not in self._properties:
            self._properties['directed'] = False

        # Symmetry enforced or not
        # Default: non-symmetric graph
        if 'symmetric' not in self._properties:
            self._properties['symmetric'] = False

        # Graph generation algorithm/prescribed topology
        # Default: empty
        if 'topology' not in self._properties:
            self._properties['topology'] = ''

        # Average percentage of other nodes for one to be connected to.
        # Default: 10%
        if 'saturation' not in self._properties:
            self._properties['saturation'] = 0.1

        # Number of diffusion dimensions
        # Default: 1
        if 'dimensions' not in self._properties:
            self._properties['dimensions'] = 1

        # Scheme for attribute visibility
        # Default: all dimensions visible to all neighbors
        if 'visibility' not in self._properties:
            self._properties['visibility'] = 'visible'

        # Weighting scheme for edges
        # Default: all edges have weight 1.0
        if 'weight' not in self._properties:
            self._properties['weight'] = 1.0
        elif self._properties['weight'] == 'random':
            if 'weight_mean' not in self._properties:
                self._properties['weight_mean']  = .5
            if 'weight_stdev' not in self._properties:
                self._properties['weight_stdev'] = .1

        # Probability of disconnecting from a neighbor
        # Default: 0%
        if 'unfriend' not in self._properties:
            self._properties['unfriend'] = 0.0

        # Probability of connecting to a new neighbor
        # Default: 0%
        if 'friend' not in self._properties:
            self._properties['friend'] = 0.0

        # Probability of updating diffusion value
        # Default: 0%
        if 'update' not in self._properties:
            self._properties['update'] = 1.0

        # Reward threshold for severing a connection
        # Default: 0.0
        if 'unf_thresh' not in self._properties:
            self._properties['unf_thresh'] = 0.5

        # Scheme for update resistance
        # Default: all nodes have resistance 0.0
        if 'resistance_param' not in self._properties:
            self._properties['resistance_param'] = 0.0

        # Scheme for attribute uncertainty
        # Default: all nodes have uncertainty 0.0
        if 'uncertainty_param' not in self._properties:
            self._properties['uncertainty_param'] = 0.0

        # Continuous or binary attributes
        # TODO: add categorical attributes
        if 'attributes' not in self._properties:
            self._properties['attributes'] = 'continuous'

        # Distance bound between agents' attributes
        if 'confidence_bound' not in self._properties:
            self._properties['confidence_bound'] = 1.0

        # Type distribution
        if 'type_dist' not in self._properties:
            self._properties['type_dist'] = {'default': 1.}

        # Whether to have all self-loops in the network
        if 'selfloops' not in self._properties:
            self._properties['selfloops'] = False

        if 'enforce_selfloops' not in self._properties:
            self._properties['enforce_selfloops'] = False

        if 'initialize_at_extremes' not in self._properties:
            self._properties['initialize_at_extremes'] = True

#############################################################
# Begin generation and initialization functions for graphs. #
# Methods:                                                  #
#     initialize_graph()                                    #
#     generate_nodes()                                      #
#     generate_edges()                                      #
#     generate_edge_weight()                                #
#     initialize_edge_weights()                             #
#     initialize_attribute_space()                          #
#     initialize_masks()                                    #
#     initialize_confidence_bound()                         #
#     initialize_correlations()                             #
#     load_agent_types()                                    #
#############################################################

    def initialize_graph(self):
        """
        Creates a networkx graph object.
        :return: None
        """

        # Initialize networkx graph based on user's preference of directed/not
        if self.prop('directed'):
            self._graph = nx.DiGraph()
        else:
            self._graph = nx.Graph()
        self._log('Setting directed=%s' % self._properties['directed'])

    def generate_nodes(self):
        """
        Create the appropriate number of nodes and add them to the graph.
        :return: None
        """
        self._graph.add_nodes_from(range(self._properties['n']))
        self._log('Created %d nodes.' % self._properties['n'])

    def generate_edges(self):
        """
        Generate a set of edges based on self._properties['topology'].
        :return: None
        """

        # Check to see if an adjacency matrix file was provided
        matrix = list()
        if 'matrix_file' in self._properties:
            f = open(self.prop('matrix_file'), 'r')
            for i in f.readlines():
                line = [float(j) for j in i[:-1].split()]
                matrix.append(line)
            f.close()

            matrix = np.array(matrix).T

            if self._graph.number_of_nodes() == 0:
                self._properties['n'] = len(matrix)

        self.generate_nodes()

        # Seed the random number generator if desired.
        if 'seed' in self._properties:
            SEED = self._properties['seed']
        else:
            SEED = None

        topology = self.prop('topology')

        # Case for blank graph.
        if topology == '' and not matrix:
            n = self.prop('n')
            self._properties['weights'] = np.zeros((n, n))
            self._properties['normalized_weights'] = np.zeros((n, n))
            return

        n = self._graph.number_of_nodes()
        edges = list()

        # Generate a set of edges based on the topology requested
        if type(matrix) == LST:
            self._log('Constructing %s graph.' % topology)
        else:
            self._log('Constructing graph from adjacency matrix: %s.' % self.prop('matrix_file'))
            for i in range(self.prop('n')):
                for j in range(self.prop('n')):
                    if matrix[i][j] > 0.:
                        edges.append((i, j))

        if type(matrix) != LST:
            pass
        elif topology == 'random':
            if self.prop('symmetric') and self.prop('directed'):
                edges = nx.erdos_renyi_graph(self._graph.number_of_nodes(),
                                             self.prop('saturation') / 2,
                                             directed=self.prop('directed'),
                                             seed=SEED).edges()
            else:
                edges = nx.erdos_renyi_graph(self._graph.number_of_nodes(),
                                             self.prop('saturation'),
                                             directed=self.prop('directed'),
                                             seed=SEED).edges()
        elif topology == 'scale free':
            edges = nx.scale_free_graph(self._graph.number_of_nodes(),
                                        seed=SEED).edges()
        elif topology == 'small world':
            sat = self.prop('saturation')
            if 'rewire' not in self._properties:
                self._properties['rewire'] = .1
            if self.prop('directed') and not self.prop('symmetric'):
                edges = nx.watts_strogatz_graph(n, int(sat * n * 2),
                                                self.prop('rewire'),
                                                seed=SEED).edges()
            else:
                edges = nx.watts_strogatz_graph(n, int(sat * n),
                                                self.prop('rewire'),
                                                seed=SEED).edges()
        elif topology == 'star':
            edges = nx.star_graph(n).edges()

        elif topology == 'complete':
            edges = nx.complete_graph(n).edges()

        elif topology == 'cycle':
            edges = nx.cycle_graph(n).edges()

        # Add generated edges to graph structure
        for e in edges:
            self._graph.add_edge(e[0], e[1])

            # Add opposite edges if network should be symmetric
            if self.prop('symmetric') and (type(matrix) == LST):
                self._graph.add_edge(e[1], e[0])

        if self.prop('selfloops'):
            for i in range(self.prop('n')):
                self._graph.add_edge(i, i)

        self.initialize_edge_weights(matrix)

    def generate_edge_weight(self):
        """
        Generates a single edge weight to assign to the graph.
        :return: One edge weight
        """

        if self.prop('weight') == 'random':
            return rnd.random()
        elif self.prop('weight') == 'gaussian':
            return np.random.normal(loc=self.prop('weight_mean'),
                                    scale=self.prop('weight_stdev'))
        else:
            return self.prop('weight')

    def initialize_edge_weights(self, matrix=None):
        """
        Initialize blank n x n matrices to hold raw and normalized values for
        edge weights.  Raw values can be between 0 and 1.  After normalization,
        the sum of edge weights coming into a node must equal 1.
        :param matrix:
        :return: None
        """

        n = self.prop('n')
        self._properties['weights'] = np.zeros((n, n))
        self._properties['normalized_weights'] = np.zeros((n, n))

        # Handle reading from an adjacency matrix file
        if type(matrix) != LST:
            for (u, v) in self._graph.edges():
                self._properties['weights'][u][v] = matrix[u][v]
                self._properties['weights'][v][u] = matrix[v][u]
            for i in range(n):
                self.update_weight_column(i)

            # print("Inside statement")
            # print(self._properties['weights'])

            return

        # Handle generating a new matrix
        for u in self._graph.nodes():
            nbrs = self.get_neighbors(u)
            for v in nbrs:
                if self.prop('weights')[v][u] > 0.:
                    continue
                self._properties['weights'][v][u] = self.generate_edge_weight()

                if (not self.prop('directed')) or (self.prop('symmetric')):
                    self._properties['weights'][u][v] = self.prop('weights')[v][u]

            if self.prop('selfloops'):
                self._properties['weights'][u][u] = self.generate_edge_weight()

            # Once all raw weights have been initialized, populate the
            # appropriate normalized weight column.
            self.update_weight_column(u, includeself=False)

        if not self.verify_normalized_weights():
            self._log('Normalized edge weights may be incorrect.  Proceed with caution.')

    def verify_normalized_weights(self):
        """
        Check that normalization was correct - i.e. that weight columns sum to 1.
        :return: True if weights are correct, False otherwise
        """
        if abs(sum(self.prop('normalized_weights').sum(axis=0)) -
               self.prop('n')) > 0.00001:
            self._log('Edge weights appear to be corrupted.')
            return False
        return True

    def initialize_attribute_space(self):
        """
        Initializes the diffusion space for the graph
        :return: None
        """

        n = self.prop('n')
        k = self.prop('dimensions')
        matrix = list()

        # Binary diffusion values (option A or option B)
        if self.prop('attributes') == 'binary':
            matrix = [[rnd.choice([-1., 1.]) for j in range(k)]
                      for i in range(n)]

        # Continuous diffusion values
        elif self.prop('attributes') == 'continuous':
            if not self.prop('initialize_at_extremes'):
                matrix = [[rnd.random() for j in range(k)]
                          for i in range(n)]
            else:
                matrix = []
                for i in range(n):
                    row = []
                    for j in range(k):
                        if random.random() >= 0.5:
                            row.append(1.)
                        else:
                            row.append(-1.)
                    matrix.append(row)

        # Convert matrix to numpy ndarray and set class attribute
        self._properties['attribute_space'] = np.array(matrix)
        self._log('Set initial diffusion values.')

    def initialize_masks(self):
        """
        Initialize 3D matrix for visibility values.
        If node i knows what node j's value in dimension k is, then
        self.prop('masks')[i][j][k] is that value, otherwise it is 0.
        :return: None
        """

        n = self.prop('n')
        K = self.prop('dimensions')
        vis = self.prop('visibility')

        self._properties['masks'] = np.zeros((n, n, K))

        for i in range(n):
            for k in range(K):
                if self.prop('selfloops'):
                    self._properties['masks'][i][i][k] = self.prop('attribute_space')[i][k]
                else:
                    self._properties['masks'][i][i][k] = 0.

            nbrs = list(self.get_neighbors(i, forward=True))
            for j in nbrs:
                if vis == 'random':
                    for k in range(K):
                        self._properties['masks'][j][i][k] = rnd.choice([0., self.prop('attribute_space')[i][k]])
                elif vis == 'visible':
                    for k in range(K):
                        self._properties['masks'][j][i][k] = self.prop('attribute_space')[i][k]

        self._log('Set initial mask values with condition \'%s\'.' % vis)

    def initialize_confidence_bound(self):
        """
        Initializes the confidence bound for each node.
        The confidence bound is the maximum distance node i can be from node
        j in diffusion space before i will no longer consider j's value
        when updating its own.
        :return: None
        """
        self._properties['confidence'] = [self.prop('confidence_bound')
                                          for i in range(self.prop('n'))]

    def initialize_correlations(self):
        """
        Initializes the correlation matrix to make some diffusion dimensions
        depend on others.
        :return: None
        """
        self._properties['correlations'] = np.identity(self.prop('dimensions'))

    def initialize_resistance(self):
        """
        Initializes resistance values for each node.
        :return: None
        """

        n = self.prop('n')
        self._properties['resistance'] = np.zeros(n)
        myparam = self.prop('resistance_param')

        # TODO: clean this up and refactor

        # Set resistance scores by agent type
        if type(myparam) is DCT:

            # Iterate through types
            for key in myparam:
                if key in [i for i in self._properties['type_dist']]:
                    idxs = self._properties['indexes_by_type'][key]
                    for idx in idxs:
                        # Initialize values uniformly at random
                        if myparam[key] == 'random':
                            self._properties['resistance'][idx] = rnd.random()
                        # Initialize to a constant for each node of type 'key'.
                        if type(myparam[key]) is FLT:
                            if myparam[key] < 0. or myparam[key] > 1.:
                                print('Resistance for [%s] must be between 0 and 1.' % myparam[key])
                                print('\tInvalid value: %f' % myparam[key])
                                break
                            self._properties['resistance'][idx] = myparam[key]
                else:
                    print('Type [%s] does not match any key in type_dist.  Skipping.' % key)

    def initialize_uncertainty(self):
        """
        Initializes uncertainty values for each node.
        :return: None
        """

        n = self.prop('n')
        self._properties['uncertainty'] = np.zeros(n)
        myparam = self.prop('uncertainty_param')

        # TODO: clean up and refactor, same as above

        # Set resistance scores by agent type
        if type(myparam) is DCT:

            # Iterate through types
            for key in myparam:
                if key in [i for i in self._properties['type_dist']]:
                    idxs = self._properties['indexes_by_type'][key]
                    for idx in idxs:
                        # Initialize values uniformly at random
                        if myparam[key] == 'random':
                            self._properties['uncertainty'][idx] = rnd.random()
                        # Initialize to a constant for each node of type 'key'.
                        if type(myparam[key]) is FLT:
                            if myparam[key] < 0. or myparam[key] > 1.:
                                print('Uncertainty for [%s] must be between 0 and 1.' % myparam[key])
                                print('\tInvalid value: %f' % myparam[key])
                                break
                            self._properties['uncertainty'][idx] = myparam[key]
                else:
                    print('Type [%s] does not match any key in type_dist.  Skipping.' % key)

    def mix_network(self):
        """
        Distribute agent types among the nodes.
        :return:
        """

        n = self.prop('n')
        self._properties['types'] = list()
        self._properties['indexes_by_type'] = dict()

        # If no distribution is given, make all nodes default types.
        if 'type_dist' not in self._properties:
            self._properties['type_dist'] = {'default': 1.}
            return

        max_num = 0
        max_t = ''

        nums = {}
        d = self.prop('type_dist')

        # Make sure type distribution sums to 1
        if abs(sum([d[key] for key in d]) - 1.) > .000001:
            print('Agent type proportions must sum to 1.')
            return

        for t in d:

            self._properties['indexes_by_type'][t] = list()
            nums[t] = int(d[t] * n)

            # This is just to correct any off-by-ones when we get done filling the type
            # vector.
            if nums[t] >= max_num:
                max_num = nums[t]
                max_t = t

            # Append nums[t] copies of this type's string representation to the vector.
            for i in range(nums[t]):
                self._properties['types'].append(t)

        # Make sure that the rounding above didn't leave us short an element.
        while len(self._properties['types']) < n:
            nums[max_t] += 1
            self._properties['types'].append(max_t)

        rnd.shuffle(self._properties['types'])
        for i in range(len(self.prop('types'))):
            self._properties['indexes_by_type'][self.prop('types')[i]].append(i)

        self._log('Distributed agent types in network.')
        for t in nums:
            self._log("\t'%s':\t%d" % (t, nums[t]))

    def load_agent_types(self):
        """

        :return:
        """
        models = self.prop('agent_models')
        if not models:
            return

        for modelname in models:
            model = models[modelname]
            AGENT_PROPS[modelname] = model

            if model['homophily'] == 'homophilic':
                HOMOPHILIC.append(modelname)
            elif model['homophily'] == 'heterophilic':
                HETEROPHILIC.append(modelname)
            elif model['homophily'] == 'halfphilic':
                HALFPHILIC.append(modelname)

            if model['conformity'] == 'conforming':
                CONFORMING.append(modelname)
            elif model['conformity'] == 'rebelling':
                REBELLING.append(modelname)

            self.set_resistance_by_type(modelname)

#########################################################
# End generation and initialization methods for graphs. #
#########################################################

    def prop(self, p):
        """
        Getter for property values.
        :param p: property name
        :return: property value if it exists, or None
        """

        if p in self._properties:
            return self._properties[p]
        return None

    def set_weight(self, u, v, w):
        """
        Set edge weight for a single edge.
        :param u: origin node
        :param v: destination node
        :param w: weight
        :return: None
        """
        self._properties['weights'][u][v] = w

    def set_resistance(self, u, val):
        """

        :param u:
        :param val:
        :return:
        """
        if val == 'random':
            self._properties['resistance'][u] = random.random()
        elif type(val) is FLT:
            self._properties['resistance'][u] = val

    def set_resistance_by_type(self, typename):
        """

        :param typename:
        :return:
        """
        val = AGENT_PROPS[typename]['resistance']
        nodes = [i for i in range(self.prop('n')) if self.prop('types')[i] == typename]
        for node in nodes:
            self.set_resistance(node, val)

    def connect(self, u, v, force=False):
        """
        Creates an edge between u and v, and sets weights and masks.
        :param u: origin node
        :param v: destination node
        :return: None
        """

        # If nodes are the same or random chance comes up False, then
        # don't connect u and v.
        if (u == v) and (not self.prop('selfloops')) and (self.prop('enforce_selfloops')):
            return False

        if not force and not coin_flip(self.prop('friend')):
            return False

        self.set_visible_dimensions(u, v)

        self._graph.add_edge(u, v)
        w = self.generate_edge_weight()
        self.set_weight(u, v, w)

        if (not self.prop('directed')) or (self.prop('symmetric')):
            self.set_weight(v, u, w)

        if self.prop('symmetric') and self.prop('directed'):
            self._graph.add_edge(v, u)

        self.update_weight_column(u)
        self.update_weight_column(v)

        return True

    def disconnect(self, u, v, force=False):
        """
        Destroys an edge between u and v, and sets weights and masks.
        :param u: origin node
        :param v: destination node
        :return: None
        """

        # If nodes are the same or random chance comes up False, then
        # don't connect u and v.
        if (u == v) and (self.prop('selfloops')) and (self.prop('enforce_selfloops')):
            return False

        if (not force and not coin_flip(self.prop('unfriend'))):
            return False

        self.hide_all(u, v)

        try:
            self._graph.remove_edge(u, v)
        except:
            pass
        self.set_weight(u, v, 0.)

        if not self.prop('directed'):
            self.set_weight(v, u, 0.)
            self.hide_all(v, u, enforce_symmetry=True)

        if self.prop('symmetric'):
            self.set_weight(v, u, 0)
            self.hide_all(v, u)
            if self.prop('directed'):
                try:
                    self._graph.remove_edge(v, u)
                except:
                    pass

        self.update_weight_column(u)
        self.update_weight_column(v)

        return True

    def reveal(self, u, v, k, enforce_symmetry=True):
        """

        :param u:
        :param v:
        :param k:
        :param enforce_symmetry:
        :return:
        """
        self._properties['masks'][v][u][k] = self.prop('attribute_space')[u][k]
        if enforce_symmetry:
            if self.prop('symmetric') or not self.prop('directed'):
                self._properties['masks'][u][v][k] = self.prop('attribute_space')[v][k]

    def hide(self, u, v, k, enforce_symmetry=False):
        """

        :param u:
        :param v:
        :param k:
        :param enforce_symmetry:
        :return:
        """
        self._properties['masks'][v][u][k] = 0.
        if enforce_symmetry:
            if self.prop('symmetric') or not self.prop('directed'):
                self._properties['masks'][u][v][k] = 0.

    def hide_all(self, u, v, enforce_symmetry=False):
        """

        :param u:
        :param v:
        :param enforce_symmetry:
        :return:
        """
        for i in range(self.prop('dimensions')):
            self.hide(u, v, i, enforce_symmetry=enforce_symmetry)

    def broadcast(self, u, k):
        """

        :param u:
        :param k:
        :return:
        """
        for i in self.get_neighbors(u):
            self.reveal(u, i, k)

    def view(self, u, v):
        """

        :param u:
        :param v:
        :return:
        """
        return self.prop('masks')[u][v]

    def set_visible_dimensions(self, u, v):
        """

        :param u:
        :param v:
        :return:
        """
        if self.prop('visibility') == 'visible':
            for k in range(self.prop('dimensions')):
                self.reveal(u, v, k, enforce_symmetry=True)

        elif self.prop('visibility') == 'random':
            for k in range(self.prop('dimensions')):
                if rnd.random() < .5:
                    self.reveal(u, v, k, enforce_symmetry=True)
                else:
                    self.hide(u, v, k, enforce_symmetry=True)

    def actions(self):

        A = ['NOP']
        for i in range(self.prop('dimensions')):
            A.append(f'rev {i}')
        A.append('unf')

        return A

    def execute_action(self, agent, action):
        if 'rev' in action:
            a = action.split()

            self.reveal(agent, int(a[1]), int(a[2]))
            return
        elif 'unf' in action:
            a = action.split()
            self.disconnect(agent, int(a[1]))
            return
        elif 'friend' in action:
            a = action.split()
            self.connect(agent, int(a[1]))
            return

    def get_neighbors(self, u, forward=False):
        """
        Get all neighbors of node u.  These are ancestors in directed graphs,
        and regular neighbors in undirected graphs.
        :param u: Integer, node number
        :param forward: Boolean, True to retrieve successors, False to retrieve predecessors
        :return: Generator for neighbors of u
        """
        if not self.prop('directed'):
            return self._graph.neighbors(u)
        else:
            if forward:
                return self._graph.successors(u)
            else:
                return self._graph.predecessors(u)

    def get_neighbors_of_neighbors(self, u):
        """

        :param u:
        :return:
        """
        candidates = set()
        nbrs = self.get_neighbors(u)
        for nbr in nbrs:
            nbrs2 = self.get_neighbors(nbr)
            for nbr2 in nbrs2:
                candidates.add(nbr2)
        if u in candidates:
            candidates.remove(u)
        return candidates

    def get_reward(self, u, v):
        """

        :param u:
        :param v:
        :return:
        """

        return dist(u, v)

    # def state(self, i, j):
    #
    #     s = list()
    #
    #     ## i's resistance to influence
    #     s.append(self.V[i].props['c'])
    #     ## i's beliefs
    #     s.extend(self.M[i][i])
    #     ## What j thinks i believes
    #     s.extend(self.M[j][i])
    #     ## What i thinks j believes
    #     s.extend(self.M[i][j])
    #     ## Average belief
    #     s.extend(self.W[i].dot(self.M[i]).tolist())
    #
    #     return np.array(s)

    def update_weight_column(self, u, includeself=False):
        """
        Update normalized weights for edges incident on node u.
        For example, if u loses a neighbor, normalized weights must be changed.
        :param u: Integer, node number
        :return: None
        """

        # Zero out row first in case someone unfriended
        for v in range(self._properties['n']):
            self._properties['normalized_weights'][v][u] = 0.

        # Calculate the total amount of influence u receives, then assign each of
        # its neighbors v a weight equal to v's proportional contribution.
        total = self.prop('weights').sum(axis=0)[u]

        nbrs = self.get_neighbors(u)
        for v in nbrs:
            self._properties['normalized_weights'][v][u] = self.prop('weights')[v][u] / total
        if includeself:
            self._properties['normalized_weights'][u][u] = 1. / total

    def get_local_average(self, u, weighted=False, v=True):
        """

        :param u:
        :param weighted:
        :return:
        """
        # if v:
        #     print(f'Getting local average for node {u}')
        #     print('Opinion:')
        #     print(self.prop('attribute_space')[u])
        #     print('Weights:')
        #     print(self.prop('weights'))
        #     print('Normalized weights:')
        #     print(self.prop('normalized_weights'))
        #     print('Masks:')
        #     print(self.prop('masks')[u])
        # print('Node: ', u)
        # print(self.prop('masks')[u])
        # print(self.prop('normalized_weights').T[u])

        if weighted:
            # result = self.prop('normalized_weights').T[u].dot(self.prop('attribute_space')[u]).round(decimals=2)
            result = self.prop('normalized_weights').T[u].dot(self.prop('masks')[u]).round(decimals=2)
            # if v:
            #     print(f'Result [1]: {result}')
            return result
        else:
            # nbrs = [self.prop('weights')[u][j]j for j in range(len(self.prop('weights')))]
            # num_nbrs = len(nbrs)
            # if num_nbrs == 0:
            #     return self.prop('attribute_space')[u]
            #
            # nbr_ops = [self.prop('attribute_space')[nbr] for nbr in nbrs]
            # return sum(nbr_ops) / (num_nbrs)
            # result = self.prop('normalized_weights').T[u].dot(self.prop('attribute_space')).round(decimals=6)
            result = self.prop('normalized_weights').T[u].dot(self.prop('masks')[u]).round(decimals=6)
            # if v:
            #     print(f'Result [2]: {result}')
            return result

    def get_global_average(self, t=None):
        """
        Get the global average over diffusion dimensions.
        Filter by agent type optionally.
        :param t: Agent type to average over.
        :return:
        """

        if t is None:
            return self.prop('attribute_space').mean(axis=0)

        matrix = [self.prop('attribute_space')[i]
                  for i in self.prop('indexes_by_type')[t]]
        return np.array(matrix).mean(axis=0)

    def get_state(self, u, v):
        """

        :param u:
        :param v:
        :return:
        """
        ret = []
        ret.extend(self.prop('attribute_space')[u])
        ret.extend(self.prop('masks')[u][v])
        ret.extend(self.prop('masks')[v][u])
        ret.extend(self.get_local_average(u))
        return np.array(ret)

    def get_vectors_for_distance(self, u, v):
        """

        :param u:
        :param v:
        :return:
        """
        vec1 = self.prop('attribute_space')[u]
        vec2 = self.prop('masks')[u][v]

        distvec1 = [vec1[i] for i in range(len(vec1)) if vec2[i] != 0.]
        if not distvec1:
            return [], []
        distvec2 = [vec2[i] for i in range(len(vec2)) if vec2[i] != 0.]

        return distvec1, distvec2

    def get_reward_for_neighbor(self, u, v):
        """

        :param u: The node assessing its reward
        :param v: The neighbor that reward is being assessed relative to
        :return: The reward u gets from v
        """

        # TODO: Get rid of this.  Handle continuous features more elegantly.
        if self.prop('attributes') == 'continuous':
            return 1 - abs(self.prop('attribute_space')[u][0] - self.prop('attribute_space')[v][0])

        vec1, vec2 = self.get_vectors_for_distance(u, v)
        d = dist(vec1, vec2)

        t = self.prop('types')[u]

        # Use this reward function for homophilic agents.
        if t in HOMOPHILIC:
            return 1 - d
        # Use this reward function for heterophilic agents.
        if t in HETEROPHILIC:
            return d
        # Use this reward function for 5050-philic agents.
        if t in HALFPHILIC:
            return 1 - (abs(d - 0.5) * 2)
        else:
            return 0.

    def get_reward_for_node(self, u):
        """

        :param u:
        :return:
        """
        rewards = [self.get_reward_for_neighbor(u, v) for v in self.get_neighbors(u)]
        if not rewards:
            return 0.
        else:
            return stats.mean(rewards)

    def reward(self, t=None, average=False):
        """

        :param t:
        :param average:
        :return:
        """
        if t is None:
            rewards = [self.get_reward_for_node(i) for i in range(self.prop('n'))]
            if not average:
                return rewards
            else:
                return stats.mean(rewards)
        else:
            rewards = [self.get_reward_for_node(i) for i in range(self.prop('n'))
                       if self.prop('types')[i] == t]
            if not average:
                return rewards
            else:
                return stats.mean(rewards)

    def update_single_attribute(self, u, k, newval):
        """

        :param u:
        :param k:
        :param newval:
        :return:
        """
        self._properties['attribute_space'][u][k] = newval
        for nbr in self.get_neighbors(u, forward=True):
            # TODO: This condition messes things up.  When the 'negative' opinion is 0.,
            #       then this causes updates to work incorrectly.  Needs fixing.
            # if self.prop('masks')[nbr][u][k] != 0.:
            #     self._properties['masks'][nbr][u][k] = newval
            self._properties['masks'][nbr][u][k] = newval

    def update_node(self, u):
        """

        :param u: node to update
        :return: None
        """

        nbrs = sorted(list(self.get_neighbors(u)))
        curr_state = self.prop('attribute_space')[u]
        local_avg = self.get_local_average(u, v=True)
        res = self.prop('resistance')[u]
        mytype = self.prop('types')[u]

        for k in range(len(curr_state)):
            if mytype in CONFORMING:
                if rnd.random() < self.prop('update'):
                    if self.prop('attributes') == 'continuous':
                        self.update_single_attribute(u, k, local_avg[k])
                        continue
                    elif self.prop('attributes') == 'binary':
                        if curr_state[k] * local_avg[k] < 0:
                            if abs(local_avg[k]) > res:
                                self.update_single_attribute(u, k, curr_state[k] * -1)


    def get_next_state(self, u):
        """

        :param u:
        :return:
        """

        next_state = []

        # Return without updating with some probability
        if not coin_flip(self.prop('update')):
            return self.prop('attribute_space')[u]

        la = self.get_local_average(u)
        if self.prop('types')[u] in CONFORMING and self.prop('attributes') == 'continuous':
            return la
        elif self.prop('types')[u] in CONFORMING and self.prop('attributes') == 'binary':
            for k in range(len(la)):
                if self.prop('attribute_space')[u][k] * la[k] > 0:
                    next_state.append(self.prop('attribute_space')[u][k])
                elif abs(la[k]) > self.prop('resistance')[u]:
                    next_state.append(-self.prop('attribute_space')[u][k])
                else:
                    next_state.append(self.prop('attribute_space')[u][k])
        elif self.prop('types')[u] in REBELLING and self.prop('attributes') == 'continuous':
            for k in range(len(la)):
                pass
        elif self.prop('types')[u] in REBELLING and self.prop('attributes') == 'binary':
            for k in range(len(la)):
                if self.prop('attribute_space')[u][k] * la[k] < 0:
                    next_state.append(self.prop('attribute_space')[u][k])
                elif abs(la[k]) > self.prop('resistance')[u]:
                    next_state.append(-self.prop('attribute_space')[u][k])
                else:
                    next_state.append(self.prop('attribute_space')[u][k])

        return next_state

    def update(self):
        """

        :return:
        """

        next_states = [self.get_next_state(u) for u in range(self.prop('n'))]
        for u in range(self.prop('n')):
            next_state = next_states[u]
            for k in range(len(next_state)):
                self.update_single_attribute(u, k, next_state[k])

    def update_old(self, v=True):
        """

        :return: None
        """
        changes = []

        for i in range(self.prop('n')):

            nbrs = sorted(list(self.get_neighbors(i)))

            curr_state = self.prop('attribute_space')[i]
            local_avg = self.get_local_average(i, v=v)
            res = self.prop('resistance')[i]

            for k in range(len(curr_state)):
                if self.prop('types')[i] in CONFORMING:
                    if rnd.random() < self.prop('update'):
                        if self.prop('attributes') == 'continuous':
                            changes.append([i, k, local_avg[k]])
                            continue
                    if curr_state[k] * local_avg[k] > 0:
                        continue
                    else:
                        if abs(local_avg[k]) > res:
                            if rnd.random() < self.prop('update'):
                                # if self.prop('attributes') == 'continuous':
                                #     changes.append([i, k, local_avg[k]])
                                # elif self.prop('attributes') == 'binary':
                                if self.prop('attributes') == 'binary':
                                    changes.append([i, k, curr_state[k] * -1])
                if self.prop('types')[i] in REBELLING:
                    if rnd.random() < self.prop('update'):
                        if self.prop('attributes') == 'continuous':
                            changes.append([i, k, 0 - local_avg[k]])
                            continue
                    if curr_state[k] * local_avg[k] < 0:
                        continue
                    else:
                        if abs(local_avg[k]) > res:
                            if rnd.random() < self.prop('update'):
                                if self.prop('attributes') == 'continuous':
                                    changes.append([i, k, 0 - local_avg[k]])
                                elif self.prop('attributes') == 'binary':
                                    changes.append([i, k, curr_state[k] * -1])

        for [node, dim, val] in changes:
            self._properties['attribute_space'][node][dim] = val
        self.refresh_masks()

    def refresh_masks(self):
        for i in range(self.prop('n')):
            myatts = self.prop('attribute_space')[i]
            nbrs = list(self.get_neighbors(i, forward=True)) + [i]
            for k in range(len(myatts)):
                for nbr in nbrs:
                    self._properties['masks'][nbr][i][k] = myatts[k]

    def create_connections_for_node(self, u, triadic=False, nbrs=1):
        """
        Connects new friends to node u.
        :param u: Node to receive the new connections
        :param triadic: Bool, whether or not to use triadic closures to decide new friends
        :param nbrs: Number of potential friend candidates to consider.
        :return: None
        """

        connections = list()
        if triadic:
            candidates = self.get_neighbors_of_neighbors(u)
        else:
            curr_nbrs = self.get_neighbors(u)
            candidates = [i for i in range(self.prop('n'))
                          if i != u and i not in curr_nbrs]
        candidates = list(candidates)
        random.shuffle(candidates)

        num_nbrs = min(len(candidates), nbrs)
        for c in candidates[:num_nbrs]:
            success = self.connect(u, c)
            if success:
                connections.append((u, c))

        return connections

    def create_connections(self, triadic=False):
        """
        Creates new connections for all nodes in the network.
        :return: None
        """
        connections = list()
        for i in range(self.prop('n')):
            newedges = self.create_connections_for_node(i, triadic=triadic)
            connections.extend(newedges)

        return connections

    def act_node(self, u):
        """

        :param u:
        :return:
        """
        removed = list()
        nbrs = list(self.get_neighbors(u))
        for nbr in nbrs:
            if self.get_reward_for_neighbor(u, nbr) < self.prop('unf_thresh'):
                success = self.disconnect(nbr, u)
                if success:
                    removed.append((nbr, u))

        return removed

    def act(self):
        """

        :return:
        """
        removed = list()
        for i in range(self.prop('n')):
            deletions = self.act_node(i)
            removed.extend(deletions)

        return removed

    def step(self):
        """

        :return:
        """
        self.act()
        self.update()
        self.create_connections()

    def is_defined_command(self, cmdname):
        """

        :param cmdname:
        :return:
        """
        mydir = os.getcwd()
        cmddir = os.path.join(mydir, 'debug_cmds')
        cmdfiles = [i[:-4] for i in os.listdir(cmddir)]
        for f in cmdfiles:
            if cmdname == f:
                return True
        return False

    def debug(self, frominherited=False, mycommand=''):

        self._temp_props['aliases'] = {}
        self._temp_props['defined_commands'] = {}

        mydir = os.getcwd()
        cmd = ''

        if not frominherited:
            print('Welcome to the debugger!')
            print('Type \'help\' for a list of commands.')

        while cmd != 'q':

            cmd = input('\n>>> ')

            if cmd == '':
                continue

            cmdline = cmd.split()

            if cmdline[0] == 'define':

                if not os.path.exists(os.path.join(mydir, 'debug_cmds')):
                    os.mkdir('debug_cmds')

                cmddir = os.path.join(mydir, 'debug_cmds')

                cmdname = cmdline[1]

                cmd_filename = os.path.join(cmddir, '%s.txt' % cmdname)
                f = open(cmd_filename, 'w')

                self._temp_props['defined_commands'][cmdname] = []
                cmd = input('\n>>>* ')
                cmdline = cmd.split()
                while cmdline[0] != 'end':
                    self._temp_props['defined_commands'][cmdname].append(cmd)
                    f.write(cmd + '\n')
                    self.execute_debug_command(cmdline)
                    cmd = input('\n>>>* ')
                    cmdline = cmd.split()
                print('Stored actions for new command \'%s\'' % cmdname)
                f.close()
                continue

            if self.is_defined_command(cmdline[0]):
                cmddir = os.path.join(mydir, 'debug_cmds')
                if not os.path.exists(cmddir):
                    print('No commands have been defined.')
                    continue

                cmd_filename = os.path.join(cmddir, '%s.txt' % cmdline[0])
                f = open(cmd_filename, 'r')
                for line in f.readlines():
                    print('::%s::' % line[:-1])
                    self.execute_debug_command(line.split())
                f.close()
                continue

            self.execute_debug_command(cmdline)

            if frominherited:
                return

        print('Exiting debugger.')
        return

    def execute_debug_command(self, cmdline, frominherited=False):
        """

        :param cmdline:
        :param frominherited:
        :return:
        """
        # Add cases for commands here
        # Update the network according to the defined rules.
        if cmdline[0] == 'step':
            # Update for one step
            if len(cmdline) == 1:
                print('Updating for [1] time step.')
                self.step()
            # Update for a user-specified number of steps
            elif len(cmdline) == 2:
                try:
                    for i in range(int(cmdline[1])):
                        self.step()
                    print('Updating for [%s] time steps.' % cmdline[1])
                except:
                    print('Could not execute [%s] steps.' % cmdline[1])

        # Print out a list of possible commands
        elif cmdline[0] == 'help':
            print('SocialNetwork Commands:')
            print('\tstep <number of steps (optional)>')
            print('\tshow attribute_space <list of nodes (optional)>')
            print('\tshow neighbors <list of nodes (optional)>')
            print('\tshow types <list of nodes (optional)>')
            print('\tshow resistance <list of nodes (optional)>')
            print('\tshow weights <list of nodes (optional)>')
            print('\tshow normalized_weights <list of nodes (optional)>')
            print('\tshow masks <node>')
            print('\tshow reward <node1> <node2>')
            print('\tshow global_average')
            print('\tshow local_average <list of nodes>')
            print('\tshow type_average <type>')
            print('\tshow type_reward <type>')
            print('\tset type <node> <type>')
            print('\tset resistance <resistance>')
            print('\tset resistance <node> <resistance>')
            print('\tset resistance <type> <resistance>')
            print('\trename <property name> <new name>')
            print('\tconnect <node1> <node2>')
            print('\tdisconnect <node1> <node2>')
            print('\treveal <node1> <node2> <dimension>')
            print('\thide <node1> <node2> <dimension>')
            print('\tsave <filename>')

        elif self.is_defined_command(cmdline[0]):
            cmddir = os.path.join(os.getcwd(), 'debug_cmds')
            if not os.path.exists(cmddir):
                print('No commands have been defined.')
                return

            cmd_filename = os.path.join(cmddir, '%s.txt' % cmdline[0])
            f = open(cmd_filename, 'r')
            for line in f.readlines():
                print('::%s::' % line[:-1])
                self.execute_debug_command(line.split())
            f.close()
            return

        elif cmdline[0] == 'update':
            self.update_node(int(cmdline[1]))

        # Use 'show' at the beginning of a command to display the value of
        # a graph attribute.
        elif cmdline[0] == 'show':

            if len(cmdline) == 1:
                print('\'show\' command requires arguments.')
                return
            if len(cmdline) > 2:
                mylist = cmdline[2:]
            else:
                mylist = sorted(list(range(self.prop('n'))))

            # Rename command if needed.
            if cmdline[1] in self._temp_props['aliases']:
                cmdline[1] = self._temp_props['aliases'][cmdline[1]]

            if cmdline[1] == 'attribute_space':
                entries = [['Node', 'Attributes']]
                for i in mylist:
                    try:
                        entries.append([i, self.prop('attribute_space')[int(i)]])
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'neighbors':
                entries = [['Node', 'Neighbors']]
                for i in mylist:
                    try:
                        entries.append([i, sorted(list(self.get_neighbors(int(i))))])
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'types':
                entries = [['Node', 'Type']]
                for i in mylist:
                    try:
                        entries.append([i, self.prop('types')[int(i)]])
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'resistance':
                entries = [['Node', 'Resistance']]
                for i in mylist:
                    try:
                        entries.append([i, self.prop('resistance')[int(i)]])
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'weights':
                if len(cmdline) == 3:
                    if cmdline[2] == 'matrix':
                        print(self.prop('weights'))
                        return
                for i in mylist:
                    print('Node: %s' % i)
                    entries = [['Edge', 'Weight']]
                    nbrs = sorted(list(self.get_neighbors(int(i))))
                    for j in nbrs:
                        try:
                            entries.append(['[%s]-->[%s]' % (j, i), self.prop('weights')[j][int(i)]])
                        except:
                            print('Node %s not in graph.  Skipping...' % i)
                            continue
                    if len(entries) > 1:
                        print(tabulate(entries, headers='firstrow') + '\n')

            elif cmdline[1] == 'normalized_weights':
                if len(cmdline) == 3:
                    if cmdline[2] == 'matrix':
                        print(self.prop('normalized_weights'))
                        return
                for i in mylist:
                    print('Node: %s' % i)
                    entries = [['Edge', 'Normalized Weight']]
                    nbrs = sorted(list(self.get_neighbors(int(i))))
                    for j in nbrs:
                        try:
                            entries.append(
                                ['[%s]-->[%s]' % (j, i), self.prop('normalized_weights')[j][int(i)]])
                        except:
                            print('Node %s not in graph.  Skipping...' % i)
                            continue
                    if len(entries) > 1:
                        print(tabulate(entries, headers='firstrow') + '\n')

            elif cmdline[1] == 'masks':
                if len(cmdline) < 3:
                    print('Usage: show masks <node> <parent list (optional)>')
                    return

                node = int(cmdline[2])
                print('Node: %s' % node)
                if cmdline[-1] == 'matrix':
                    print(self.prop('masks')[node])
                    return
                entries = [['Parent', 'Perceived Opinion']]
                if len(cmdline) > 3:
                    parents = [int(i) for i in cmdline[3:]]
                else:
                    parents = sorted(list(self.get_neighbors(node)))

                for parent in parents:
                    if parent in list(self.get_neighbors(node)):
                        entries.append([parent, self.prop('masks')[node][parent]])
                    else:
                        print('No edge [%d]-->[%d]' % (parent, node))
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'reward':

                if len(cmdline) < 3:
                    print('Usage: show reward <node> <parent list (optional)>')
                    return

                node = int(cmdline[2])
                print('Node: %s' % node)
                entries = [['Parent', 'Reward']]
                if len(cmdline) > 3:
                    parents = [int(i) for i in cmdline[3:]]
                else:
                    parents = sorted(list(self.get_neighbors(node)))

                for parent in parents:
                    if parent in list(self.get_neighbors(node)):
                        entries.append([parent, self.get_reward_for_neighbor(node, parent)])
                    else:
                        print('No edge [%d]-->[%d]' % (parent, node))
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'properties':
                print('Currently recorded network properties:')
                for key in self._properties:
                    print('\t%s' % key)

            elif cmdline[1] == 'density':
                print('Density: %s' % nx.density(self._graph))

            elif cmdline[1] == 'local_average':
                for i in mylist:
                    try:
                        print(
                            'Average opinion in neighborhood of %s: %s' % (int(i), self.get_local_average(int(i))))
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue

            elif cmdline[1] == 'global_average':
                try:
                    print('Global average: %s' % self.get_global_average())
                except:
                    print('Error calculating global average.')
                    return

            elif cmdline[1] == 'type_average':
                if len(cmdline) == 2:
                    mylist = [key for key in self.prop('type_dist')]
                for t in mylist:
                    if t not in [key for key in self.prop('type_dist')]:
                        print('[%s] is not a known type.' % t)
                        continue
                    else:
                        print('Type average for [%s]: %s' % (t, self.get_global_average(t=t)))

            elif cmdline[1] == 'type_reward':
                if len(cmdline) == 2:
                    mylist = [key for key in self.prop('type_dist')]
                for t in mylist:
                    if t not in [key for key in self.prop('type_dist')]:
                        print('[%s] is not a known type.' % t)
                        continue
                    else:
                        rwds = self.reward(t=t)
                        print('Type average for [%s]: %s' % (t, stats.mean(rwds)))

            elif cmdline[1] == 'type_degree':
                if len(cmdline) == 2:
                    mylist = [key for key in self.prop('type_dist')]
                for t in mylist:
                    if t not in [key for key in self.prop('type_dist')]:
                        print('[%s] is not a known type.' % t)
                        continue
                    else:
                        mynodes = [node for node in range(self.prop('n')) if
                                   self.prop('types')[node] == t]
                        degrees = [self._graph.degree[node] for node in mynodes]
                        print('Average degree for [%s]: %s' % (t, stats.mean(degrees)))

            elif cmdline[1] == 'degree':
                entries = [['Node', 'Degree']]
                for i in mylist:
                    try:
                        entries.append([i, self._graph.degree[int(i)]])
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'friend_dist':
                entries = [['Node', 'Friend Type', 'Count']]
                for i in mylist:
                    try:
                        friends = {}
                        for nbr in self.get_neighbors(int(i)):
                            nbrtype = self.prop('types')[nbr]
                            if nbrtype not in friends:
                                friends[nbrtype] = 1
                            else:
                                friends[nbrtype] += 1
                        isthere = False
                        for t in friends:
                            if not isthere:
                                entries.append([i, t, friends[t]])
                                isthere = True
                            else:
                                entries.append(['', t, friends[t]])
                    except:
                        print('Node %s not in graph.  Skipping...' % i)
                        continue
                if len(entries) > 1:
                    print(tabulate(entries, headers='firstrow'))

            elif cmdline[1] == 'aliases':
                for key in self._temp_props['aliases']:
                    print('[ %s ] ---> [ %s ]' % (self._temp_props['aliases'][key], key))

            else:
                if cmdline[1] in self._properties:
                    myprop = self.prop(cmdline[1])
                    print('Property: %s' % cmdline[1])
                    if type(myprop) is DCT:
                        for key in myprop:
                            print('-- %s : %s' % (key, myprop[key]))
                    else:
                        print('-- %s' % myprop)
                else:
                    print('No known property [%s].' % cmdline[1])

        # Set individual or group data values
        elif cmdline[0] == 'set':

            if cmdline[1] == 'type':
                if cmdline[3] not in set([t for t in self.prop('type_dist')]):
                    print('[%s] is not a known type.' % cmdline[3])
                else:
                    self._properties['indexes_by_type'][self.prop('types')[int(cmdline[2])]].remove(int(cmdline[2]))
                    self._properties['indexes_by_type'][cmdline[3]].append(int(cmdline[2]))
                    self._properties['types'][int(cmdline[2])] = cmdline[3]
                    print('Set type of node %s to: %s' % (cmdline[2], cmdline[3]))

            elif cmdline[1] == 'resistance':
                if len(cmdline) == 3:
                    if 0 > float(cmdline[2]) or 1 < float(cmdline[2]):
                        print('Resistance values must fall between 0 and 1.')
                        return
                    for i in range(self.prop('n')):
                        self._properties['resistance'][i] = float(cmdline[2])
                    print('Set all resistance values to: %s' % cmdline[2])
                if len(cmdline) == 4:
                    if 0 < float(cmdline[3]) < 1:
                        print('Resistance values must fall between 0 and 1.')
                        return
                    try:
                        int(cmdline[2])
                        try:
                            self._properties['resistance'][int(cmdline[2])] = float(cmdline[3])
                            print('Set resistance value for node %s to: %s' % (cmdline[2], cmdline[3]))
                        except:
                            print('Node %s not in graph.  Skipping...' % cmdline[2])
                            return
                    except:
                        if cmdline[2] not in set([t for t in self.prop('type_dist')]):
                            print('[%s] is not a known type.' % cmdline[2])
                        idxs = self.prop('indexes_by_type')[cmdline[2]]
                        for i in idxs:
                            self._properties['resistance'][i] = float(cmdline[3])
                        print('Set all resistance values for nodes of type [%s] to: %s' % (cmdline[2], cmdline[3]))

            elif cmdline[1] == 'attr':
                try:
                    print('WARNING: No bounds checks are performed when resetting values.')
                    node = int(cmdline[2])
                    dim = int(cmdline[3])
                    val = float(cmdline[4])
                    self._properties['attribute_space'][node][dim] = val
                    for x in range(self.prop('n')):
                        if self.prop('masks')[x][node][dim] != 0.:
                            self._properties['masks'][x][node][dim] = val
                except:
                    print('Usage: set attr <node> <dimension> <value>')

            else:
                prop = self.prop(cmdline[1])
                mytype = type(prop)
                if mytype is BOOL:
                    try:
                        self._properties[cmdline[1]] = bool(cmdline[2])
                    except:
                        pass
                elif mytype is INT:
                    try:
                        self._properties[cmdline[1]] = int(cmdline[2])
                    except:
                        pass
                elif mytype is FLT:
                    try:
                        self._properties[cmdline[1]] = float(cmdline[2])
                    except:
                        pass
                else:
                    self._properties[cmdline[1]] = cmdline[2]

        elif cmdline[0] == 'rename':
            try:
                self._temp_props['aliases'][cmdline[2]] = cmdline[1]
                print('Renamed [ %s ] to [ %s ]' % (cmdline[1], cmdline[2]))
            except:
                print('Usage: rename <old name> <new name>')

        # Destroy an edge
        elif cmdline[0] == 'disconnect':
            success = self.disconnect(int(cmdline[1]), int(cmdline[2]), force=True)
            if success:
                print(f'Destroyed edge: [{cmdline[1]}]-->[{cmdline[2]}]')
            else:
                print('Could not destroy edge.')

        # Create an edge
        elif cmdline[0] == 'connect':
            success = self.connect(int(cmdline[1]), int(cmdline[2]), force=True)
            if success:
                print(f'Created edge: [{cmdline[1]}]-->[{cmdline[2]}]')
            else:
                print('Could not create edge.')

        # Reveal a topic from one node to another
        elif cmdline[0] == 'reveal':
            node1 = int(cmdline[1])
            node2 = int(cmdline[2])
            topic = int(cmdline[3])
            if node1 not in list(self.get_neighbors(node2)):
                print('Nodes %d and %d are not connected.' % (node1, node2))
                return
            self._properties['masks'][node2][node1][topic] = self.prop('attribute_space')[node1][topic]
            print('Node %s revealed topic %s to node %s' % (node1, topic, node2))

        # Hide a topic from one topic to another
        elif cmdline[0] == 'hide':
            node1 = int(cmdline[1])
            node2 = int(cmdline[2])
            topic = int(cmdline[3])
            if node1 not in list(self.get_neighbors(node2)):
                print('Nodes %d and %d are not connected.' % (node1, node2))
                return
            self._properties['masks'][node2][node1][topic] = 0.
            print('Node %s hid topic %s from node %s' % (node1, topic, node2))

        # Save the current graph to a specified filename
        elif cmdline[0] == 'save':
            try:
                self._save(cmdline[1])
            except:
                print('Problem saving file: %s' % cmdline[1])
        else:
            if cmdline[0] != 'q' and not frominherited:
                print('No known command: [%s].' % cmdline[0])

    def get_networkx_metric(self, metric, t=None):
        """

        :param metric:
        :param t:
        :return:
        """
        d = None
        if metric == 'degree':
            d = nx.degree_centrality(self._graph)
        elif metric == 'eigenvector':
            d = nx.eigenvector_centrality(self._graph)
        elif metric == 'betweenness':
            d = nx.betweenness_centrality(self._graph)
        elif metric == 'closeness':
            d = nx.closeness_centrality(self._graph)
        elif metric == 'clustering':
            d = nx.clustering(self._graph)
        elif metric == 'dispersion':
            dis = nx.dispersion(self._graph)
            d = {}
            for key in dis:
                mylist = [dis[key][subkey] for subkey in dis[key]]
                if not mylist: continue
                d[key] = stats.mean(mylist)

        if t is None:
            return [d[key] for key in d]
        else:
            return [d[key] for key in d if self._properties['types'][key] == t]

    def update_plot(self, objs, ax, axes, plot_data, sizes=None,
                    borders=None, static_pos=True, pos=None,
                    sizeby=None, friending='triadic'):
        """

        :param objs:
        :param plot_data:
        :param sizes:
        :param borders:
        :param static_pos:
        :param pos:
        :return:
        """
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        new_y = np.sin(x - 0.5 * random.random())

        removed = self.act()
        for (u, v) in removed:
            if (u, v) in objs['main']['edges']:
                ax.lines.remove(objs['main']['edges'][(u, v)])
                del objs['main']['edges'][(u, v)]
            if (v, u) in objs['main']['edges']:
                ax.lines.remove(objs['main']['edges'][(v, u)])
                del objs['main']['edges'][(v, u)]

        self.update()

        if friending == 'triadic':
            triadic = True
        else:
            triadic = False
        added = self.create_connections(triadic=triadic)
        if static_pos:
            for (u, v) in added:
                if (u,v) in objs['main']['edges']:
                    continue
                line, = ax.plot([plot_data['main']['x'][u], plot_data['main']['x'][v]],
                                [plot_data['main']['y'][u], plot_data['main']['y'][v]],
                                'gray', alpha=.25, zorder=-1)
                objs['main']['edges'][(u, v)] = line

        labels = [attr_str(self.prop('attribute_space')[i]) for i in range(self.prop('n'))]
        for i in range(len(plot_data['main']['texts'])):
            plot_data['main']['texts'][i].set_text(labels[i])

        if not static_pos:
            netx, nety, pos = self.get_xy()
            plot_data['main']['x'] = netx
            plot_data['main']['y'] = nety
            objs['main']['nodes'].set_offsets(np.c_[netx, nety])

            for i in range(len(plot_data['main']['texts'])):
                plot_data['main']['texts'][i].set_position((netx[i], nety[i]))

            edges = list(objs['main']['edges'])
            for (u, v) in edges:
                ax.lines.remove(objs['main']['edges'][(u, v)])
                del objs['main']['edges'][(u, v)]

            for (u, v) in self._graph.edges():
                line, = ax.plot([netx[u], netx[v]], [nety[u], nety[v]],
                                'gray', alpha=.25, zorder=-1)
                objs['main']['edges'][(u, v)] = line

        sizes = self.get_plot_sizes(sizeby=sizeby)
        objs['main']['nodes'].set_sizes(sizes)

        for obj in objs:
            if obj == 'main':
                continue
            myobjs = objs[obj]
            new_data = list()
            if obj == 'global_average':
                new_data = self.get_global_average()
            if obj == 'density':
                new_data = [nx.density(self._graph)]
            for i in range(len(new_data)):
                plot_data[obj][i].append(new_data[i])
            #if obj == 'density':
            #    plot_data[obj][0].append(nx.density(self._graph))
            #    new_y = plot_data[obj][0]
            #    x = list(range(len(new_y)))
            #    axes[obj].set_xlim(0, len(x))
            #    axes[obj].set_ylim(0, max(new_y))

            for i in range(len(plot_data[obj])):
                x = list(range(len(plot_data[obj][0])))
                y = plot_data[obj][i]
                objs[obj][i].set_xdata(x)
                objs[obj][i].set_ydata(y)

            axes[obj].relim()
            # update ax.viewLim using the new dataLim
            axes[obj].autoscale_view()

    def get_xy(self, initial_pos=None):
        """

        :param initial_pos:
        :return:
        """
        x, y = list(), list()
        pos = nx.spring_layout(self._graph, pos=initial_pos)
        for key in pos:
            x.append(pos[key][0])
            y.append(pos[key][1])
        return x, y, pos

    def get_plot_colors(self):
        """

        :return:
        """
        colors = list()
        for i in range(self.prop('n')):
            colors.append(AGENT_PROPS[self.prop('types')[i]]['color'])
        return colors

    def get_plot_sizes(self, sizeby=None):
        """

        :return:
        """
        
        mydict = {}
        if sizeby is None:
            return [300 for i in range(self.prop('n'))]
        elif sizeby == 'closeness':
            mydict = self.get_networkx_metric('closeness')
        elif sizeby == 'betweenness':
            mydict = self.get_networkx_metric('betweenness')
        elif sizeby == 'degree':
            mydict = self.get_networkx_metric('degree')
        elif sizeby == 'eigenvector':
            mydict = self.get_networkx_metric('eigenvector')
        elif sizeby == 'reward':
            mydict = self.reward()
        return [mydict[i] * 750 + 100 for i in range(self.prop('n'))]

    def plot_network(self, ax, plot_objects, plot_data):

        sizeby = 'reward'

        plot_objects['main'] = dict()

        x, y, pos = self.get_xy()
        plot_data['main']['x'] = x
        plot_data['main']['y'] = y
        plot_data['main']['texts'] = list()
        colors = self.get_plot_colors()
        sizes = self.get_plot_sizes(sizeby=sizeby)

        edges = dict()
        for (u, v) in self._graph.edges():
            line, = ax.plot([x[u], x[v]], [y[u], y[v]], 'gray', alpha=.25, zorder=-1)
            edges[(u, v)] = line
        sc = ax.scatter(x, y, c=colors, s=sizes, alpha=.75)
        labels = [attr_str(self.prop('attribute_space')[i]) for i in range(self.prop('n'))]
        for i in range(len(labels)):
            plot_data['main']['texts'].append(ax.text(x[i], y[i], labels[i]))

        plot_objects['main']['nodes'] = sc
        plot_objects['main']['edges'] = edges

    def plot_metric(self, metric, ax, plot_objects, plot_data):
        plot_data[metric] = []
        plot_objects[metric] = []
        if metric == 'density':
            plot_data[metric].append([nx.density(self._graph)])
        if metric == 'global_average':
            plot_data['global_average'] = []
            ga = self.get_global_average()
            for i in range(len(ga)):
                plot_data['global_average'].append([ga[i]])
            x = range(len(plot_data['global_average'][0]))
        for i in plot_data[metric]:
            line, = ax.plot(list(range(len(i))), i)
            plot_objects[metric].append(line)

    def animate(self, numsteps=1, interactive=False,
                colors='type', sizes=None, borders=None,
                plots=[], static_pos=True, sizeby=None,
                friending='triadic'):
        """

        :param numsteps:
        :param interactive:
        :param colors:
        :param sizes:
        :param borders:
        :param plots:
        :param static_pos:
        :return:
        """

        # Initialize figure, a reference for Axes objects, a reference for plotted objects,
        # and a reference for past plotted data.
        fig = plt.figure(figsize=(10, 7))
        axes = dict()
        plot_objects, plot_data = dict(), dict()

        # Initialize subplots
        numplots = len(plots)
        plot_data['main'] = {'x': None, 'y': None, 'texts': None}
        if numplots == 0:
            axes['main'] = plt.subplot(1, 1, 1)
        else:
            axes['main'] = plt.subplot(2, 1, 2)
            for i in range(numplots):
                axes[plots[i]] = plt.subplot(2, numplots, i+1)
                plot_data[plots[i]] = dict()

        # Construct the initial network plot
        self.plot_network(axes['main'], plot_objects, plot_data)
        for p in plots:
            self.plot_metric(p, axes[p], plot_objects, plot_data)

        #y = [nx.density(self._graph)]
        #x = list(range(len(y)))

        #for i in range(numplots):
        #    line, = axes['ax%d' % i].plot(x, y)
        #    plot_objects['line%d' % i] = line

        plt.ion()
        plt.show()

        countdown = 1

        isdone = False
        step = 0
        while not isdone:

            if countdown > 0:
                countdown -= 1
            elif interactive:
                usr = input('ENTER to continue, q to quit, or enter the number of steps to execute.')
                if usr == 'q':
                    break
                elif usr != '':
                    countdown = int(usr)

            # updating data values
            fig.canvas.flush_events()
            self.update_plot(plot_objects, axes['main'], axes, plot_data, static_pos=static_pos,
                             sizeby=sizeby, friending=friending)

            # drawing updated values
            # fig.canvas.draw()
            fig.canvas.draw_idle()

            plt.pause(.1)

            if (not interactive) and (step == numsteps):
                isdone = True

            step += 1

        plt.ioff()
        plt.show()
