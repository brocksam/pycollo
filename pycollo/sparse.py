import collections

import sympy as sym

from .node import Node



class SparseCOOMatrix:

	ZERO = sym.core.numbers.Zero()

	def __new__(cls, entries, n_rows, n_cols, expr_graph):
		self = object.__new__(cls)
		self._shape = [n_rows, n_cols]
		self._entries = entries
		self._expr_graph = expr_graph
		return self

	@classmethod
	def _new(cls, *args):
		return cls(*args)

	@property
	def shape(self):
		return self._shape

	@property
	def entries(self):
		return self._entries

	@property
	def num_rows(self):
		return self._shape[0]
	
	@property
	def num_cols(self):
		return self._shape[1]

	@property
	def nnz(self):
		return self._nnz

	@property
	def free_symbols(self):
		free_symbols = {v for v in self._entries.values()}
		return free_symbols

	def sort(self):
		self._entries = {i: self._entries[i] for i in sorted(self._entries.keys())}

	def make_lower_triangular(self):
		self.sort()
		entries = {}
		for (row_ind, col_ind), value in self._entries.copy().items():
			if row_ind >= col_ind:
				entries[(row_ind, col_ind)] = value
			else:
				entries[(col_ind, row_ind)] = value
		new = self._new(entries, self.num_rows, self.num_cols, self._expr_graph)
		new.sort()
		return new

	def scalar_multiply(self, scalar):
		entries = {}
		for index, value in self._entries.copy().items():
			entries[index] = Node(sym.Mul(value, scalar), self._expr_graph).symbol
		return self._new(entries, self.num_rows, self.num_cols, self._expr_graph)

	def vector_premultiply(self, pre_vector):
		if len(pre_vector) != self.num_rows:
			msg = (f"Vector for pre-multiplication must have same number of "
				f"entries as rows in matrix.")
			raise ValueError(msg)

		# Create quick lookup for pre-vector and sparse matrix
		vector_lookup = collections.defaultdict(dict)
		for i, val in enumerate(pre_vector):
			vector_lookup[i] = val
		matrix_lookup = collections.defaultdict(dict)
		for (i, j), val in self._entries.items():
			matrix_lookup[j][i] = val
		vector_indices = set(k for k in vector_lookup.keys())

		smat = {}
		for col_ind, col in matrix_lookup.items():
			indices = vector_indices & set(col.keys())
			if indices:
				val = sym.Add(*[Node(sym.Mul(vector_lookup[i], matrix_lookup[col_ind][i]), self._expr_graph).symbol 
					for i in indices])
				smat[0, col_ind] = val

		return self._new(smat, 1, self.num_cols, self._expr_graph)

	def to_dense_sympy_matrix(self):
		return sym.Matrix(sym.SparseMatrix(*self._shape, self._entries))

	def get_subset(self, row_slice, col_slice):
		if row_slice is None:
			row_slice = slice(0, self.num_rows)
		if col_slice is None:
			col_slice = slice(0, self.num_cols)
		smat = {}
		for (i, j), val in self._entries.items():
			in_row_range = i >= row_slice.start and i < row_slice.stop
			in_col_range = j >= col_slice.start and j < col_slice.stop
			if in_row_range and in_col_range:
				smat[i, j] = val

		return self._new(smat, self.num_rows, self.num_cols, self._expr_graph)

	def __iter__(self):
		return iter(self._entries)

	def __add__(self, other):
		if not isinstance(other, self.__class__):
			raise NotImplementedError 
		smat = {}
		for key in set().union(self._entries.keys(), other._entries.keys()):
			sum = self._entries.get(key, self.ZERO) + other._entries.get(key, self.ZERO)
			if sum != 0:
				smat[key] = sum
	  	# Add new nodes to expression graph and return new sparse matrix with
	  	# nodes as nonzero entries
		new = self._new(smat, *self._shape, self._expr_graph)
		return new

	def __mul__(self, other):
		"""Fast multiplication exploiting the sparsity of the matrix."""
		if not isinstance(other, self.__class__):
			if other == 1:
				return self
			raise NotImplementedError

		# if we made it here, we're both sparse matrices
		# create quick lookups for rows and cols
		row_lookup = collections.defaultdict(dict)
		for (i,j), val in self._entries.items():
			row_lookup[i][j] = val
		col_lookup = collections.defaultdict(dict)
		for (i,j), val in other._entries.items():
			col_lookup[j][i] = val

		smat = {}
		for row in row_lookup.keys():
			for col in col_lookup.keys():
				# find the common indices of non-zero entries.
				# these are the only things that need to be multiplied.
				indices = set(col_lookup[col].keys()) & set(row_lookup[row].keys())
				if indices:

					val = sum(Node(row_lookup[row][k]*col_lookup[col][k], self._expr_graph).symbol for k in indices)
					smat[row, col] = val
		new = self._new(smat, self.num_rows, other.num_cols, self._expr_graph)
		return new
