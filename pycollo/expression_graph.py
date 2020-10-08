import abc
import collections
import functools
import itertools
import numbers
from timeit import default_timer as timer
import weakref

import numpy as np
import sympy as sym

from .node import Node
from .sparse import SparseCOOMatrix
from .utils import (console_out, dict_merge)

"""

Notes:
------
	Todo - add checking for:
	* Auxiliary data
	* Objective function
	* State equations
	* Path constraints
	* Integrand functions
	* State endpoint constraints
	* Endpoint constraints

	Todo - use sympy.matrix.sparse in hSAD

	Optimisations:
	* Add handling for _n0 vs 0 when making L matricies lower triangular
	
"""

class ExpressionGraph:

    def __init__(self, ocp_backend, problem_variables, objective, constraints,
                 auxiliary_information):

        self.ocp_backend = ocp_backend
        self.phases = ocp_backend.p
        self.objective = objective
        self.constraints = constraints
        self.console_out_begin_expression_graph_creation()
        self.initialise_node_symbol_number_counters()
        self.initialise_node_mappings()
        self.initialise_problem_variable_information(problem_variables)
        self.initialise_default_singleton_number_nodes()
        self.initialise_auxiliary_constant_nodes(auxiliary_information)
        self.initialise_time_normalisation_nodes()
        self.initialise_auxiliary_intermediate_nodes()

    def console_out_begin_expression_graph_creation(self):
        if self.ocp_backend.ocp.settings.console_out_progress:
            msg = (f"Beginning expression graph creation.")
            console_out(msg)

    def initialise_node_symbol_number_counters(self):
        self._number_node_num_counter = itertools.count()
        self._constant_node_num_counter = itertools.count()
        self._intermediate_node_num_counter = itertools.count()

    def initialise_node_mappings(self):
        self._node_syms = set()
        self._variable_nodes = {}
        self._number_nodes = {}
        self._constant_nodes = {}
        self._intermediate_nodes = {}
        self._precomputable_nodes = {}

    def initialise_problem_variable_information(self, x_vars):
        self.initialise_problem_variable_attributes(x_vars)
        self.initialise_problem_variable_nodes()

    def initialise_problem_variable_attributes(self, x_vars):
        x_continuous, x_endpoint = x_vars

        self.problem_variables_continuous_ordered = x_continuous
        self.problem_variables_endpoint_ordered = x_endpoint
        self.problem_variables_ordered = x_continuous + x_endpoint
        self.problem_variables_continuous_set = set(x_continuous)
        self.problem_variables_endpoint_set = set(x_endpoint)
        self.problem_variables_set = set(x_continuous + x_endpoint)

        self.lagrange_syms = ()

    def initialise_problem_variable_nodes(self):
        self.time_function_variable_nodes = set()

        self._continuous_variable_nodes = []
        for x_var in self.problem_variables_continuous:
            node = Node(x_var, self)
            self._continuous_variable_nodes.append(node)
            if str(node.symbol)[1] in {"y", "u"}:
                self.time_function_variable_nodes.add(node)

        self._endpoint_variable_nodes = []
        for x_b_var in self.problem_variables_endpoint:
            node = Node(x_b_var, self)
            self._endpoint_variable_nodes.append(node)

    def initialise_default_singleton_number_nodes(self):
        self._zero_node = Node(0, self)
        self._one_node = Node(1, self)
        two_node = Node(2, self)
        neg_one_node = Node(-1, self)
        half_node = Node(0.5, self)

    def initialise_auxiliary_constant_nodes(self, aux_info):
        self.user_symbol_to_expression_auxiliary_mapping = {}
        self._user_constants_ordered = tuple()
        self._user_constants_set = set()
        for key, value in aux_info.items():
            is_expression = isinstance(value, (sym.Expr, sym.Symbol))
            if is_expression and (not value.is_Number):
                self.user_symbol_to_expression_auxiliary_mapping[key] = value
            else:
                self._user_constants_set.add(key)
                node = Node(key, self, value=value)

    def initialise_time_normalisation_nodes(self):
        self._t_norm_nodes = tuple(Node(p.t_norm, self)
                                   for p in self.phases)

    def initialise_auxiliary_intermediate_nodes(self):
        iterable = self.user_symbol_to_expression_auxiliary_mapping.items()
        for node_symbol, node_expr in iterable:
            _ = Node(node_symbol, self, equation=node_expr)

    def form_functions_and_derivatives(self):
        self._form_time_normalisation_functions()
        self._form_objective_function_and_derivatives()
        self._form_constraints_and_derivatives()
        if self.ocp_backend.ocp.settings.derivative_level == 2:
            self._form_lagrangian_and_derivatives()

    def _form_time_normalisation_functions(self):
        for p in self.phases:
            self._form_function_and_derivative(
                func=p.t_norm,
                wrt=None,
                derivative=False,
                hessian=False,
                func_abrv=f"t_norm_P{p.i}",
                init_func=True,
                completion_msg=f"time normalisation of phase #{p.i}",
            )

    def _form_objective_function_and_derivatives(self):
        self._form_function_and_derivative(
            func=self.objective,
            wrt=self._endpoint_variable_nodes,
            derivative=True,
            hessian=False,
            func_abrv="J",
            init_func=True,
            completion_msg="objective gradient",
        )

    def _form_constraints_and_derivatives(self):

        def form_continuous(continuous_constraints):
            form_function_and_derivative(func=continuous_constraints,
                                         wrt=self._continuous_variable_nodes, func_abrv="c",
                                         completion_msg="Jacobian of the continuous constraints")

        def form_endpoint(endpoint_constraints):
            form_function_and_derivative(func=endpoint_constraints,
                                         wrt=self._endpoint_variable_nodes, func_abrv="b",
                                         completion_msg="Jacobian of the endpoint constraints")

        form_function_and_derivative = functools.partial(
            self._form_function_and_derivative, derivative=True, hessian=False, init_func=True)

        continuous_constraints = sym.Matrix(
            self.constraints[self.ocp_backend.c_continuous_slice])
        endpoint_constraints = sym.Matrix(
            self.constraints[self.ocp_backend.c_endpoint_slice])

        form_continuous(continuous_constraints)
        form_endpoint(endpoint_constraints)

    def _form_lagrangian_and_derivatives(self):

        form_function_and_derivative = functools.partial(
            self._form_function_and_derivative, derivative=True, hessian=True, init_func=False)

        sigma = sym.symbols("_sigma")
        self.ocp_backend.sigma_sym = sigma

        L_syms = tuple(sym.symbols(f"_lambda_{n}") for n in range(self.ocp_backend.num_c))
        self.ocp_backend.lagrange_syms = L_syms

        self.lagrange_syms = tuple((sigma, ) + L_syms)
        for L_sym in self.lagrange_syms:
            _ = Node(L_sym, self)

        dL_dxb_objective = self.dJ_dxb.scalar_multiply(sigma)
        L_syms_endpoint = L_syms[self.ocp_backend.c_endpoint_slice]
        dL_dxb_endpoint = self.db_dxb.vector_premultiply(L_syms_endpoint)
        dL_dxb = dL_dxb_objective + dL_dxb_endpoint
        dL_dxb_terms = dL_dxb.to_dense_sympy_matrix()
        form_function_and_derivative(func=dL_dxb_terms,
                                     wrt=self._endpoint_variable_nodes, func_abrv="L",
                                     completion_msg=f"Hessian of the endpoint Lagrangian")
        self.ddL_dxbdxb = self.ddL_dxbdxb.make_lower_triangular()

        L_syms_continuous_time_stretched = []
        for p_c_defect_slice, p_c_path_slice, p_c_integral_slice, t_norm_node in zip(
            self.ocp_backend.phase_defect_constraint_slices,
            self.ocp_backend.phase_path_constraint_slices,
            self.ocp_backend.phase_integral_constraint_slices,
            self._t_norm_nodes,
        ):
            terms = [Node(sym.Mul(t_norm_node.symbol, L_sym), self).symbol
                     for L_sym in L_syms[p_c_defect_slice]]
            L_syms_continuous_time_stretched.extend(terms)
            terms = [L_sym for L_sym in L_syms[p_c_path_slice]]
            L_syms_continuous_time_stretched.extend(terms)
            terms = [Node(sym.Mul(t_norm_node.symbol, L_sym), self).symbol
                     for L_sym in L_syms[p_c_integral_slice]]
            L_syms_continuous_time_stretched.extend(terms)

        dL_dx_continuous = self.dc_dx.vector_premultiply(
            L_syms_continuous_time_stretched)
        dL_dx_terms = dL_dx_continuous.to_dense_sympy_matrix()
        form_function_and_derivative(func=dL_dx_terms,
                                     wrt=self._continuous_variable_nodes, func_abrv="L",
                                     completion_msg=f"Hessian of the continuous Lagrangian")
        self.ddL_dxdx = self.ddL_dxdx.make_lower_triangular()
        self.ddL_dxdx_nodes = self.ddL_dxdx_nodes.make_lower_triangular()

        portions_requiring_summing = {}
        for p, p_var_slice in zip(self.ocp_backend.p, self.ocp_backend.phase_variable_slices):
            offset = p_var_slice.start
            p_qt_slice = slice(p.qt_slice.start + offset,
                               p.qt_slice.stop + offset)
            portions_requiring_summing.update({**self.ddL_dxdx.get_subset(self.ocp_backend.variable_slice, p_qt_slice).entries})
        portions_requiring_summing.update({**self.ddL_dxdx.get_subset(self.ocp_backend.variable_slice, self.ocp_backend.variable_slice).entries})
        final_nodes = set(Node(symbol, self)
                          for symbol in portions_requiring_summing.values())

        ddL_dxdx_dependent_nodes = set(
            node for tier in self.ddL_dxdx_dependent_tiers.values() for node in tier)
        nodes_requiring_summing = set()

        def requires_summing(node):
            if node in final_nodes:
                summing_required = True
            else:
                summing_required = any(requires_summing(child)
                                       for child in node.child_nodes)
            if summing_required:
                nodes_requiring_summing.add(node)
            return summing_required

        for L in L_syms:
            _ = requires_summing(Node(L, self))

        self.ddL_dxdx_sum_nodes = nodes_requiring_summing.difference(
            set(Node(symbol, self) for symbol in L_syms))

        L_nodes = set(Node(L, self) for L in L_syms)
        nodes_requiring_summing = set()

        def requires_summing(node):
            if node in L_nodes:
                return True
            try:
                if any([requires_summing(node) for node in node.parent_nodes]):
                    nodes_requiring_summing.add(node)
                    return True
                else:
                    return False
            except NameError:
                return False

        for node in final_nodes:
            requires_summing(node)

        self.ddL_dxdx_sum_nodes = nodes_requiring_summing

    def _form_function_and_derivative(self, func, wrt, derivative, hessian, func_abrv, init_func, completion_msg=None):

        def create_derivative_abbreviation(wrt, func_abrv):
            if wrt is self._continuous_variable_nodes:
                wrt_abrv = "x"
            elif wrt is self._endpoint_variable_nodes:
                wrt_abrv = "xb"

            if hessian:
                return f"dd{func_abrv}_d{wrt_abrv}d{wrt_abrv}"
            else:
                return f"d{func_abrv}_d{wrt_abrv}"

        def add_to_namespace(self, args, func_abrv):
            setattr(self, f"{func_abrv}", args[0])
            setattr(self, f"{func_abrv}_nodes", args[1])
            setattr(self, f"{func_abrv}_precomputable", args[2])
            setattr(self, f"{func_abrv}_dependent_tiers", args[3])
            return self

        init_args = self._initialise_function(func)
        temp = init_args[0]

        if init_func is True:
            self = add_to_namespace(self, init_args, func_abrv)

        if derivative:
            deriv = self.hybrid_symbolic_algorithmic_differentiation(
                *init_args, wrt)
            init_args = self._initialise_function(deriv)
            deriv_abrv = create_derivative_abbreviation(wrt, func_abrv)
            self = add_to_namespace(self, init_args, deriv_abrv)

        if completion_msg is not None:
            completion_msg = f"Symbolic {completion_msg} calculated."
            console_out(completion_msg)

        return init_args

    def _initialise_function(self, expr):

        def substitute_function_for_root_symbols(expr):

            def traverse_root_branch(expr, max_tier):
                root_node = self.get_node_from_expr(expr)
                max_tier = max(max_tier, root_node.tier)
                return (root_node.symbol, root_node, max_tier)

            if isinstance(expr, sym.Expr):
                return_vals = traverse_root_branch(expr, 0)
                root_symbol, root_node, max_tier = return_vals
                return (root_symbol, [root_node], max_tier)
            elif isinstance(expr, SparseCOOMatrix):
                max_tier = 0
                return_expr_entries = {}
                return_node_entries = {}
                expr.sort()
                for index, value in expr.entries.copy().items():
                    return_vals = traverse_root_branch(value, max_tier)
                    root_symbol, root_node, max_tier = return_vals
                    return_expr_entries[index] = root_symbol
                    return_node_entries[index] = root_node
                return_matrix = SparseCOOMatrix(return_expr_entries, *expr.shape, self)
                expr_nodes = SparseCOOMatrix(return_node_entries, *expr.shape, self)
                return (return_matrix, expr_nodes, max_tier)
            else:
                expr_subbed = []
                expr_nodes = []
                max_tier = 0
                for entry_expr in expr:
                    return_vals = traverse_root_branch(entry_expr, max_tier)
                    root_symbol, root_node, max_tier = return_vals
                    expr_subbed.append(root_symbol)
                    expr_nodes.append(root_node)
                return_matrix = sym.Matrix(np.array(expr_subbed).reshape(
                    expr.shape))
                return (return_matrix, expr_nodes, max_tier)

        def separate_precomputable_and_dependent_nodes(expr, nodes):
            precomputable_nodes = set()
            dependent_nodes = set()
            if isinstance(expr, SparseCOOMatrix):
                all_nodes = nodes.free_symbols
            else:
                all_nodes = set(nodes)
            for free_symbol in expr.free_symbols:
                node = self.symbols_to_nodes_mapping[free_symbol]
                all_nodes.update(node.dependent_nodes)
            precomputable_nodes = set()
            dependent_nodes = set()
            for node in all_nodes:
                if node.is_precomputable:
                    precomputable_nodes.add(node)
                else:
                    dependent_nodes.add(node)
            return (precomputable_nodes, dependent_nodes)

        def sort_dependent_nodes_by_tier(dependent_nodes, max_tier):
            dependent_tiers = {i: set() for i in range(max_tier + 1)}
            for node in dependent_nodes:
                dependent_tiers[node.tier].add(node)
            return dependent_tiers

        def check_root_tier_is_exlusively_continuous_or_endpoint(
                dependent_tiers):
            pass

        return_vals = substitute_function_for_root_symbols(expr)
        expr_subbed, expr_nodes, max_tier = return_vals
        return_vals = separate_precomputable_and_dependent_nodes(
            expr_subbed, expr_nodes)
        precomputable_nodes, dependent_nodes = return_vals
        dependent_tiers = sort_dependent_nodes_by_tier(dependent_nodes,
                                                       max_tier)
        check_root_tier_is_exlusively_continuous_or_endpoint(dependent_tiers)
        return expr_subbed, expr_nodes, precomputable_nodes, dependent_tiers

    @property
    def variable_nodes(self):
        return tuple(self._variable_nodes.values())

    @property
    def constant_nodes(self):
        return tuple(self._constant_nodes.values())

    @property
    def number_nodes(self):
        return tuple(self._number_nodes.values())

    @property
    def intermediate_nodes(self):
        return tuple(self._intermediate_nodes.values())

    @property
    def root_nodes(self):
        return self.variable_nodes + self.constant_nodes + self.number_nodes

    @property
    def nodes(self):
        return self.root_nodes + self.intermediate_nodes

    @property
    def precomputable_nodes(self):
        return tuple(self._precomputable_nodes.values())

    def get_node_from_expr(self, expr):
        node = self._variable_nodes.get(expr)
        if node is not None:
            return node
        node = self._constant_nodes.get(expr)
        if node is not None:
            return node
        node = self._number_nodes.get(expr)
        if node is not None:
            return node
        node = self._intermediate_nodes.get(expr)
        if node is not None:
            return node
        return Node(expr, self)

    @property
    def symbols_to_nodes_mapping(self):
        return {
            **self._variable_nodes,
            **self._constant_nodes,
            **self._number_nodes,
            **self._intermediate_nodes,
        }

    def hybrid_symbolic_algorithmic_differentiation(self, target_function,
                                                    function_nodes, precomputable_nodes, dependent_nodes_by_tier, wrt):

        def differentiate(function_nodes, wrt_nodes):
            n_rows = len(function_nodes)
            n_cols = len(wrt_nodes)
            wrt_mapping = {wrt_node: i for i, wrt_node in enumerate(wrt_nodes)}
            nonzeros = {}
            for i_row, node in enumerate(function_nodes):
                diff_nodes = node.differentiable_by
                for wrt in diff_nodes:
                    i_col = wrt_mapping.get(wrt)
                    if i_col is not None:
                        nonzeros[(i_row, i_col)
                                 ] = node.derivative_as_symbol(wrt)
            return SparseCOOMatrix(nonzeros, n_rows, n_cols, self)
            # return sym.SparseMatrix(n_rows, n_cols, nonzeros)

        def compute_target_function_derivatives_for_each_tier(
                dependent_nodes_by_tier_collapsed):
            df_de = []
            for node_tier in dependent_nodes_by_tier_collapsed:
                derivative = differentiate(function_nodes, node_tier)
                df_de.append(derivative)
            return df_de

        def compute_delta_matrices_for_each_tier(num_e0,
                                                 dependent_nodes_by_tier_collapsed):
            delta_matrices = [1]
            for tier_num, dependent_nodes_tier in enumerate(
                    dependent_nodes_by_tier_collapsed[1:], 1):
                num_ei = len(dependent_nodes_tier)
                delta_matrix_i = SparseCOOMatrix({}, num_ei, num_e0, self)
                # delta_matrix_i = sym.SparseMatrix(num_ei, num_e0, {})
                for by_tier_num in range(tier_num):
                    delta_matrix_j = delta_matrices[by_tier_num]
                    deriv_matrix = differentiate(dependent_nodes_tier,
                                                 dependent_nodes_by_tier_collapsed[by_tier_num])
                    delta_matrix_i += deriv_matrix * delta_matrix_j
                delta_matrices.append(delta_matrix_i)
            return delta_matrices

        def compute_derivative_recursive_hSAD_algorithm():
            num_f = len(function_nodes)
            derivative = SparseCOOMatrix({}, num_f, num_e0, self)
            # derivative = sym.SparseMatrix(num_f, num_e0, {})
            for df_dei, delta_i in zip(df_de, delta_matrices):
                # TO DO: Understand why this is required
                if df_dei.shape != (0, 0):
                    derivative += df_dei * delta_i
            return derivative

        dependent_nodes_by_tier_collapsed = [wrt]
        for nodes in list(dependent_nodes_by_tier.values())[1:]:
            if nodes:
                dependent_nodes_by_tier_collapsed.append(tuple(nodes))

        df_de = compute_target_function_derivatives_for_each_tier(
            dependent_nodes_by_tier_collapsed)

        num_e0 = len(dependent_nodes_by_tier_collapsed[0])
        delta_matrices = compute_delta_matrices_for_each_tier(num_e0,
                                                              dependent_nodes_by_tier_collapsed)

        derivative = compute_derivative_recursive_hSAD_algorithm()

        return derivative

    def __str__(self):
        cls_name = self.__class__.__name__
        return (f"{cls_name}(({self.problem_variables_continuous}, "
                f"{self.problem_variables_endpoint}))")

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (f"{cls_name}(problem_variables="
                f"({self.problem_variables_continuous}, "
                f"{self.problem_variables_endpoint}))")


def kill():
    print("\n\n")
    raise ValueError


def cout(*args):
    print("\n\n")
    for arg in args:
        print(f"{arg}\n")
