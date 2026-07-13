"""
Vendored core algorithm classes from PrivSyn (Zhang, Wang, Li, Honorio, Backes, He,
Chen, Zhang. "PrivSyn: Differentially Private Data Synthesis", USENIX Security 2021,
https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf).

Copied (with only trivial import-path adjustments) from the reference implementation
bundled in the SynMeter benchmark repo (https://github.com/zealscott/SynMeter,
Apache License 2.0), file paths:
    synthesizer/syn/view.py
    synthesizer/syn/consistent.py
    synthesizer/syn/record_synthesizer.py

These three classes implement the "GUM" (Gradually Update Method) record-synthesis
algorithm and the marginal-consistency ("Consistenter") step. They operate purely on
numpy arrays of integer-encoded records/domains and noisy marginal counts, with no
dependency on SynMeter's own DataLoader/DataTransformer config-file machinery — so
they are vendored standalone here. The DP marginal *measurement* (noise calibration,
marginal selection, budget splitting) is NOT part of this file; that is implemented
fresh in sdg/privsyn_method.py using the same zCDP-Gaussian-mechanism convention
already used elsewhere in this repo (sdg/mwem_pgm_method.py).
"""

import copy

import numpy as np
from numpy import linalg as LA


class View:
    def __init__(self, attr_one_hot: np.array, domain_size_list: np.array):
        self.attr_one_hot = attr_one_hot
        self.domain_size_list = domain_size_list

        self.domain_size = np.product(self.domain_size_list[np.nonzero(self.attr_one_hot)[0]])
        self.total_num_attr = len(self.attr_one_hot)
        self.view_num_attr = np.count_nonzero(self.attr_one_hot)

        self.encode_num = np.zeros(self.view_num_attr, dtype=np.uint32)
        self.cum_mul = np.zeros(self.view_num_attr, dtype=np.uint32)
        self.attributes_index = np.nonzero(self.attr_one_hot)[0]

        self.count = np.zeros(self.domain_size)
        self.sum = 0
        self.calculate_encode_num(self.domain_size_list)

        self.attributes_set = set()
        self.tuple_key = np.array([0], dtype=np.uint32)
        self.count_matrix = None
        self.summations = None
        self.weights = []
        self.delta = 0
        self.weight_coeff = 1

    def calculate_encode_num(self, domain_size_list):
        if self.view_num_attr != 0:
            categories_index = self.attributes_index

            categories_num = domain_size_list[categories_index]
            categories_num = np.roll(categories_num, 1)
            categories_num[0] = 1
            self.cum_mul = np.cumprod(categories_num)

            categories_num = domain_size_list[categories_index]
            categories_num = np.roll(categories_num, self.view_num_attr - 1)
            categories_num[-1] = 1
            categories_num = np.flip(categories_num)
            self.encode_num = np.flip(np.cumprod(categories_num))

    def calculate_tuple_key(self):
        self.tuple_key = np.zeros([self.domain_size, self.view_num_attr], dtype=np.uint32)

        if self.view_num_attr != 0:
            for i in range(self.attributes_index.shape[0]):
                index = self.attributes_index[i]
                categories = np.arange(self.domain_size_list[index])
                column_key = np.tile(np.repeat(categories, self.encode_num[i]), self.cum_mul[i])

                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.domain_size = 1

    def count_records(self, records):
        encode_records = np.matmul(records[:, self.attributes_index], self.encode_num)
        encode_key, count = np.unique(encode_records, return_counts=True)

        indices = np.where(np.isin(np.arange(self.domain_size), encode_key))[0]
        self.count[indices] = count

    def calculate_count_matrix(self):
        shape = []

        for attri in self.attributes_index:
            shape.append(self.domain_size_list[attri])

        self.count_matrix = np.copy(self.count).reshape(tuple(shape))

        return self.count_matrix

    def generate_attributes_index_set(self):
        self.attributes_set = set(self.attributes_index)

    def calculate_encode_num_general(self, attributes_index):
        categories_index = attributes_index

        categories_num = self.domain_size_list[categories_index]
        categories_num = np.roll(categories_num, attributes_index.size - 1)
        categories_num[-1] = 1
        categories_num = np.flip(categories_num)
        encode_num = np.flip(np.cumprod(categories_num))

        return encode_num

    def count_records_general(self, records):
        count = np.zeros(self.domain_size)

        encode_records = np.matmul(records[:, self.attributes_index], self.encode_num)
        encode_key, value_count = np.unique(encode_records, return_counts=True)

        indices = np.where(np.isin(np.arange(self.domain_size), encode_key))[0]
        count[indices] = value_count

        return count

    def calculate_count_matrix_general(self, count):
        shape = []

        for attri in self.attributes_index:
            shape.append(self.domain_size_list[attri])

        return np.copy(count).reshape(tuple(shape))

    def calculate_tuple_key_general(self, unique_value_list):
        self.tuple_key = np.zeros([self.domain_size, self.view_num_attr], dtype=np.uint32)

        if self.view_num_attr != 0:
            for i in range(self.attributes_index.shape[0]):
                categories = unique_value_list[i]
                column_key = np.tile(np.repeat(categories, self.encode_num[i]), self.cum_mul[i])

                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.domain_size = 1

    def project_from_bigger_view_general(self, bigger_view):
        encode_num = np.zeros(self.total_num_attr, dtype=np.uint32)
        encode_num[self.attributes_index] = self.encode_num
        encode_num = encode_num[bigger_view.attributes_index]

        encode_records = np.matmul(bigger_view.tuple_key, encode_num)

        for i in range(self.domain_size):
            key_index = np.where(encode_records == i)[0]
            self.count[i] = np.sum(bigger_view.count[key_index])

    def initialize_consist_parameters(self, num_target_views):
        self.summations = np.zeros([self.domain_size, num_target_views])
        self.weights = np.zeros(num_target_views)

    def calculate_delta(self):
        target = np.matmul(self.summations, self.weights) / np.sum(self.weights)
        self.delta = -(self.summations - target.reshape(len(target), 1))

    def project_from_bigger_view(self, bigger_view, index):
        encode_num = np.zeros(self.total_num_attr, dtype=np.uint32)
        encode_num[self.attributes_index] = self.encode_num
        encode_num = encode_num[bigger_view.attributes_index]

        encode_records = np.matmul(bigger_view.tuple_key, encode_num)

        self.weights[index] = bigger_view.weight_coeff / np.product(
            self.domain_size_list[np.setdiff1d(bigger_view.attributes_index, self.attributes_index)]
        )

        for i in range(self.domain_size):
            key_index = np.where(encode_records == i)[0]
            self.summations[i, index] = np.sum(bigger_view.count[key_index])

    def update_view(self, common_view, index):
        encode_num = np.zeros(self.total_num_attr, dtype=np.uint32)
        encode_num[common_view.attributes_index] = common_view.encode_num
        encode_num = encode_num[self.attributes_index]

        encode_records = np.matmul(self.tuple_key, encode_num)

        for i in range(common_view.domain_size):
            key_index = np.where(encode_records == i)[0]
            self.count[key_index] += common_view.delta[i, index] / len(key_index)

    def non_negativity(self):
        count = np.copy(self.count)
        self.norm_cut(count)
        self.count = count

    @staticmethod
    def norm_sub(count):
        while (np.fabs(sum(count) - 1) > 1e-6) or (count < 0).any():
            count[count < 0] = 0
            total = sum(count)
            mask = count > 0
            if sum(mask) == 0:
                count[:] = 1.0 / len(count)
                break
            diff = (1 - total) / sum(mask)
            count[mask] += diff
        return count

    @staticmethod
    def norm_cut(count):
        negative_indices = np.where(count < 0.0)[0]
        negative_total = abs(np.sum(count[negative_indices]))
        count[negative_indices] = 0.0

        positive_indices = np.where(count > 0.0)[0]

        if positive_indices.size != 0:
            positive_sort_indices = np.argsort(count[positive_indices])
            sort_cumsum = np.cumsum(count[positive_indices[positive_sort_indices]])

            threshold_indices = np.where(sort_cumsum <= negative_total)[0]

            if threshold_indices.size == 0:
                count[positive_indices[positive_sort_indices[0]]] = sort_cumsum[0] - negative_total
            else:
                count[positive_indices[positive_sort_indices[threshold_indices]]] = 0.0
                next_index = threshold_indices[-1] + 1

                if next_index < positive_sort_indices.size:
                    count[positive_indices[positive_sort_indices[next_index]]] = (
                        sort_cumsum[next_index] - negative_total
                    )
        else:
            count[:] = 0.0

        return count


class Consistenter:
    class SubsetWithDependency:
        def __init__(self, attributes_set):
            self.attributes_set = attributes_set
            self.dependency = set()

    def __init__(self, views, num_categories):
        self.views = views
        self.num_categories = num_categories
        self.iterations = 30

    def compute_dependency(self):
        subsets_with_dependency = {}
        ret_subsets = {}

        for key, view in self.views.items():
            new_subset = self.SubsetWithDependency(view.attributes_set)
            subsets_temp = copy.deepcopy(subsets_with_dependency)

            for subset_key, subset_value in subsets_temp.items():
                attributes_intersection = subset_value.attributes_set & view.attributes_set

                if attributes_intersection:
                    if tuple(attributes_intersection) not in subsets_with_dependency:
                        intersection_subset = self.SubsetWithDependency(attributes_intersection)
                        subsets_with_dependency[tuple(attributes_intersection)] = intersection_subset

                    if not tuple(attributes_intersection) == subset_key:
                        subsets_with_dependency[subset_key].dependency.add(tuple(attributes_intersection))
                    new_subset.dependency.add(tuple(attributes_intersection))

            subsets_with_dependency[tuple(view.attributes_set)] = new_subset

        for subset_key, subset_value in subsets_with_dependency.items():
            if len(subset_key) == 1:
                subset_value.dependency = set()

            ret_subsets[subset_key] = subset_value

        return subsets_with_dependency

    def consist_views(self):
        def find_subset_without_dependency():
            for key, subset in subsets_with_dependency_temp.items():
                if not subset.dependency:
                    return key, subset

            return None, None

        def find_views_containing_target(target):
            result = []

            for key, view in self.views.items():
                if target <= view.attributes_set:
                    result.append(view)

            return result

        def consist_on_subset(target):
            target_views = find_views_containing_target(target)

            common_view_indicator = np.zeros(self.num_categories.shape[0])
            for index in target:
                common_view_indicator[index] = 1

            common_view = View(common_view_indicator, self.num_categories)
            common_view.initialize_consist_parameters(len(target_views))

            for index, view in enumerate(target_views):
                common_view.project_from_bigger_view(view, index)

            common_view.calculate_delta()
            if np.sum(np.absolute(common_view.delta)) > 1e-3:
                for index, view in enumerate(target_views):
                    view.update_view(common_view, index)

        def remove_subset_from_dependency(target):
            for _, subset in subsets_with_dependency_temp.items():
                if tuple(target.attributes_set) in subset.dependency:
                    subset.dependency.remove(tuple(target.attributes_set))

        for key, view in self.views.items():
            view.calculate_tuple_key()
            view.generate_attributes_index_set()
            view.sum = np.sum(view.count)

        subsets_with_dependency = self.compute_dependency()

        non_negativity = True
        iterations = 0

        while non_negativity and iterations < self.iterations:
            consist_on_subset(set())

            for key, view in self.views.items():
                view.sum = np.sum(view.count)

            subsets_with_dependency_temp = copy.deepcopy(subsets_with_dependency)

            while len(subsets_with_dependency_temp) > 0:
                key, subset = find_subset_without_dependency()

                if not subset:
                    break

                consist_on_subset(subset.attributes_set)
                remove_subset_from_dependency(subset)
                subsets_with_dependency_temp.pop(key, None)

            nonneg_view_count = 0

            for key, view in self.views.items():
                if (view.count < 0.0).any():
                    view.non_negativity()
                    view.sum = np.sum(view.count)
                else:
                    nonneg_view_count += 1

                if nonneg_view_count == len(self.views):
                    non_negativity = False

            iterations += 1

        for key, view in self.views.items():
            view.sum = np.sum(view.count)
            view.normalize_count = view.count if view.sum <= 0 else view.count / view.sum


class RecordSynthesizer:
    """The GUM (Gradually Update Method) record synthesizer: starts from a randomly
    (or singleton-marginal-) initialized synthetic dataset and iteratively re-assigns
    records so that its empirical marginals converge to the (consistentized, noisy)
    target marginals in attrs_view_dict."""

    rounding_method = "deterministic"

    def __init__(self, attrs, domains, num_records):
        self.attrs = attrs
        self.domains = domains
        self.num_records = num_records

        self.records = None
        self.df = None
        self.error_tracker = None

        self.under_cell_indices = None
        self.zero_cell_indices = None
        self.over_cell_indices = None
        self.records_throw_indices = np.array([], dtype=np.uint32)

        self.add_amount = 0
        self.add_amount_zero = 0
        self.reduce_amount = 0

        self.actual_marginal = None
        self.synthesize_marginal = None
        self.alpha = 1.0

        self.encode_records = None
        self.encode_records_sort_index = None

    def update_alpha(self, iteration):
        self.alpha = 1.0 * 0.84 ** (iteration // 20)

    def update_order(self, iteration, views, iterate_keys):
        errors = []
        for key in iterate_keys:
            self.update_records_before(views[key], key, iteration, mute=True)
            errors.append(LA.norm(self.actual_marginal - self.synthesize_marginal, 1))
        order = [k for _, k in sorted(zip(errors, iterate_keys), key=lambda t: -t[0])]
        return order

    def update_records(self, original_view, iteration, attrs):
        view = copy.deepcopy(original_view)

        self.update_records_before(view, attrs, iteration)
        self.update_records_main(view)
        self.determine_throw_indices()
        self.handle_zero_cells(view)

        if iteration % 2 == 0:
            self.complete_partial_ratio(view, 0.5)
        else:
            self.complete_partial_ratio(view, 1.0)

        self.update_records_before(view, attrs, iteration)

    def initialize_records(self, iterate_keys, method="random", singleton_views=None):
        self.records = np.empty([self.num_records, len(self.attrs)], dtype=np.uint32)

        for attr_i, attr in enumerate(self.attrs):
            if method == "random":
                self.records[:, attr_i] = np.random.randint(0, self.domains[attr_i], size=self.num_records)
            elif method == "singleton":
                self.records[:, attr_i] = self.generate_singleton_records(singleton_views[attr])

        import pandas as pd

        self.df = pd.DataFrame(self.records, columns=self.attrs)

    def generate_singleton_records(self, singleton):
        record = np.empty(self.num_records, dtype=np.uint32)
        dist_cumsum = np.cumsum(singleton.count)
        start = 0

        for index, value in enumerate(dist_cumsum):
            end = int(round(value * self.num_records))
            record[start:end] = index
            start = end

        np.random.shuffle(record)

        return record

    def update_records_main(self, view):
        alpha = self.alpha

        self.under_cell_indices = np.where(
            (self.synthesize_marginal < self.actual_marginal) & (self.synthesize_marginal != 0)
        )[0]

        under_rate = (
            self.actual_marginal[self.under_cell_indices] - self.synthesize_marginal[self.under_cell_indices]
        ) / self.synthesize_marginal[self.under_cell_indices]
        ratio_add = np.minimum(under_rate, np.full(self.under_cell_indices.shape[0], alpha))
        self.add_amount = self._rounding(
            ratio_add * self.synthesize_marginal[self.under_cell_indices] * self.num_records
        )

        self.zero_cell_indices = np.where((self.synthesize_marginal == 0) & (self.actual_marginal != 0))[0]
        self.add_amount_zero = self._rounding(alpha * self.actual_marginal[self.zero_cell_indices] * self.num_records)

        self.over_cell_indices = np.where(self.synthesize_marginal > self.actual_marginal)[0]
        num_add_total = np.sum(self.add_amount) + np.sum(self.add_amount_zero)

        beta = self.find_optimal_beta(num_add_total, self.over_cell_indices)
        over_rate = (
            self.synthesize_marginal[self.over_cell_indices] - self.actual_marginal[self.over_cell_indices]
        ) / self.synthesize_marginal[self.over_cell_indices]
        ratio_reduce = np.minimum(over_rate, np.full(self.over_cell_indices.shape[0], beta))
        self.reduce_amount = self._rounding(
            ratio_reduce * self.synthesize_marginal[self.over_cell_indices] * self.num_records
        ).astype(int)

        selected_record = self.records[:, view.attributes_index]
        self.encode_records = np.matmul(selected_record, view.encode_num)
        self.encode_records_sort_index = np.argsort(self.encode_records)
        self.encode_records = self.encode_records[self.encode_records_sort_index]

    def determine_throw_indices(self):
        valid_indices = np.nonzero(self.reduce_amount)[0]
        valid_cell_over_indices = self.over_cell_indices[valid_indices]
        valid_cell_num_reduce = self.reduce_amount[valid_indices]
        valid_data_over_index_left = np.searchsorted(self.encode_records, valid_cell_over_indices, side="left")
        valid_data_over_index_right = np.searchsorted(self.encode_records, valid_cell_over_indices, side="right")

        valid_num_reduce = np.sum(valid_cell_num_reduce)
        self.records_throw_indices = np.zeros(valid_num_reduce, dtype=np.uint32)
        throw_pointer = 0

        for i, cell_index in enumerate(valid_cell_over_indices):
            match_records_indices = self.encode_records_sort_index[
                valid_data_over_index_left[i] : valid_data_over_index_right[i]
            ]
            throw_indices = np.random.choice(match_records_indices, valid_cell_num_reduce[i], replace=False)

            self.records_throw_indices[throw_pointer : throw_pointer + throw_indices.size] = throw_indices
            throw_pointer += throw_indices.size

        np.random.shuffle(self.records_throw_indices)

    def handle_zero_cells(self, view):
        if self.zero_cell_indices.size != 0:
            for index, cell_index in enumerate(self.zero_cell_indices):
                num_partial = int(self.add_amount_zero[index])

                if num_partial != 0:
                    for i in range(view.view_num_attr):
                        self.records[self.records_throw_indices[:num_partial], view.attributes_index[i]] = (
                            view.tuple_key[cell_index, i]
                        )

                self.records_throw_indices = self.records_throw_indices[num_partial:]

    def complete_partial_ratio(self, view, complete_ratio):
        num_complete = np.rint(complete_ratio * self.add_amount).astype(int)
        num_partial = np.rint((1 - complete_ratio) * self.add_amount).astype(int)

        valid_indices = np.nonzero(num_complete + num_partial)
        num_complete = num_complete[valid_indices]
        num_partial = num_partial[valid_indices]

        valid_cell_under_indices = self.under_cell_indices[valid_indices]
        valid_data_under_index_left = np.searchsorted(self.encode_records, valid_cell_under_indices, side="left")
        valid_data_under_index_right = np.searchsorted(self.encode_records, valid_cell_under_indices, side="right")

        for valid_index, cell_index in enumerate(valid_cell_under_indices):
            match_records_indices = self.encode_records_sort_index[
                valid_data_under_index_left[valid_index] : valid_data_under_index_right[valid_index]
            ]

            np.random.shuffle(match_records_indices)

            if self.records_throw_indices.shape[0] >= (num_complete[valid_index] + num_partial[valid_index]):
                if num_complete[valid_index] != 0:
                    self.records[self.records_throw_indices[: num_complete[valid_index]]] = self.records[
                        match_records_indices[: num_complete[valid_index]]
                    ]

                if num_partial[valid_index] != 0:
                    self.records[
                        np.ix_(
                            self.records_throw_indices[
                                num_complete[valid_index] : (num_complete[valid_index] + num_partial[valid_index])
                            ],
                            view.attributes_index,
                        )
                    ] = view.tuple_key[cell_index]

                self.records_throw_indices = self.records_throw_indices[
                    num_complete[valid_index] + num_partial[valid_index] :
                ]
            else:
                self.records[self.records_throw_indices] = self.records[
                    match_records_indices[: self.records_throw_indices.size]
                ]

    def find_optimal_beta(self, num_add_total, cell_over_indices):
        actual_marginal_under = self.actual_marginal[cell_over_indices]
        synthesize_marginal_under = self.synthesize_marginal[cell_over_indices]

        lower_bound = 0.0
        upper_bound = 1.0
        beta = 0.0
        current_num = 0.0
        iteration = 0

        while abs(num_add_total - current_num) >= 1.0:
            beta = (upper_bound + lower_bound) / 2.0
            current_num = np.sum(
                np.minimum(
                    (synthesize_marginal_under - actual_marginal_under) / synthesize_marginal_under,
                    np.full(cell_over_indices.shape[0], beta),
                )
                * synthesize_marginal_under
                * self.records.shape[0]
            )

            if current_num < num_add_total:
                lower_bound = beta
            elif current_num > num_add_total:
                upper_bound = beta
            else:
                return beta

            iteration += 1
            if iteration > 50:
                break

        return beta

    def update_records_before(self, view, view_key, iteration, mute=False):
        self.actual_marginal = view.count
        count = view.count_records_general(self.records)
        self.synthesize_marginal = count / np.sum(count)

    def _rounding(self, vector):
        if self.rounding_method == "stochastic":
            ret_vector = np.zeros(vector.size)
            rand = np.random.rand(vector.size)

            integer = np.floor(vector)
            decimal = vector - integer

            ret_vector[rand > decimal] = np.floor(decimal[rand > decimal])
            ret_vector[rand < decimal] = np.ceil(decimal[rand < decimal])
            ret_vector += integer
            return ret_vector
        elif self.rounding_method == "deterministic":
            return np.round(vector)
        else:
            raise NotImplementedError(self.rounding_method)
