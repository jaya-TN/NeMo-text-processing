# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path, load_labels


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu ordinals.

    Args:
        cardinal: CardinalFst instance used to obtain the raw cardinal graph.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))

        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties.tsv"))

        graph_hundred = pynini.string_file(get_abs_path("data/ordinals/hundred_digit.tsv"))

        morph_features = load_labels(get_abs_path("data/ordinals/morph_features.tsv"))

        canonical_features = {row[1] for row in morph_features}

        if len(canonical_features) != 1:
            raise ValueError("Expected exactly one canonical ordinal morphosyntactic feature")

        morph_feature = next(iter(canonical_features))

        ordinal_tail = pynini.union(
            graph_digit,
            graph_teens,
            graph_hundred,
        ).optimize()

        graph = pynini.compose(
            pynini.closure(NEMO_CHAR) + ordinal_tail,
            cardinal_graph,
        ).optimize()

        final_graph = (
            pynutil.insert('integer: "')
            + graph
            + pynutil.insert('"')
            + pynutil.insert(f' morphosyntactic_features: "{morph_feature}"')
        )

        self.fst = self.add_tokens(final_graph).optimize()
