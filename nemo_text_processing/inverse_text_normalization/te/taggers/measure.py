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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_NOT_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class MeasureFst(GraphFst):
    """Finite state transducer for classifying Telugu measures and dimensions."""

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative
        fraction_graph = fraction.fst

        measurements_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert().optimize()

        units = pynutil.insert('units: "') + measurements_graph + pynutil.insert('"')
        unit_inputs = pynini.project(measurements_graph, "input")

        number_context_words = pynini.union(
            pynini.project(cardinal_graph, "input"),
            pynini.project(
                pynini.string_file(get_abs_path("data/numbers/hundred.tsv")),
                "input",
            ),
            pynini.project(
                pynini.string_file(get_abs_path("data/numbers/thousand.tsv")),
                "input",
            ),
            pynini.project(
                pynini.string_file(get_abs_path("data/numbers/lakh.tsv")),
                "input",
            ),
            pynini.project(
                pynini.string_file(get_abs_path("data/numbers/crore.tsv")),
                "input",
            ),
        ).optimize()

        context_word = pynini.difference(
            pynini.closure(NEMO_NOT_SPACE, 1),
            pynini.union(unit_inputs, number_context_words),
        ).optimize()

        context_separator = pynini.cross(" ", " ")
        context = context_separator + context_word + pynini.closure(context_separator + context_word)

        graph_cardinal = pynutil.insert('cardinal { integer: "') + cardinal_graph + pynutil.insert('" }')

        graph_decimal = pynutil.insert("decimal { ") + decimal_graph + pynutil.insert(" }")

        graph_dimension = (
            pynutil.insert('cardinal { integer: "')
            + cardinal_graph
            + delete_space
            + pynini.cross("బై", "x")
            + delete_space
            + cardinal_graph
            + context
            + pynutil.insert('" }')
        )

        graph_measurement = (graph_decimal | graph_cardinal | fraction_graph) + delete_extra_space + units

        self.graph = (graph_dimension | graph_measurement).optimize()
        self.fst = self.add_tokens(self.graph).optimize()
