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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class MeasureFst(GraphFst):
    """Finite state transducer for verbalizing Telugu measures."""

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst):
        super().__init__(name="measure", kind="verbalize")

        unit = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        integer_half = (
            pynutil.delete('integer_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynutil.insert(".౫")
            + delete_space
        )

        fraction_half = (
            pynutil.delete("fraction {")
            + delete_space
            + (integer_half | pynutil.insert(".౫"))
            + pynutil.delete('numerator: "౧"')
            + delete_space
            + pynutil.delete('denominator: "౨"')
            + delete_space
            + pynutil.delete("}")
        )

        graph_measurement = (
            (cardinal.fst | decimal.fst | pynutil.add_weight(fraction_half, -0.1) | fraction.fst)
            + delete_space
            + pynutil.insert(" ")
            + unit
        )

        graph_dimension = cardinal.fst

        self.fst = self.delete_tokens(graph_dimension | graph_measurement).optimize()
