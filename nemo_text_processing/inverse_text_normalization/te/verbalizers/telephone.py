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


class TelephoneFst(GraphFst):
    """
    Verbalizes Telugu telephone fields while preserving digits and context.

    Example:
        telephone { country_code: "+౯౧" number_part: "౯౮౭౬౫౪౩౨౧౦" }
        -> +౯౧ ౯౮౭౬౫౪౩౨౧౦
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        value = pynini.closure(NEMO_NOT_QUOTE, 1)
        country_code = (
            pynutil.delete('country_code: "') + value + pynutil.delete('"') + delete_space + pynutil.insert(" ")
        )
        number_part = pynutil.delete('number_part: "') + value + pynutil.delete('"')
        graph = pynini.closure(country_code, 0, 1) + number_part
        self.fst = self.delete_tokens(graph).optimize()
