#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Dict, List, Optional, Union

import yacs.config


class ConfigNode(yacs.config.CfgNode):
    def __init__(self, init_dict: Optional[Dict] = None, key_list: Optional[List] = None, new_allowed: bool = False):
        super().__init__(init_dict, key_list, new_allowed)

    def __str__(self) -> str:
        def _indent(s_: str, num_spaces: int) -> Union[str, List[str]]:
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)      # type: ignore
            s = first + '\n' + s  # type: ignore
            return s

        r = ''
        s = []
        for k, v in self.items():
            separator = '\n' if isinstance(v, ConfigNode) else ' '
            if isinstance(v, str) and not v:
                v = '\'\''
            attr_str = f'{str(k)}:{separator}{str(v)}'
            attr_str = _indent(attr_str, 2)     # type: ignore
            s.append(attr_str)
        r += '\n'.join(s)
        return r

    def as_dict(self) -> Dict:
        def convert_to_dict(node: ConfigNode) -> Dict:
            if not isinstance(node, ConfigNode):
                return node
            else:
                dic = dict()
                for k, v in node.items():
                    dic[k] = convert_to_dict(v)
                return dic

        return convert_to_dict(self)
