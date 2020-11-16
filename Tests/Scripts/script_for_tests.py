#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import sys

"""
This is a script that will be invoked from the test suite.
"""

if __name__ == '__main__':
    arg = sys.argv[1]
    if arg != "correct":
        raise ValueError("Not correct")
