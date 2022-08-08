import os
import re

for root, _, files in os.walk('./'):
        for filename in files:
            if filename.endswith('.py') and filename != 'rewrite_params.py':
                if root == "./":
                    filepath = filename
                else:
                    filepath = root + "/" + filename

                with open(filepath, 'r') as file_reader:
                    lines = file_reader.readlines()
                    for i, line in enumerate(lines):
                        if ':param' in line:
                            prev_line = lines[i-1]
                            if prev_line != "\n" and '"""' not in prev_line and ":param" not in prev_line:
                                lines[i] = "\n" + lines[i]
                    with open(filepath, 'w') as file_reader:
                        file_reader.writelines(lines)
