import os
import re

def getHighestModelNumber(path):
    
    files = os.listdir(path)
    print('files in direcotry: ', files)
    pattern = r"(\d+)*"

    if len(files) == 0:
        return str(1)
    
    max_num = 1
    for file in files:
        match = re.match(pattern, file)

        if match:
            if int(match.group(1)) > max_num:
                max_num = int(match.group(1))

    return str(max_num)

