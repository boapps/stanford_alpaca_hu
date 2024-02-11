import json
import re
 
f = open('regen_bad_3.json')
data = json.load(f)
f.close()

def filteri(row):
    if '<üres>' in row['input']:
        row['input'] = ''
    return row

#fg = open('regen.json', 'w')
#json.dump([filteri(row) for row in data], fg)
#fg.close()
