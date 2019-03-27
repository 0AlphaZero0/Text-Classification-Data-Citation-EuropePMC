import lxml.etree as let
import os
import sys

path = sys.argv[1]

for f in os.listdir(path):
	if '.py' not in f:
		print(f)
		fpath = os.path.join(path, f)
		try:
			tree = let.parse(fpath)
			print(tree.getroot().tag)
		except:
			print(f'{f} format doesnt right...')