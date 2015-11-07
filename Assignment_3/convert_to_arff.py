def convert_to_arff(path):
	print "start" + path + "end"
	f = open(path, 	'r')
	lines = f.readlines()
        headers = len(lines[0].split(','))
	w = open(path[:-3] + 'arff', 'w')
        w.write("@relation whatever\n")
        w.write("\n")
	for i in range(headers-1):
		w.write("@attribute Component" + str(i) + " numeric\n")
        w.write("@attribute type {'0','1'}\n")
	w.write("\n")
	w.write("\n")
	w.write("@data\n")
	w.write(''.join(lines))


import glob, os
os.chdir("datasets/")
for file in glob.glob("*.csv"):
    convert_to_arff(file)


