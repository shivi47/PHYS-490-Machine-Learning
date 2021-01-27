from LRMGradientDescent import GD
from LRMAnalyticSolution import AS
import numpy as np
import json
import sys

if len(sys.argv) != 3:
    sys.stderr.write("Usage: %s data/file_name.in data/file_name.json" % sys.argv[0])
    sys.exit(1)



#setting variables based in given arguments from command line
data = np.loadtxt('./%s' % sys.argv[1])
params = json.load(open('%s' % sys.argv[2]))

if len(data[0, :]) == 3:
    w_gd = GD(data, params)
    w_as = AS(data, params)
    fout = open('data/1.out', 'w')
    fout.write('%s \n%s \n%s \n\n%s \n%s \n%s' % (w_as[0], w_as[1], w_as[2], w_gd[0], w_gd[1], w_gd[2]))
    fout.close()

elif len(data[0, :]) == 5:
    w_gd = GD(data, params)
    w_as = AS(data, params)
    fout = open('data/2.out', 'w')
    fout.write('%s \n%s \n%s \n%s \n%s \n\n%s \n%s \n%s \n%s \n%s' % (w_as[0], w_as[1], w_as[2], w_as[3], w_as[4], w_gd[0], w_gd[1], w_gd[2], w_gd[3], w_gd[4]))
    fout.close()
