import sys
from misc.log import log
from engine.engine import JupiterEngine


def cfgname():
    # cfgname="nyf.dssaccl.cuda@1.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@2.ne" # 92.8 - map96.39
    
    # cfgname="parisf.dss.cuda@0.batch@16.lr@20.loss@infoNCE" # 81.39
    # cfgname="parisf.dssaccl.cuda@1.batch@16.lr@20.loss@infoNCEWeightedDirect.k@8.ne" # 84.96

    # cfgname="parisf.dssaccl.cuda@2.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@7.k@16.ne2" # 79.95 
    # cfgname="parisf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@4.k@16.ne2" # 79.95 
    # cfgname="parisf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.k@32.ne" # 80.37
    # cfgname="parisf.dssaccl.cuda@1.batch@16.lr@40.loss@infoNCEWeightedDirect.k@8.ne" # 81.13
    # cfgname="parisf.dssaccl.cuda@2.batch@16.lr@40.loss@infoNCEWeightedDirect.k@64.ne" # 77.90
    # cfgname="parisf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.k@4.ne" # 80.79
    
    # cfgname="lvf.dss.cuda@1.batch@16.lr@40.loss@infoNCE" # 81.9 - 90.48
    # cfgname="lvf.dssaccl.cuda@3.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@4.k@16.ne2" # 83.90
    # cfgname="lvf.dssaccl.cuda@2.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@7.k@16.ne2" # 84.87 - map92.41

    return cfgname


if __name__ == "__main__":
    mode = None
    nobuf = False

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"

    if mode is None:
        log.inf("Using default mode [pipeline]")
        mode = "pipe"

    JupiterEngine(
        mode=mode,
        cfgname=cfgname(),
        nobuf=nobuf,
    ).run()
