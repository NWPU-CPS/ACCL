import sys
from misc.log import log
from engine.engine import JupiterEngine


def cfgname():
    # cfgname="nyf.dssaccl.cuda@1.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@2.ne" # 92.8 - map96.39

    # cfgname="parisf.dss.cuda@3.batch@16.lr@40.loss@infoNCE" # 81.30

    # cfgname="parisf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@4.k@16.ne2" # 84.87
    # cfgname="parisf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@2.k@16.ne2" # 84.11
    # cfgname="parisf.dssaccl.cuda@3.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@8.k@16.ne2" # 84.20
    # cfgname="parisf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@4.k@8.ne2" # 82.49
    cfgname="parisf.dssaccl.cuda@3.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@4.k@32.ne2" # 83.17
    
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
