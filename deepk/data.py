from collections import defaultdict
import torch

from deepk import config as cfg
from deepk import utils


class DataHandler:
    def __init__(self, Xtr, ttr, Xva=None, tva=None, Xte=None, tte=None):
        self.Xtr = utils._tensorize(Xtr, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.Xva = utils._tensorize(Xva, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.Xte = utils._tensorize(Xte, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.ttr = utils._tensorize(ttr, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.tva = utils._tensorize(tva, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.tte = utils._tensorize(tte, dtype=cfg._RTYPE, device=cfg._DEVICE)

        ## Define Xscale, and normalize X data if applicable
        self.Xscale = torch.max(torch.abs(self.Xtr))
        if cfg.normalize_Xdata:
            self.Xtr = utils._scale(self.Xtr, scale=self.Xscale)
            self.Xva = utils._scale(self.Xva, scale=self.Xscale)
            self.Xte = utils._scale(self.Xte, scale=self.Xscale)

        ## Shift t data to make ttr start from 0 if it doesn't
        self.tshift = self.ttr[0].item()
        self.ttr = utils._shift(self.ttr, shift=self.tshift)
        self.tva = utils._shift(self.tva, shift=self.tshift)
        self.tte = utils._shift(self.tte, shift=self.tshift)

        ## Find differences between ttr values
        self.dts = defaultdict(int) # define this as a class attribute because it will be accessed outside for writing to log file
        for i in range(len(self.ttr)-1):
            self.dts[self.ttr[i+1].item()-self.ttr[i].item()] += 1

        ## Define t scale as most common difference between ttr values, and normalize t data by it
        self.tscale = max(self.dts, key=self.dts.get)
        self.ttr = utils._scale(self.ttr, scale=self.tscale)
        self.tva = utils._scale(self.tva, scale=self.tscale)
        self.tte = utils._scale(self.tte, scale=self.tscale)

        ## Ensure that ttr now goes as [0,1,2,...], i.e. no gaps
        self.ttr = torch.round(self.ttr)
        dts_set = set(self.ttr[i+1].item()-self.ttr[i].item() for i in range(len(self.ttr)-1))
        if dts_set != {1}:
            raise ValueError(f"Training indexes are not equally spaced and cannot be rounded to get equal spacing. Please check 'ttr' = {ttr}")
