"""
Dataset handler for EvoDevo dataset.
"""

import pandas as pd
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
from config import EVO_DEVO_DATA_LOC


time_mapper = {
        'Chicken': ['e10', 'e12', 'e14', 'e17', 'P0', 'P7', 'P35', 'P70', 'P155'],
        'Human': ['4wpc', '5wpc', '6wpc', '7wpc', '8wpc', '9wpc', '10wpc', '11wpc', '12wpc', '13wpc',
                  '16wpc', '18wpc', '19wpc', '20wpc', 'newborn', 'infant', 'toddler', 'school',
                  'youngTeenager', 'teenager', 'oldTeenager', 'youngAdult', 'youngMidAge',
                  'olderMidAge', 'Senior'],
        'Mouse': ['e10.5', 'e11.5', 'e12.5', 'e13.5', 'e14.5', 'e15.5', 'e16.5', 'e17.5', 'e18.5', 'P0',
                  'P3', 'P14', 'P28', 'P63'],
        'Opossum': ['13.5', '14', '16', '18', '20', '24', '28', '35', '42', '56', '74', '104', '134',
                    '164', '194'],
        'Rabbit': ['e12', 'e13', 'e14', 'e15.5', 'e16.5', 'e18', 'e19.5', 'e21', 'e23', 'e24', 'e27',
                   'P0', 'P14', 'P84', 'P186'],
        'Rat': ['e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'P0', 'P3', 'P7',
                'P14', 'P42', 'P112'],
        'RhesusMacaque': ['e93', 'e108', 'e112', 'e123', 'e130', 'P0', 'P23', 'P152', 'P183', 'P365',
                          'P1095', 'P3285', 'P5475', 'P8030'],
    }
for spec in time_mapper:
    time_mapper[spec] = [t.lower() for t in time_mapper[spec]]  # make everything lower case!


def load_data():
    attr = pd.read_csv(EVO_DEVO_DATA_LOC + 'attribute.txt')
    expr = np.loadtxt(EVO_DEVO_DATA_LOC + 'expression.txt')
    assert len(attr) == len(expr), "Must have same number of samples!"
    # map time to its index
    attr['time'] = attr[['species', 'time']].apply(  # map time to its index
        lambda x: time_mapper[x[0]].index(x[1].lower()), axis=1)
    return attr, expr


class EvoDevoDataset(torch.utils.data.Dataset):
    """
    Data loader for EvoDevo dataset
    """
    def __init__(self, train: bool = True, device=torch.device('cuda:0'), train_ordering=None):
        super().__init__()
        #print('...loading data')
        self.attr, self.expr = load_data()
        self.species = sorted(np.unique(self.attr['species']))
        self.organs = sorted(np.unique(self.attr['organ']))
        self.spec_to_idx = {spec: i for (i, spec) in enumerate(self.species)}
        self.org_to_idx = {org: i for (i, org) in enumerate(self.organs)}
        self.spec_org_combos = self.attr[['species', 'organ']].drop_duplicates()
        self.N = len(self.spec_org_combos)

        self.ordering = np.random.choice(range(self.N), size=self.N, replace=False
                                        )  # shuffle the dataset
        if train:
            self.ordering = self.ordering[:int(len(self.ordering) * 0.8)]
        else:
            # take the rows that aren't part of the train!
            self.ordering = [o for o in self.ordering if o not in train_ordering]

        self.device = device
        self.max_unique_times = max(len(time_mapper[s]) for s in time_mapper)

    def get_species(self):
        return self.species

    def get_organs(self):
        return self.organs

    def num_species(self):
        return len(self.species)

    def num_organs(self):
        return len(self.organs)

    def get_species_encoding(self, species):
        enc = torch.zeros(len(self.spec_to_idx))
        enc[self.spec_to_idx[species]] = 1
        return enc

    def get_organ_embedding(self, organ):
        enc = torch.zeros(len(self.org_to_idx))
        enc[self.org_to_idx[organ]] = 1
        return enc

    def __getitem__(self, item):
        rowOI = self.spec_org_combos.iloc[self.ordering[item]]
        species = rowOI['species']
        organ = rowOI['organ']

        # find (and sort) the relevant times
        attrOI = self.attr[(self.attr['species'] == species) & (self.attr['organ'] == organ)]
        attrOI = attrOI.sort_values(by='time', ascending=True)
        idxsOI = attrOI.index.to_list()
        exprs = self.expr[idxsOI]

        spec_onehot = self.get_species_encoding(species)
        org_onehot = self.get_organ_embedding(organ)
        exprs = torch.tensor(np.stack(exprs, axis=0))

        times = torch.tensor(list(attrOI['time']))
        mask = torch.zeros(self.max_unique_times)
        mask[times.long()] = 1

        full_expr = torch.zeros((self.max_unique_times, exprs.shape[-1]), dtype=torch.float)
        full_expr[times.long()] = exprs.float()
        # finally, make times arange
        times = torch.tensor(np.arange(self.max_unique_times), dtype=float)

        return spec_onehot, org_onehot, times, full_expr, mask

    def __len__(self):
        return self.N
