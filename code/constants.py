""" constants used throughout codebase """

# list of chars that can be encountered in any sequence
CHARS = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# number of chars
NUM_CHARS = 21

# dictionary mapping chars->int
C2I_MAPPING = {c: i for i, c in enumerate(CHARS)}

AA_3TO1_MAP = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
               'TER': '*'}

# all the standard rosetta attributes corresponding to all the columns stored in the main database
ROSETTA_ATTRIBUTES = ('total_score', 'dslf_fa13', 'fa_atr', 'fa_dun', 'fa_elec',
                      'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                      'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'lk_ball_wtd',
                      'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'ref', 'yhh_planarity',
                      'filter_total_score', 'buried_all', 'buried_np', 'contact_all',
                      'contact_buried_core', 'contact_buried_core_boundary', 'degree',
                      'degree_core', 'degree_core_boundary', 'exposed_hydrophobics',
                      'exposed_np_AFIMLWVY', 'exposed_polars', 'exposed_total',
                      'one_core_each', 'pack', 'res_count_all', 'res_count_buried_core',
                      'res_count_buried_core_boundary', 'res_count_buried_np_core',
                      'res_count_buried_np_core_boundary', 'ss_contributes_core', 'ss_mis',
                      'total_hydrophobic', 'total_hydrophobic_AFILMVWY', 'total_sasa',
                      'two_core_each', 'unsat_hbond', 'centroid_total_score', 'cbeta',
                      'cenpack', 'env', 'hs_pair', 'linear_chainbreak', 'overlap_chainbreak',
                      'pair', 'rg', 'rsigma', 'sheet', 'ss_pair', 'vdw')

# break down into where the energies come from (used for plotting in one or two places, not necessary for training)
ROSETTA_ATTRIBUTES_REF15 = ('total_score', 'dslf_fa13', 'fa_atr', 'fa_dun', 'fa_elec',
                            'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                            'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'lk_ball_wtd',
                            'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'ref', 'yhh_planarity')

ROSETTA_ATTRIBUTES_FILTER = ('filter_total_score', 'buried_all', 'buried_np', 'contact_all',
                             'contact_buried_core', 'contact_buried_core_boundary', 'degree',
                             'degree_core', 'degree_core_boundary', 'exposed_hydrophobics',
                             'exposed_np_AFIMLWVY', 'exposed_polars', 'exposed_total',
                             'one_core_each', 'pack', 'res_count_all', 'res_count_buried_core',
                             'res_count_buried_core_boundary', 'res_count_buried_np_core',
                             'res_count_buried_np_core_boundary', 'ss_contributes_core', 'ss_mis',
                             'total_hydrophobic', 'total_hydrophobic_AFILMVWY', 'total_sasa',
                             'two_core_each', 'unsat_hbond')

ROSETTA_ATTRIBUTES_CENTROID = ('centroid_total_score', 'cbeta',
                               'cenpack', 'env', 'hs_pair', 'linear_chainbreak', 'overlap_chainbreak',
                               'pair', 'rg', 'rsigma', 'sheet', 'ss_pair', 'vdw')

# only the rosetta attributes that are used for training (excluding some columns)
ROSETTA_ATTRIBUTES_TRAINING = ('total_score', 'fa_atr', 'fa_dun', 'fa_elec',
                               'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                               'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'lk_ball_wtd',
                               'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'ref', 'yhh_planarity',
                               'buried_all', 'buried_np', 'contact_all',
                               'contact_buried_core', 'contact_buried_core_boundary', 'degree',
                               'degree_core', 'degree_core_boundary', 'exposed_hydrophobics',
                               'exposed_np_AFIMLWVY', 'exposed_polars', 'exposed_total',
                               'one_core_each', 'pack', 'res_count_buried_core',
                               'res_count_buried_core_boundary', 'res_count_buried_np_core',
                               'res_count_buried_np_core_boundary', 'ss_contributes_core', 'ss_mis',
                               'total_hydrophobic', 'total_hydrophobic_AFILMVWY', 'total_sasa',
                               'two_core_each', 'unsat_hbond', 'centroid_total_score', 'cbeta',
                               'cenpack', 'env', 'hs_pair',
                               'pair', 'rg', 'rsigma', 'sheet', 'ss_pair', 'vdw')

# all the attributes calculated during the docking run (includes standard energies for the complex)
ROSETTA_ATTRIBUTES_DOCKING = ("total_score", "complex_normalized", "dG_cross", "dG_cross/dSASAx100", "dG_separated",
                              "dG_separated/dSASAx100", "dSASA_hphobic", "dSASA_int", "dSASA_polar",
                              "delta_unsatHbonds", "dslf_fa13", "fa_atr", "fa_dun", "fa_elec", "fa_intra_rep",
                              "fa_intra_sol_xover4", "fa_rep", "fa_sol", "hbond_E_fraction", "hbond_bb_sc",
                              "hbond_lr_bb", "hbond_sc", "hbond_sr_bb", "hbonds_int", "lk_ball_wtd", "nres_all",
                              "nres_int", "omega", "p_aa_pp", "packstat", "per_residue_energy_int", "pro_close",
                              "rama_prepro", "ref", "sc_value", "side1_normalized", "side1_score", "side2_normalized",
                              "side2_score", "yhh_planarity")

ROSETTA_ATTRIBUTES_DOCKING_TRAINING = ("complex_normalized", "dG_cross", "dG_cross/dSASAx100", "dG_separated",
                                       "dG_separated/dSASAx100", "dSASA_hphobic", "dSASA_int", "dSASA_polar",
                                       "delta_unsatHbonds", "hbond_E_fraction", "hbonds_int", "nres_int",
                                       "per_residue_energy_int", "side1_normalized", "side1_score",
                                       "side2_normalized", "side2_score")
