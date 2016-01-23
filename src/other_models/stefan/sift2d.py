﻿from __future__ import print_function

import os
import sys
import optparse
import numpy as np
import matplotlib.pyplot as plt
import inspect
import math
import itertools
from string import Template
try:
    from schrodinger.structutils import analyze
    from schrodinger.structutils import interactions
    from schrodinger.structutils import measure
    from schrodinger import structure
except:
    pass

#Iteraction type to row/column number
columns = {
    'ANY' : 0,  
    'BACKBONE' : 1,  
    'SIDECHAIN' : 2,  
    'POLAR' : 3,  
    'HYDROPHOBIC' : 4,  
    'H_ACCEPTOR' : 5,  
    'H_DONOR' : 6,  
    'AROMATIC' : 7,  
    'CHARGED' : 8
    }
rows = {
    'H_DONOR' : 0, 
    'H_ACCEPTOR' : 1, 
    'HYDROPHOBIC' : 2, 
    'N_CHARGED' : 3, 
    'P_CHARGED' : 4, 
    'AROMATIC' : 5,
    'ANY' : 6,
    }

#Residue sets by properties
residue_sets = {
    "POLAR_RESIDUES": ["ARG","ASP","GLU","HIS","ASN","GLN","LYS","SER","THR","ARN","ASH","GLH","HID","HIE","LYN"],
    "HYDROPHOBIC_RESIDUES": ["PHE","LEU","ILE","TYR","TRP","VAL","MET","PRO","CYS","ALA","CYX"],
    "AROMATIC_RESIDUES": ["PHE","TYR","TRP","TYO"],
    "P_CHARGED_RESIDUES": ["ARG","LYS","HIS","HID"],
    "N_CHARGED_RESIDUES": ["ASP","GLU"],
    }


#Rules for distance-based interactions, h_bonds and aromatic interactions are evaluated with builtin Schrodinger functions
interaction_rules = {
    'P_CHARGED': [['N_CHARGED_RESIDUES'], ['CHARGED']], 
    'N_CHARGED': [['P_CHARGED_RESIDUES'], ['CHARGED']],
    'H_DONOR': [['POLAR_RESIDUES'], ['POLAR']],
    'H_ACCEPTOR': [['POLAR_RESIDUES'], ['POLAR']],
    'HYDROPHOBIC': [['HYDROPHOBIC_RESIDUES'], ['HYDROPHOBIC']],
    }
interaction_dict = {
    'residue_features': 0,
    'byte_to_switch': 1        
    }

#SMARTS patterns for ligand features (except AROMATIC)
ph_patterns = {
    'H_DONOR': 
    ['[#1][O;X2]', '[#1]S[#6]', '[#1][C;X2]#[C;X2]', '[#1][NX3]C(=[NX2])[#6]', '[#1][#7]'],
    'H_ACCEPTOR': 
    ['[N;X1]#[#6]', '[N;X1]#CC', '[N;X2](=C~[C,c])C', '[N;X2](O)=N[a]', '[N;X2](=N-O)[a]', '[n;X2]1ccccc1', '[n;X2]([a])([a])', '[N;X2](=C~[C,c])(~[*])', '[N;X3](C)(C)[N;X3]C', '[N;X2](=C)(~[*])', '[N;X2](~[C,c])=[N;X2]', '[n;X2]1c[nH]cc1', 'O=[S;X4](=O)([!#8])([!#8])', '[O;X2]C', '[O;X2]N', '[O;X1]=[C,c]', 'o', '[O;X2](C)C', '[O;X2]c1ncccc1', '[O;X2]~[a]', 'O=PO([!#1])', '[O;X2]', '[S;X2](C)C', '[S;X2](=C)N'], 
    'P_CHARGED': 
    ['[NX3][#6](=[NX2,NX3+])[#6]', '[NX2,NX3+]=[#6]([NH;X3])([NH;X3])', '[NX2,NX3+]=[#6]([NX3])([NX3])', 'n1c([NH2])ccnc1([NH2])', '[NX2,NX3+]=C([NX3])c1ccccc1', '[NH2;X3,NH3]([#6;X4])', '[NH;X3,NH2]([#6;X4])([#6;X4])', '[NX3,NH]([#6;X4])([#6;X4])([#6;X4])', 'N1CCCCC1', '[+]'],
    'HYDROPHOBIC': 
    ['[a]F', '[a]Cl', '[a]Br', '[a]I', '[a]C(F)(F)(F)', '[a][CH2]C(F)(F)(F)', '[a]O[CH3]', '[a]S[CH3]', '[a]OC(F)(F)(F)', 'C(F)(F)(F)', 'F', 'Cl', 'Br', 'I', 'C[S;X2]C', '[S;X2]CC', '[S;X2]C'],
    'N_CHARGED': 
    ['O=C[O-]','O=C[OH]','[S;X4](=O)(=O)([OH])','[S;X4](=O)(=O)([O-])','[S;X3](=O)([OH])','[S;X3](=O)([O-])','[P;X4](=O)([OH])([OH])','[P;X4](=O)([OH])([O-])','[P;X4](=O)([O-])','[P;X4](=O)([OH])','n1nc[nH]n1','n1ncnn1','[#1]N([S;X4](=O)(=O))(C(F)(F)(F))','[-]'],
    }

#Output templates
output_format = {
    'txt' : Template("$receptor:$ligand:$start:$fp"),
    }


#==============================================================================  
class SIFt2DChunk:
    """
    Class storing information about interactions for a single residue.
    """  
    #odwolania do tablic numpy tabl[wiersz, kolumna] przy tworzeniu i
    #czytaniu/modyfikacji
    
    def __init__(self, resnum=None, data=None):
        """
        Class storing interaction matrix for a single residue. 
        """
        self.resnum = resnum
        #Schrodinger's version of numpy returns shape in form of (xL, yL) - can't just compare shape to a tuple (rows, columns)
        if data is None or data.shape != np.zeros((len(rows), len(columns))).shape:
            self.chunk = np.zeros((len(rows), len(columns)), dtype=np.int)
        else:
            self.chunk = data


    def increment_bit(self, row, col):
        """
        Increase value at given position by 1.
        """
        self.chunk[rows[row]][columns[col]] += 1


    def __repr__(self):

        return str(self.chunk)


    def __eq__(self, other):
        """
        Allows comparison of the chunks based on the interaction matrix
        """
        return (self.chunk == other.chunk).all()


    def __ne__ (self, other):
        """
        Allows comparison of the chunks based on the interaction matrix
        """
        return not (self.chunk == other.chunk).all()


    def __sub__(self, other):
        """
        Returns the difference between two chunks.
        """
        return SIFt2DChunk(self.resnum, self.chunk - other.chunk)


    def get_row(self, row_num):
        """
        Returns interactions of given pharmacophore feature.
        """
        return self.chunk[row_num, :]


    def get_column(self, col_num):
        """
        Returns interactions of given type.
        """
        return self.chunk[:, col_num]


    def get_heatmap(self, vmax=None):
        """
        Returns a standaried heatmap figure.
        """
        heatmap = plt.figure()
        ax = heatmap.gca(xlabel="Residue interactions", ylabel="Ligand features")
        ax.set_yticks(np.arange(len(rows))+0.5, minor=False)
        ax.set_yticklabels(['D','A','H','N','P','R','vdW'])
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(columns))+0.5, minor=False)
        ax.set_xticklabels(['Any', 'BB', 'SC', 'Polar', 'H', 'A', 'D', 'R', 'Charged'])
        coll = plt.pcolor(self.chunk, vmax=vmax, cmap=plt.cm.Greys, figure=heatmap)
        
        return heatmap



#==============================================================================  
class SIFt2D:
    """
    Class storing interaction matrix for the whole receptor.
    """
    generic_chunk = SIFt2DChunk()


    def __init__(self, ligand_name, receptor_name, start, end, custom_residues_set = None):
        """
        Constructor takes a starting and ending residue number.
        """
        self.ligand_name = ligand_name
        self.receptor_name = receptor_name
        self.start = start
        self.end = end
        self._chunks = []
        self._mapping = None
        #For truncated binding sites and sets of generic numbers
        self.custom_residues_set = custom_residues_set 
        self._init_chunks()


    def _init_chunks(self):
        """
        Initialize empty chunks for every residue in range.
        """
        if self.custom_residues_set is None:
            for x in range(self.start, self.end+1):
                self._chunks.append(SIFt2DChunk(x))
        else:
            if self.custom_residues_set:
                self._mapping = {}
            for idx, item in enumerate(self.custom_residues_set):
                self._mapping[item] = idx
                self._chunks.append(SIFt2DChunk(item))

    #TODO: Add support for custom numbers for __len__, __iter__ and __getitem__
    def __len__(self):
        """
        Return the number of per residue chunks.
        """
        return len(self._chunks)


    def __iter__(self):
        """
        Iterator over all per residue chunks.
        """
        for chunk in self._chunks:
            yield chunk

    def __getitem__(self, resnum):
        """
        Get chunk(s) related to a given residue or residue range.
        """
        if self._mapping:
            try:
                return self._chunks[self._mapping[resnum]]
            except KeyError:
                raise KeyError("No chunk found for residue {}.".format(resnum))
        else:
            try:
                return self._chunks[resnum - self.start]
            except KeyError:
                if resnum > self.end or resnum < self.start:
                    raise KeyError("Residue number {:n} out of range {:n}:{:n}".format(resnum, self.start, self.end))
                else:
                    raise KeyError("No chunk found for residue {:n}".format(resnum))

    def __sub__(self, other):
        """
        Compare two SIFt2D objects and return a new differential SIFt2D. 
        """
        if self._mapping:
            if self.custom_residues_set == other.custom_residues_set:
                common_residues_set = self.custom_residues_set
                result = SIFt2D("",self.receptor_name, self.start, self.end, self.custom_residues_set)
            else:
                common_residues_set = sorted(list(set(self.custom_residues_set) | set(other.custom_residues_set)))
                result = SIFt2D("",self.receptor_name, self.start, len(common_residues_set), common_residues_set)

            for res in common_residues_set:
                if res in self.custom_residues_set:
                    if res in other.custom_residues_set:
                        result.set_chunk(self[res] - other[res], res, True)
                    else:
                        result.set_chunk(self[res] - self.generic_chunk, res, True)
                else:
                    if res in other.custom_residues_set:
                        result.set_chunk(SIFt2DChunk(res) - other[res], res, True)
                    else:
                        print("This situation shouldn't take place. Residue: {}".format(res))
        else:
            result = SIFt2D("",self.receptor_name, min(self.start, other.start), max(self.end, other.end))

            for res in range(min(self.start, other.start), max(self.end, other.end)+1):
                if res < self.start:
                    result.set_chunk(SIFt2DChunk(res) - other[res], res)
                elif res > self.end:
                    result.set_chunk(SIFt2DChunk(res) - other[res], res)
                elif res < other.start:
                    result.set_chunk(self[res] - self.generic_chunk, res)
                elif res > other.end:
                    result.set_chunk(self[res] - self.generic_chunk, res)
                else:
                    try:
                        result.set_chunk(self[res] - other[res], res)
                    except Exception as e:
                        print("Something gone wrong for residue {!n}:\n{}".format(res, e))
        return result


    def prepend_chunk(self, custom_residue=None):
        """
        Insert empty chunks at the beginning of the chunk list.
        """
        tmp = []
        #self._chunks
        if custom_residue:
            tmp_mapping = {}
            tmp.append(SIFt2DChunk(custom_residue))
            tmp_mapping[custom_residue] = 0
            for key in self._mapping.keys():
                tmp_mapping[key] = self._mapping[key] + 1


        else:
            tmp.append(SIFt2DChunk(self.start-1))
            self.start -= 1
        tmp.extend(self._chunks)
        self._chunks = tmp


    def append_chunk(self, custom_residue=None):
        """
        Add empty chunk at the end of the chunk list.
        """
        if custom_residue:
            self._chunks.append(SIFt2DChunk(custom_residue))
            self._mapping[custom_residue] = len(self._chunks) -1
        else:
            self._chunks.append(SIFt2DChunk(self.end + 1))
            self.end += 1


    def set_chunk(self, chunk, residue_number, custom_residue=False):
        """
        Replace chunk at given position with the input.
        """
        if chunk.resnum != residue_number:
            raise KeyError("Residue number mismatch {:n}, chunk: {:n}".format(residue_number, chunk.resnum))

        if custom_residue:
            if residue_number not in self._mapping.keys():
                raise KeyError("Residue {} not found!".format(residue_number))
            self._chunks[self._mapping[residue_number]] = chunk
        else:
            if not self.start <= residue_number <= self.end:
                raise KeyError("Residue {:n} out of range ({:n} {:n})".format(residue_number, self.start, self.end))           
            if self._chunks[residue_number - self.start].resnum == residue_number:
                self._chunks[residue_number - self.start] = chunk


    def sort_chunks(self):
        """
        Applicable only for custom residues sets (residues don't cover given range).
        """
        tmp = []
        tmp_mapping = {}

        for idx, gn in enumerate(sorted(self._mapping.keys)):
            tmp_mapping[gn] = idx
            tmp.append(self._chunks[self._mapping[gn]])
        self._chunks = tmp
        self._mapping = tmp_mapping


    def get_chunks(self):
        """
        Returns a list of chunks.
        """
        if self._mapping:
            return [self._chunks[self._mapping[x]] for x in sorted(self._mapping.keys())]
        else:
            return self._chunks


    def get_listed_chunks(self, res_list):
        """
        The function returns a list of SIFt2DChunk objects corresponding to the input list of positions. In case of KeyError, an empty chunk is added.
        """
        if self._mapping:
            output = []
            for num in res_list:
                try:
                    output.append(self._chunks[self._mapping[num]])
                except KeyError:
                    output.append(SIFt2DChunk(num))
            return output


    def get_interacting_chunks(self):
        """
        Return out non-zero chunks.
        """
        return [x for x in self._chunks if x != self.generic_chunk]


    def get_numpy_array(self):

        return np.concatenate([x.chunk for x in self.get_chunks()], axis=1)


    def write(self, format='txt', filename='', filehandle=None):
        """
        Export 2D-SIFt data into a file or file handle. Supported formats are "txt" and "yaml".
        File specified with @filename is opened in append mode. 

        The function returns the output string in the specified format.
        """
        
        fh = None
        if filename != '':
            fh = open(filename, 'a')
        elif filehandle is not None:
            fh = filehandle
        try:
            interaction_matrix = np.concatenate([x.chunk for x in self._chunks], axis=1)
            fp_string = ';'.join([''.join([str(x) for x in interaction_matrix[y,:]]) for y in range(interaction_matrix.shape[0])])
            if fh is not None:
                fh.write(output_format[format].substitute(receptor=self.receptor_name, ligand=self.ligand_name,start=self.start,fp=fp_string) + '\n')
        except KeyError:
                print("The specified format is not supported {!s}".format(format))

        if filename != '':
            fh.close()
        return output_format[format].substitute(receptor=self.receptor_name, ligand=self.ligand_name,start=self.start,fp=fp_string)


    def get_heatmap(self, selected_chunks=None, colormap=plt.cm.Reds, vmin=None, vmax=None):
        """
        Returns a standarized heatmap figure.
        """
        heatmap = plt.figure(figsize=(20,5))
        ax = heatmap.gca(xlabel="Residue number", ylabel="Pharmacophore features")
        ax.set_yticks(np.arange(len(rows))+0.5, minor=False)
        ax.set_yticklabels(['D','A','H','N','P','R','vdW'], size='large')
        ax.invert_yaxis()
        ax.set_xlabel("Residue number", size='large')
        ax.set_ylabel("Pharmacophore features", size='large')

        if selected_chunks:
            ax.set_xticks(np.arange(len(selected_chunks))*len(columns)+0.5*len(columns), minor=False)
            ax.set_xticklabels(["{}".format(x) for x in selected_chunks], size='large', rotation="vertical")
            if self._mapping:
                coll = plt.pcolor(np.concatenate([x.chunk for x in [self._chunks[self._mapping[x]] for x in selected_chunks]], axis=1), cmap=colormap, figure=heatmap, vmin=vmin, vmax=vmax)
            else:
                coll = plt.pcolor(np.concatenate([x.chunk for x in [self._chunks[x - self.start] for x in selected_chunks]], axis=1), cmap=colormap, figure=heatmap, vmin=vmin, vmax=vmax)
        else:
            coll = plt.pcolor(np.concatenate([x.chunk for x in self._chunks], axis=1), cmap=colormap, figure=heatmap, vmin=vmin, vmax=vmax)
        ax.grid(b=True, which='minor', axis='x')
        heatmap.colorbar(coll)

        return heatmap


#==============================================================================  
class SIFt2DGenerator:
    """
    The class generating 2D-SIFt matrices for a given set of ligands and a receptor.
    The SIFt bits are generated for every residue in the input file, however the output file contains starting residue number.

    """
        
    generic_chunk = SIFt2DChunk()    


    def __init__(self, rec_struct, ligs, cutoff = 3.5, use_generic_numbers=False, unique=False, property=None):
    
        self.sifts = [] #List of SIFt2D objects
        self.starting_res_num = 0
        self.ending_res_num = 0
        self.seq_len = 0
        self._lig_list = []
        self._rings = []
        self._generic_numbers = None
        
        self.receptor_st = rec_struct
        self.cutoff = cutoff
        if use_generic_numbers:
            self.receptor_st = rec_struct.extract(analyze.get_atoms_from_asl(rec_struct, "fillres (a. CA and a.pdbbfactor > -8.1 and a.pdbbfactor < 8.1)"))
            self._generic_numbers = self._get_generic_numbers_list()
            self.receptor_st.title = rec_struct.title
        self._get_receptor_params()
                
        self.backbone_set = set(analyze.evaluate_asl(self.receptor_st, 
                                                     "backbone"))
        for num, lig in enumerate(ligs):
            if property:
                if unique and lig.property[property] in self.lig_list:
                    continue
                self.sifts.append(SIFt2D(lig.property[property], self.receptor_st.title, self.starting_res_num, self.ending_res_num, self._generic_numbers))
                print("Working on ligand {!s} {:n}".format(lig.property[property], num))
            else:
                if unique and lig.title in self.lig_list:
                    continue
                self.sifts.append(SIFt2D(lig.title, self.receptor_st.title, self.starting_res_num, self.ending_res_num, self._generic_numbers))
                print("Working on ligand {!s} {:n}".format(lig.title, num))
         
            try:
                self.generate_2d_sift(lig, self.sifts[-1])
            except Exception as msg:
                print(msg)
                continue
            if unique and property:
                self.lig_list.append(lig.property[property])
            elif unique:
                self.lig_list.append(lig.title)
            

    def __iter__(self):
        """
        Iterator over all 2DSIFts
        """
        for sift in self.sifts:
            yield sift


    def _get_receptor_params(self):
        """
        Extract starting and ending residues from structure. Prepare rings for finding pi-pi and pi-cation interactions(speedup trick). Private function.
        """
        if self._generic_numbers:
            self.starting_res_num = min(self._generic_numbers)
            self.ending_res_num = max(self._generic_numbers)
            self.seq_len = len(self._generic_numbers)
        else:
            self.starting_res_num = min([x.resnum for x in self.receptor_st.residue])
            self.ending_res_num = max([x.resnum for x in self.receptor_st.residue])
            self.seq_len = self.ending_res_num - self.starting_res_num
        self._rings = interactions.gather_rings(self.receptor_st)
        

    def _get_generic_numbers_list(self):
        """
        Extract the list of generic numbers assigned to the input receptor. Following GPCRdb convention, GPCRdb generic numbers are stored as b factors of CA atom of each residue.
        """
        return map(lambda x: (x.temperature_factor > 0 and x.temperature_factor) or (x.temperature_factor < 0 and -x.temperature_factor + 0.001), analyze.get_atoms_from_asl(self.receptor_st, "a. CA and a.pdbbfactor > -8.1 and a.pdbbfactor < 8.1"))


    def _get_generic_number(self, residue_number):
        """
        Retrieve GPCRdb generic number stored in CA temperature factor.
        """
        return (lambda x: (x > 0 and x) or (x < 0 and -x + 0.001)) (analyze.get_atoms_from_asl(self.receptor_st, "a. CA and r. {!s}".format(residue_number)).next().temperature_factor)


    def generate_2d_sift(self, ligand, matrix):
        """
        Find interactions and encode the interaction matrix.
        """
        features = self.assign_pharm_feats(ligand)
        for residue in self.receptor_st.residue:
            for ftype in features.keys():
                if ftype == 'AROMATIC':
                    continue
                for feat in features[ftype]:
                    active_bits = self.find_feature_interactions(ftype, feat, ligand, residue)
                    if self._generic_numbers:
                        self.activate_bits(active_bits, ftype, self._get_generic_number(residue.resnum), matrix)
                    else:
                        self.activate_bits(active_bits, ftype, residue.resnum, matrix)
        #Aromatic interactions
        pipi_int = interactions.find_pi_pi_interactions(self.receptor_st, rings1=self._rings, struct2=ligand)
        #ring1 and ring2 are Centroid objects, ring1 comes from receptor, ring2 from ligand, atoms in Centroid.atoms list are sorted
        for ppi in pipi_int:
            for aromatic_feat in features['AROMATIC']:
                if ppi.ring2.atoms == sorted(aromatic_feat):
                    if self._generic_numbers:
                        self.activate_bits(['AROMATIC'], 'AROMATIC', self._get_generic_number(self.receptor_st.atom[ppi.ring1.atoms[0]].resnum), matrix)
                    else:
                        self.activate_bits(['AROMATIC'], 'AROMATIC', self.receptor_st.atom[ppi.ring1.atoms[0]].resnum, matrix)
        picat_int = interactions.find_pi_cation_interactions(self.receptor_st, struct2=ligand)
        for pci in picat_int:
            if pci.cation_structure.title == self.receptor_st.title:
                if self._generic_numbers:
                    self.activate_bits(['CHARGED'], 'AROMATIC', self._get_generic_number(self.receptor_st.atom[pci.cation_centroid.atoms[0]].resnum), matrix)
                else:
                    self.activate_bits(['CHARGED'], 'AROMATIC', self.receptor_st.atom[pci.cation_centroid.atoms[0]].resnum, matrix)
            else:
                if self._generic_numbers:
                    self.activate_bits(['AROMATIC'], 'P_CHARGED', self._get_generic_number(self.receptor_st.atom[pci.pi_centroid.atoms[0]].resnum), matrix)
                else:
                    self.activate_bits(['AROMATIC'], 'P_CHARGED', self.receptor_st.atom[pci.pi_centroid.atoms[0]].resnum, matrix)


    def find_feature_interactions(self, feat_name, feat_atoms, ligand_st, residue):
        """
        The function evaluates type/distance-based interactions and h_bonds. Aromatic interactions are treated globally (on the receptor level), not on the atomic level.
        """
        rec_act_bits = []
        
        for ratom in residue.getAtomList():
            for atom in feat_atoms:
                atom_atom_dist = measure.measure_distance(ligand_st.atom[atom],  self.receptor_st.atom[ratom])
                if atom_atom_dist <= self.cutoff:
                    if feat_name == 'ANY':
                        rec_act_bits.append('ANY')                
                        if int(ratom) in self.backbone_set:
                            rec_act_bits.append('BACKBONE')
                        else:
                            rec_act_bits.append('SIDECHAIN')
                        continue
                    #if feat_name == 'ANY':
                    #    continue
                    if analyze.match_hbond(ligand_st.atom[atom], self.receptor_st.atom[ratom], distance_max=2.8, distance=atom_atom_dist):
                        if self.receptor_st.atom[ratom].atomic_number == 1:
                            rec_act_bits.append('H_DONOR')
                        else:
                            rec_act_bits.append('H_ACCEPTOR')
                        if int(ratom) in self.backbone_set:
                            rec_act_bits.append('BACKBONE')
                        else:
                            rec_act_bits.append('SIDECHAIN')

                    if feat_name not in interaction_rules.keys():
                        continue
                    for r_feat in interaction_rules[feat_name][interaction_dict['residue_features']]:
                        if residue.pdbres.strip() in residue_sets[r_feat]:
                            rec_act_bits.extend(interaction_rules[feat_name][interaction_dict['byte_to_switch']])
        return list(set(rec_act_bits))


    def activate_bits(self, act_bits, feat, resnum, matrix):
        """
        Increment the value of residue chunk for given interaction.
        """
        for bit in act_bits:
            matrix[resnum].increment_bit(feat, bit)


    def assign_pharm_feats(self, lig_struct):
        """
        SMARTS based assignment of pharmacophore features (using Phase feature patterns). Rings are extracted using Structure.ring iterator.
        """
        matches = {}
        for key in ph_patterns.keys():
            matches[key] = []
            for pattern in ph_patterns[key]:
                s = sorted(analyze.evaluate_smarts_canvas(lig_struct, pattern, uniqueFilter=True))
                if s != []:
                    matches[key] = s
            if matches[key] == []:
                del matches[key]
        matches['AROMATIC'] = [x.getAtomList() for x in lig_struct.ring]
        matches['ANY'] = [[x.index for x in lig_struct.atom]]
        return matches


    def write_all(self, outfile=None, mode='w'):
        """
        Batch saving all 2D-SIFt object stored in generator.
        """

        print("Writing interaction matrix to a file.")
        if outfile is None:
            outfile = '{!s}_2dfp.dat'.format(self.receptor_st.title)
        outfh = open(outfile, mode)
        if self._generic_numbers:
            outfh.write("#{!s}\n".format(';'.join([str(x) for x in sorted(self._generic_numbers)])))

        for sift in self.sifts:
            sift.write(filehandle=outfh)
        outfh.close()


#==============================================================================  
class SIFt2DReader:
    """
    Class for reading stored interaction matrices. Returning iterator of SIFt2D objects read.
    """


    def __init__(self, filename=None, filehandle=None, input_string=None, format='txt'):
        
        self._sifts = []
        self.custom_residues_set = None

        if not filename and not filehandle and not input_string:
            raise IOError("Some input must be specified, file name, file handle or input string.")
        if format not in output_format.keys():
            raise IOError("Specified format is not supported.")
        if filename:
            fh = open(filename, 'r')
            self._sifts = self.fh_reader(fh, format)
        elif filehandle:
            self._sifts = self.fh_reader(filehandle, format)
        else:
            self._sifts = self.string_reader(input_string, format)


    def fh_reader(self, fh, format):

        output = []

        resi_defs = fh.readline()
        if resi_defs.startswith('#'):
            self.custom_residues_set = resi_defs[1:].strip().split(';')
        else:
            output.append(self.format_parser(resi_defs, format))
        for line in fh:
            if line.strip() == '':
                continue
            output.append(self.format_parser(line.strip(), format))

        return output


    def format_parser(self, line, format):

        if format == 'txt':
            receptor, ligand, start, im = line.strip().split(':')
            start = int(math.floor(float(start)))
            im = im.split(';')
            if len(im) != len(rows.keys()):
                raise IOError("Shape of input interaction matrix does not correspond to defined shape.")
            end = start + len(im[0][::9]) - 1
            im = np.array([[int(x) for x in y] for y in im])
            chunks = np.hsplit(im, len(im[0][::9]))
        if self.custom_residues_set:
            output = SIFt2D(ligand, receptor, start, end, self.custom_residues_set)
            for num, chunk in zip(self.custom_residues_set, chunks):
                output.set_chunk(SIFt2DChunk(num, chunk), num, True)
        else:
            output = SIFt2D(ligand, receptor, start, end)
            for num, chunk in enumerate(chunks):
                output.set_chunk(SIFt2DChunk(start+num, chunk), start+num)
        return output


    def __len__(self):

        return len(self._sifts)


    def __iter__(self):
        """
        Iterate over read SIFt2D objects.
        """
        for sift in self._sifts:
            yield sift



#==============================================================================  
class SIFt2DProfile(SIFt2D):
    """
    Class representing averaged SIFt2D objects.
    """

    def __init__(self, sifts=None, start=None, end=None, generic_numbers = None):
        SIFt2D.__init__(self, '', 'average', 0, 0, generic_numbers)
        self._sifts = sifts
        if sifts:
            if not generic_numbers:
                self.start, self.end = self._get_residue_range()
        elif start and end:
            self.start = start
            self.end = end
        self._chunks = []
        self._init_chunks()
    

    def _get_residue_range(self):
        """
        Return starting and ending residue for given set of input interaction matrices.
        """       
        start = min([x.start for x in self._sifts])
        end = max([x.end for x in self._sifts])
        if start > end:
            raise ValueError("Starting residue greater than ending one. Start: {:n} End {:n}".format(start, end))
        return (start, end)


    def calculate_average(self):

        if self._mapping:
            gn = self.get_common_generic_residues_list()
            #TODO Czy to dobry pomysl?
            if gn == sorted(self._mapping.keys()):
                avg = np.mean([[x.chunk for x in y.get_listed_chunks(gn)] for y in self._sifts], axis=0)
                for gn, x in zip(sorted(self._mapping), avg):
                    self.set_chunk(SIFt2DChunk(gn, x), gn, True)
            else:
                print("Nie w domu, ale tez na d")
                print(gn)
                print(self._mapping.keys())
        else:
            self.normalize_sifts()
            avg = np.mean([[x.chunk for x in y.get_chunks()] for y in self._sifts], axis=0)
            for idx, x in enumerate(avg):
                self.set_chunk(SIFt2DChunk(idx + self.start, x), idx + self.start)


    def get_interacting_chunks(self, cutoff=0.3):
        """
        Return chunks with at least one value above the cutoff.
        """
        return [x for x in self._chunks if x.chunk.max() >= cutoff]


    def get_common_generic_residues_list(self):

        return sorted(list(set(itertools.chain.from_iterable([x.custom_residues_set for x in self._sifts]))))


    def normalize_sifts(self):
        """
        The function evens the lenghts of sift matrices. 
        """

        if self._mapping:
            print("Not yet supported for generic numbers.")
        start = min([x.start for x in self._sifts])
        end = max([x.end for x in self._sifts])
        for sift in self._sifts:
            if sift.start > start:
                for i in range(sift.start - start):
                    sift.prepend_chunk()
            if sift.end < end:
                for i in range(end - sift.end):
                    sift.append_chunk()


    def write(self, format='txt', filename='', filehandle=None):
        """
        Export 2D-SIFt profile data into a file or file handle. Supported formats are "txt" and "yaml".
        File specified tih @filename is opened in append mode. 

        The function returns the output string in the specified format.
        """
        
        fh = None
        if filename != '':
            fh = open(filename, 'w')
        elif filehandle is not None:
            fh = filehandle
        try:
            if self._mapping:
                fh.write("#{!s}\n".format(';'.join([str(x) for x in sorted(self._mapping.keys())])))
            interaction_matrix = np.concatenate([x.chunk for x in self.get_chunks()], axis=1)
            fp_string = ';'.join([' '.join([str(x) for x in interaction_matrix[y,:]]) for y in range(interaction_matrix.shape[0])])
            if fh is not None:
                fh.write(output_format[format].substitute(receptor=self.receptor_name, ligand=self.ligand_name,start=self.start,fp=fp_string))
        except KeyError:
                print("The specified format is not supported {!s}".format(format))

        if filename != '':
            fh.close()
        return output_format[format].substitute(receptor=self.receptor_name, ligand=self.ligand_name,start=self.start,fp=fp_string)


#==============================================================================  
class SIFt2DProfileReader:
    """
    Class for reading interaction profiles
    """


    def __init__(self, filename=None, filehandle=None, input_string=None, format='txt'):

        self._profile = None
        self.custom_residues_set = None

        if not filename and not filehandle and not input_string:
            raise IOError("Some input must be specified, file name, file handle or input string.")
        if format not in output_format.keys():
            raise IOError("Specified format is not supported.")
        if filename:
            fh = open(filename, 'r')
            self._profile = self.fh_reader(fh, format)
        elif filehandle:
            self._profile = self.fh_reader(filehandle, format)
        else:
            self._profile = self.string_reader(input_string, format)


    def fh_reader(self, fh, format):

        output = None
        
        #The profile should contain 2 lines maximum. Anything more will be ignored
        profile_lines = fh.readlines()
        if len(profile_lines) > 1:
            resi_defs = profile_lines[0]
            if resi_defs.startswith('#'):
                self.custom_residues_set = resi_defs[1:].strip().split(';')
                output = self.format_parser(profile_lines[1], format)
            else:
                output = self.format_parser(profile_lines[0], format)
        else:
            output = self.format_parser(profile_lines[0], format)
        return output


    def format_parser(self, line, format):

        output = []
        if format == 'txt':
            receptor, ligand, start, im = line.strip().split(':')
            start = int(math.floor(float(start)))
            im = im.split(';')
            if len(im) != len(rows.keys()):
                raise IOError("Shape of input interaction matrix does not correspond to defined shape.")
            end = start + len(im[0].split(' ')[::9]) - 1
            im = np.array([[float(x) for x in y.split(' ')] for y in im])
            chunks = np.hsplit(im, len(im[0][::9]))
        if self.custom_residues_set:
            output = SIFt2DProfile(generic_numbers=self.custom_residues_set)
            for num, chunk in zip(self.custom_residues_set, chunks):
                output.set_chunk(SIFt2DChunk(num, chunk), num, True)
        else:
            output = SIFt2DProfile(start=start, end=end)
            for num, chunk in enumerate(chunks):
                output.set_chunk(SIFt2DChunk(start+num, chunk), start+num)
        return output


    def get_profile(self):
        return self._profile


#==============================================================================  
def readProfile(filename='', filehandle=None, input_string=None, format='txt'):

    if filename != '':
        print(filename)
        return SIFt2DProfileReader(filename, format=format).get_profile()
    elif filehandle:
        return SIFt2DProfileReader(filehandle=filehandle, format=format).get_profile()
    elif input_string:
        return SIFt2DProfileReader(input_string=input_string, format=format).get_profile()



#==============================================================================  
if __name__ == '__main__':
    print("This is the library for generation and manipulation of the 2D-SIFt descriptors.")