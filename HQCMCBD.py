# Gurobi
import gurobipy as gp
from gurobipy import GRB, Model, quicksum
import numpy as np
import re
import os
import dimod
from dimod.binary.binary_quadratic_model import BinaryQuadraticModel, Binary, Spin, as_bqm
from dimod.discrete.discrete_quadratic_model import DiscreteQuadraticModel
from dimod import ConstrainedQuadraticModel, Binary
from dimod import cqm_to_bqm
#import dwave.inspector
#import dwave.system
from dwave.system import LeapHybridCQMSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler
from dwave.system import EmbeddingComposite
import time
import json

# Bifurcation
import torch
import simulated_bifurcation as sb
#openjij
import openjij as oj


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Function to write config data to a JSON file
def create_config_file(config_file, config_data, round):
    with open(config_file, 'w') as file:
        json.dump(config_data, file, indent=4)
    print(f"The {round}-th Config file of quantum sampling is created successfully at {config_file}.")
    
"""
Part of QUBO Construction
"""    
def add_front(a,b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        
        c[:len(b)] += b  
    return c

def add_end(a,b):
    if len(a) < len(b):
        c = b.copy()
        c[-len(a):] += a
    else:
        c = a.copy()
        c[-len(b):] += b  
    return c

def QUBO_Cons_ineq_normal(QUBO_pre, lhs, symbol, rhs, round_num = None, penalty = None, dictionary = None):
    if penalty == None:
        penalty = 1e3
    disturbance = 1e-6
    QUBO = QUBO_pre.copy()
    # M
    M = rhs
    if symbol == "<=":
        min_value = np.sum(lhs[lhs<0])
        Slack_Var_Add_num = max(0,np.ceil(np.log2(rhs - min_value + disturbance)))
        # Coeff = C for C * s
        Slack_Var_Add_coeff = 2 ** np.arange(Slack_Var_Add_num)
        for slack_index in range(int(Slack_Var_Add_num)):
            if not dictionary == None:
                next_index = len(dictionary)
                dictionary["s_%d_%d"%(round_num, slack_index)] = next_index
            QUBO = np.pad(QUBO, [(0, 1), (0, 1)], mode='constant', constant_values = 0)   #local QUBO arithmaic
        var_matched = np.zeros(len(QUBO))
        #The Final answer of is  (∑Ax + ∑Cs - M)^2
        #  (∑Ax)*(∑Ax) + (∑Cs)*(∑Cs) + (∑Ax)(∑Cs) + (∑Cs)(∑Ax) 
        # - 2(∑MCs) - 2(∑MAx)
        #(Ax)
        var_matched_1 = add_front(lhs, var_matched)
        #(Cs)
        if Slack_Var_Add_num > 0:
            var_matched_2 = add_end(Slack_Var_Add_coeff, var_matched)        
        #(∑Ax)*(∑Ax)
        QUBO += penalty * (np.outer(var_matched_1, var_matched_1))
        if Slack_Var_Add_num > 0:       
            #(∑Cs)*(∑Cs)   
            QUBO += penalty * (np.outer(var_matched_2, var_matched_2))
            #(∑Ax)(∑Cs) + (∑Cs)(∑Ax) 
            QUBO += penalty * (np.outer(var_matched_1, var_matched_2) + np.outer(var_matched_2, var_matched_1))
            # -2(∑MCs) 
            QUBO -= penalty * 2 * M *(np.diag(var_matched_2))
        #- 2(∑MAx)
        QUBO -= penalty * 2 * M *(np.diag(var_matched_1))                  
    else: # symbol == ">=":
        max_value = np.sum(lhs[lhs>0])
        Slack_Var_Add_num = max(0,np.ceil(np.log2(max_value - rhs + disturbance)))
        Slack_Var_Add_coeff = 2 ** np.arange(Slack_Var_Add_num)
        for slack_index in range(int(Slack_Var_Add_num)):
            if not dictionary == None:
                next_index = len(dictionary)
                dictionary["s_%d_%d"%(round_num, slack_index)] = next_index
            QUBO = np.pad(QUBO, [(0, 1), (0, 1)], mode='constant', constant_values = 0)   #local QUBO arithmaic      
        var_matched = np.zeros(len(QUBO))
        #The Final answer of is  (∑Ax - ∑Cs - M)^2 == (∑Ax)*(∑Ax) + (∑Cs)*(∑Cs) - (∑Ax)(∑Cs) - (∑Cs)(∑Ax) + 2(∑MCs) - 2(∑MAx)
        # (Ax)
        var_matched_1 = add_front(lhs, var_matched) 
        # (Cs)
        if Slack_Var_Add_num > 0:
            var_matched_2 = add_end(Slack_Var_Add_coeff, var_matched)        
        # (∑Ax)*(∑Ax)
        QUBO += penalty * (np.outer(var_matched_1, var_matched_1))
        if Slack_Var_Add_num > 0:    
            # (∑Cs)*(∑Cs)   
            QUBO += penalty * (np.outer(var_matched_2, var_matched_2))
            # -(∑Ax)(∑Cs) - (∑Cs)(∑Ax) 
            QUBO -= penalty * (np.outer(var_matched_1, var_matched_2) + np.outer(var_matched_2, var_matched_1))
            # +2(∑MCs) 
            QUBO += penalty * 2 * M *(np.diag(var_matched_2))
        # -2(∑MAx)
        QUBO -= penalty * 2 * M *(np.diag(var_matched_1))        
    #if no dictionary is needed. just do QUBO, _ = function(*args) 
    return QUBO, dictionary

def QUBO_Cons_ineq_0(QUBO_pre, lhs, symbol, rhs, round_num = None, penalty = None, dictionary = None):
    #set up a default penalty just in case ... 
    if penalty == None:
        penalty = 1e3
    QUBO = QUBO_pre.copy()
    modification_region = len(lhs) 
    # Contraint QUBOlize : 
    # Cons = x-xy when x <= y 
    if symbol == "<=":  
        vector_x = lhs == 1
        vector_y = lhs == -1
        QUBO[:modification_region, :modification_region] += penalty * (np.diag(vector_x) - np.outer(vector_x, vector_y))     
    # Cons = y-yx when x >= y
    else: # symbol == ">=":
        vector_x = lhs == -1
        vector_y = lhs == 1
        QUBO[:modification_region, :modification_region] += penalty * (np.diag(vector_x) - np.outer(vector_x, vector_y))     
    #if no dictionary is needed. just do QUBO, _ = function(*args) 
    return QUBO, dictionary

def QUBO_Cons_leq_1(QUBO_pre, lhs, symbol, rhs, round_num = None, penalty = None, dictionary = None):
    """
    Cons = Sum(xy)
           ───┬───
         np.triu(np.outer(lhs, lhs), k = 1)
    """
    #set up a default penalty just in case ... 
    if penalty == None:
        penalty = 1e3
    QUBO = QUBO_pre.copy()
    modification_region = len(lhs) 
    QUBO[:modification_region, :modification_region] += penalty * (np.triu(np.outer(lhs, lhs), k = 1))     
    # if no dictionary is needed. just do QUBO, _ = function(*args) 
    return QUBO, dictionary

def QUBO_Cons_geq_1(QUBO_pre, lhs, symbol, rhs, round_num = None, penalty = None, dictionary = None):
    # Cons = 1 (- x - y) +  xy
    #           ───┬───    ──┬─
    #           diag(arg)    │
    #                        │ 
    #          np.triu( np.outer(lhs, lhs), k = 1)
    #set up a default penalty just in case ... 
    if penalty == None:
        penalty = 1e3
    QUBO = QUBO_pre.copy()
    modification_region = len(lhs)
    QUBO[:modification_region, :modification_region] += penalty * ( np.triu( np.outer(lhs, lhs), k = 1) - np.diag(lhs) )     
    # if no dictionary is needed. just do QUBO, _ = function(*args) 
    return QUBO, dictionary

def QUBO_Cons_eq(QUBO_pre, lhs, rhs, round_num = None, penalty = None, dictionary = None):
    # Let assume  Ax = b =====> lhs * x = rhs
    #
    # Cons =    x'A'Ax   -   2Abx + (b^2) constant neglect 
    #           ───┬───      ──┬─
    #      np.outer(lhs,lhs)   │
    #                          │ 
    #                 2 * rhs * np.diag(lhs)
    #set up a default penalty just in case ... 
    if penalty == None:
        penalty = 1e3
    QUBO = QUBO_pre.copy()
    modification_region = len(lhs)
    QUBO[:modification_region, :modification_region] += penalty * ( np.outer(lhs,lhs) - 2 * rhs * np.diag(lhs) )     
    # if no dictionary is needed. just do QUBO, _ = function(*args) 
    return QUBO, dictionary

# QUBO_Cons_selection blue print
# if symbol == "="
# function(QUBO_Cons_eq)
#
# if not (normalize?)     
#  rhs  other   1   0
#        ─┬    ─┬  ─┬  
#         │     │   │  
#         │     │   └┤ lhs Bin && np.sum(lhs) == 0 && np.sum(abs(lhs))== 2 ? ─┬┤yes: function(QUBO_Cons_ineq_0)
#         │     │                                                             │ 
#         │     │                                                             └┤No: function(QUBO_Cons_ineq_normal)
#         │     │
#         │     └────┤ lhs Bin && symbol == "<=" ? &&─┬────┤yes: function(QUBO_Cons_leq_1)
#         │                                           │                                               
#         │                                           └────┤No (symbol == ">=") if lhs Bin && np.sum(lhs)== 2   ─┬┤yes: function(QUBO_Cons_geq_1)
#         │                                                                                                      │ 
#         │                                                                                                      └┤No: function(QUBO_Cons_ineq_normal)
#         └──────────┤ function(QUBO_Cons_ineq_normal)

def QUBO_Cons_selection(QUBO_pre, lhs, symbol, rhs, round_num = None, penalty = None, dictionary = None):
    if symbol == "=":
        QUBO, dictionary = QUBO_Cons_eq(QUBO_pre, lhs, rhs, round_num, penalty, dictionary)
    elif rhs == 0:
        if (np.sum(lhs) == 0) and (np.sum(abs(lhs))== 2):
            QUBO, dictionary = QUBO_Cons_ineq_0(QUBO_pre, lhs, symbol, rhs, round_num, penalty, dictionary)
        else:
            QUBO, dictionary = QUBO_Cons_ineq_normal(QUBO_pre, lhs, symbol, rhs, round_num, penalty, dictionary)
    elif rhs == 1:
        if (np.array_equal(lhs, lhs.astype(bool))) and (symbol == "<=") :
            QUBO, dictionary = QUBO_Cons_leq_1(QUBO_pre, lhs, symbol, rhs, round_num, penalty, dictionary)           
        elif (np.array_equal(lhs, lhs.astype(bool))) and (len(lhs) == 2) :
            QUBO, dictionary = QUBO_Cons_geq_1(QUBO_pre, lhs, symbol, rhs, round_num, penalty, dictionary)
        else:
            QUBO, dictionary = QUBO_Cons_ineq_normal(QUBO_pre, lhs, symbol, rhs, round_num, penalty, dictionary)
    else:
        QUBO, dictionary = QUBO_Cons_ineq_normal(QUBO_pre, lhs, symbol, rhs, round_num, penalty, dictionary)    
    return QUBO, dictionary    

"""
Part HQCMCBD_algorithm (HQC-Bend)
"""

class HQCMCBD_algorithm:
    
    def __init__(self, m, *args, **kwargs):
        # Initialize positional arguments
        mode = kwargs.get('mode')    
        if mode == "manual":
            config_file = 'config.json'
        elif mode == "default":
            config_file = 'config_default.json'  
        else:
            print("Invalid Input")
        config = load_config(config_file)
        lambda_config = config.get("lambda_var")
        submethod = config.get("submethod")
        debug_mode = config.get("debug_mode")
        Hybrid_mode = config.get("Hybrid_mode")
        dwave_config = config.get("dwave")
        threshold_config = config.get("threshold")
        self.lambda_nneg_bit = lambda_config.get("nonneg_bits_length")
        self.lambda_dec_bit = lambda_config.get("decimal_bits_length")
        self.lambda_neg_bit = lambda_config.get("negative_bits_length")
        self.sub_method = submethod
        self.flag_print = debug_mode
        self.Hybrid_mode = Hybrid_mode
        self.threshold_type = threshold_config.get("type")
        self.threshold_gap = float(threshold_config.get("gap"))
        self.max_steps = int(config.get("max_steps"))
        self.Msense = 0
        
        if self.Hybrid_mode:
            self.dwave_solver = dwave_config.get("mode")
            self.dwave_token = str(dwave_config.get("DWave_token"))
            self.MC_flag = dwave_config.get("Mcut_flag")
            self.num_of_read = dwave_config.get("num_of_read")
            if self.MC_flag:
                self.mcut_num = int(dwave_config.get("Cutnums"))
            else:
                self.mcut_num = 1
        else:
            self.mcut_num = 1

        current_folder = os.getcwd()
        self.data_folder = os.path.join(current_folder, "data_output")
        os.makedirs(self.data_folder, exist_ok=True)
        print(f"Folder '{self.data_folder}' has been created or already exists for storing the json data")
        self.LP_folder = os.path.join(current_folder, "LP_output")
        os.makedirs(self.LP_folder, exist_ok=True)
        self.preprocessing(m)

    def preprocessing(self, m):
        self.lambda_bits = self.lambda_nneg_bit + self.lambda_dec_bit + self.lambda_neg_bit
        self.lambda_coeff = np.append(-2**np.arange(1, self.lambda_neg_bit + 1, dtype=float), 2**np.arange(-self.lambda_dec_bit, self.lambda_nneg_bit, dtype=float))
        self.master_constraint_dict = {}
        
        # SET up the binary decision Var
        self.Bin_varname = []
        self.Cont_varname = []
        self.Cont_UB = []
        self.free_Cont_varindex = []
        for i, item in enumerate(m.getVars()):
            if item.vtype == GRB.BINARY:
                self.Bin_varname.append(item.VarName)
            else:
                self.Cont_varname.append(item.VarName)
                self.Cont_UB.append(item.ub)
            if item.LB < 0:
                ### negative cont var is not allowed
                print(item.VarName, "is a negative continuous variable, please let it be non-negative continuous variable")
                
        self.num_vars = m.numVars
        self.num_binvars = len(self.Bin_varname)
        self.num_constrs = m.numConstrs

        # Initialize the constraint matrix, objective coefficients, and RHS
        self.obj_c = np.zeros(self.num_binvars)
        self.obj_d = np.zeros(len(self.Cont_varname))
        self.rhs = np.zeros(self.num_constrs)
        self.relation = []

        # Populate the objective coefficients
        for j, v in enumerate(m.getVars()):
            if v.VarName in self.Bin_varname:
                self.obj_c[self.Bin_varname.index(v.VarName)] = v.Obj
            else:
                self.obj_d[self.Cont_varname.index(v.VarName)] = v.Obj
        if m.ModelSense == GRB.MAXIMIZE:
            self.obj_c = -self.obj_c
            self.obj_d = -self.obj_d
            self.Msense =  1
        
        # Master problem lambda bits        
        for i in range(self.lambda_bits):
            self.Bin_varname.append(f't_bits[{i}]')
        
        self.A = np.zeros((self.num_constrs, len(self.Bin_varname)))
        self.G = np.zeros((self.num_constrs, len(self.Cont_varname)))
        self.nonA_index = []
        
        for i, constr in enumerate(m.getConstrs()):
            expr = m.getRow(constr) # Get the expression for this constraint
            constrs_type_list = [expr.getVar(i).Vtype == 'B' for i in range(expr.size())] #if all binary continue
            if all(constrs_type_list): 
                self.nonA_index.append(i)
            self.rhs[i] = constr.RHS      # Get the RHS value for this constraint
            self.relation.append(constr.Sense)
            for j in range(expr.size()):
                if expr.getVar(j).VarName in self.Bin_varname:
                    self.A[i, self.Bin_varname.index(expr.getVar(j).VarName)] = expr.getCoeff(j)  # Get coefficient of var in this constraint
                else:
                    self.G[i, self.Cont_varname.index(expr.getVar(j).VarName)] = expr.getCoeff(j)  # Get coefficient of var in this constraint
            
        self.A_map = np.zeros((len(self.nonA_index) , self.num_binvars))
        self.rhs_map = np.zeros(len(self.nonA_index))    
        self.A_sub = np.zeros(( self.num_constrs, len(self.Bin_varname) - self.lambda_bits))
        self.G_sub = np.zeros(( self.num_constrs, len(self.Cont_varname) ))
        self.rhs_sub = np.zeros(self.num_constrs)
        self.sub_relation = []
        self.map_relation = []
        self.eq_constraint = []
        for row, symb in enumerate(self.relation):
            if symb == GRB.LESS_EQUAL: 
                multiplier = -1         
                if row in self.nonA_index:
                    self.map_relation.append(GRB.GREATER_EQUAL) 
                else:
                    self.sub_relation.append(GRB.GREATER_EQUAL) 
            elif symb == GRB.EQUAL:
                multiplier = 1
                if row in self.nonA_index:
                    self.map_relation.append(symb)
                else:
                    self.sub_relation.append(symb)
            else:
                multiplier = 1
                if row in self.nonA_index:
                    self.map_relation.append(symb)
                else:
                    self.sub_relation.append(symb)         
            if row in self.nonA_index:
                self.A_map[self.nonA_index.index(row), :self.num_binvars] = multiplier * self.A[row, :self.num_binvars]
                self.rhs_map[self.nonA_index.index(row)] = multiplier * self.rhs[row]                
            else:
                self.A_sub[row, :] = multiplier * self.A[row, :- self.lambda_bits]
                self.G_sub[row, :] = multiplier * self.G[row, :]
                self.rhs_sub[row] = multiplier * self.rhs[row]

        self.A_sub = np.delete(self.A_sub, self.nonA_index, axis=0)
        self.G_sub = np.delete(self.G_sub, self.nonA_index, axis=0)
        self.rhs_sub = np.delete(self.rhs_sub, self.nonA_index)
        
        for row, symb in enumerate(self.sub_relation):
            if symb == GRB.EQUAL:
                self.eq_constraint.append(row)
                
        self.lambda_upper = 1e30
        self.lambda_lower = -1e30
        self.obj_value = 0
        
    def extract_parts(self, input_string):
        # Define the regex pattern with capturing groups
        pattern = r"^([^\[\]]+)\[(\d+)\]$"
        # Use re.match to check if the input string matches the pattern and extract groups
        match = re.match(pattern, input_string)
        if match:
            # Extract the %s and %d parts
            string_part = match.group(1)
            integer_part = int(match.group(2))
            return string_part, integer_part
        else:
            return None
        
    def bin_name(self):
        bin_name_list = []
        self.bin_var_count = {}
        for item in self.Bin_varname:
            name, index = self.extract_parts(item)
            if name in bin_name_list:
                old_index = self.bin_var_count[name]
                self.bin_var_count[name] = max(old_index, index + 1)
            else:
                bin_name_list.append(name)
                self.bin_var_count[name] = index + 1

    def build_cqm_master_problem(self):
        self.bin_x_cqm = [dimod.Binary(f'bin_{i}') for i in range(self.num_binvars)] 
        self.bin_lambda_cqm = [dimod.Binary(f'lambda_bin_{i}') for i in range(self.lambda_bits)]
        self.cqm = dimod.ConstrainedQuadraticModel()
        self.cqm.set_objective(sum(self.bin_x_cqm[i] * self.obj_c[i] for i in range(self.num_binvars)) + \
                          sum(self.bin_lambda_cqm[j] * self.lambda_coeff[j] for j in range(self.lambda_bits)))
        
        for order in range(len(self.A_map)):
            if self.map_relation[order] == GRB.EQUAL:
                self.cqm.add_constraint( sum(self.A_map[order, i] * self.bin_x_cqm[i] for i in range(self.num_binvars)) == self.rhs_map[order],\
                    label=f'CQM_MAP_constraint_{order}')
            elif self.map_relation[order] == GRB.GREATER_EQUAL:
                self.cqm.add_constraint( sum(self.A_map[order, i] * self.bin_x_cqm[i] for i in range(self.num_binvars)) >= self.rhs_map[order],\
                    label=f'CQM_MAP_constraint_{order}')
            elif self.map_relation[order] == GRB.LESS_EQUAL:
                self.cqm.add_constraint( sum(self.A_map[order, i] * self.bin_x_cqm[i] for i in range(self.num_binvars)) <= self.rhs_map[order],\
                    label=f'CQM_MAP_constraint_{order}')

    def build_gurobi_master_problem(self):
        self.MAP = gp.Model("Master_Problem")
        # Set parameters to control output
        self.MAP.setParam('LogToConsole', 0)  # This disables all console output from Gurobi
        self.MAP.setParam("OutputFlag", 0)
        self.MAP.setParam("InfUnbdInfo", 1)
        self.bin_x = self.MAP.addMVar(shape= self.num_binvars, vtype=GRB.BINARY, name="bin_x")
        self.bin_lambda = self.MAP.addMVar(shape= self.lambda_bits, vtype=GRB.BINARY, name="bin_lambda")
        # Set the objective function
        self.MAP.setObjective(self.obj_c @ self.bin_x + self.lambda_coeff @ self.bin_lambda , GRB.MINIMIZE)
        # Set the MAP Constr
        self.MAP.addMConstr(self.A_map, self.bin_x, self.map_relation, self.rhs_map, name=f'MAP_init_constraints')

    def build_QUBO_master_problem(self):
        num_map_constrs =  len(self.map_relation)
        Qubo_obj = np.concatenate((self.obj_c, self.lambda_coeff))
        self.bifurcation_QUBO = np.diag(Qubo_obj)
        self.qubo_slack_dict = {}
        for const_ in range(num_map_constrs):
            lhs = np.concatenate((self.A_map[const_] , np.zeros_like(self.lambda_coeff)))
            self.bifurcation_QUBO, self.qubo_slack_dict  = QUBO_Cons_selection(self.bifurcation_QUBO, lhs, self.map_relation[const_],\
                                                                                self.rhs_map[const_], round_num= 0, penalty=10, \
                                                                                    dictionary = self.qubo_slack_dict)

    def solve_master_problem(self, count=0):
        if self.dwave_solver == "cqm":
            print("Solver: ", self.dwave_solver)
            if self.MC_flag == False:
                self.cqm_solve_for_value(count)
            else:
                self.cqm_solve_for_value_mul_cuts(count)      
        elif self.dwave_solver == "bqm_hybrid":
            print("Solver: ", self.dwave_solver)
            self.hbqm_solve_for_value(count)
        elif self.dwave_solver == "bqm_quantum":
            print("Solver: ", self.dwave_solver)
            if self.MC_flag == False:
                self.qbqm_solve_for_value(count)
            else:
                self.qbqm_solve_for_value_mul_cuts(count)   
        elif self.dwave_solver == "bifurcation":   
            print("Solver: ", self.dwave_solver)
            self.bifurcation_solve_for_value_classic(count)
        elif self.dwave_solver == "openjij":   
            print("Solver: ", self.dwave_solver)
            self.openjij_solve_for_value(count)
        else:
            print("The quantum solver name is invalid")

    def hbqm_solve_for_value(self, count=0):
        bqm, _ = dimod.cqm_to_bqm(self.cqm)

        st = time.time()
        sampler = LeapHybridSampler(token = self.dwave_token)
        et = time.time()
        exe_time = (et - st) * 1000

        sampleset = sampler.sample(bqm, label="Master_problem")
        best = sampleset.first
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))
        
        self.obj_value = sampleset.first.energy
        local_lambda_lower = 0
        for key, val in best.sample.items():
            if key.startswith("bin_"):
                index = int(key.replace("bin_",""))
                self.rearranged_bin_mat[1][index] = val  
            elif key.startswith("lambda_bin_"):
                index = int(key.replace("lambda_bin_",""))
                local_lambda_lower += val * self.lambda_coeff[index]
        self.lambda_lower = local_lambda_lower

        Dwave_data = {}
        Dwave_data["info"] = sampleset.info
        Dwave_data["wait_id"] = sampleset.wait_id()
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Dwave_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Dwave_hbqm_info-round-{count}.json")
        create_config_file(dest_path, Dwave_data, count)

    def qbqm_solve_for_value(self, count=0):
        bqm, _ = dimod.cqm_to_bqm(self.cqm)
        # OBJ # Select a solver
        sampler = EmbeddingComposite(DWaveSampler(token = self.dwave_token))
        # Measure Exe time
        st = time.time()
        sampleset = sampler.sample(bqm, num_reads=self.num_of_read, label="Master_problem_HQCBD_bqm")
        et = time.time()
        exe_time = (et - st) * 1000

        best = sampleset.first
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))
        self.obj_value = sampleset.first.energy
        
        local_lambda_lower = 0
        for key, val in best.sample.items():
            if key.startswith("bin_"):
                index = int(key.replace("bin_",""))
                self.rearranged_bin_mat[1][index] = val
            
            elif key.startswith("lambda_bin_"):
                index = int(key.replace("lambda_bin_",""))
                local_lambda_lower += val * self.lambda_coeff[index]
        self.lambda_lower = local_lambda_lower

        Dwave_data = {}
        Dwave_data["info"] = sampleset.info
        Dwave_data["wait_id"] = sampleset.wait_id()
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Dwave_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Dwave_qbqm_info-round-{count}.json")
        create_config_file(dest_path, Dwave_data, count)
    
    def qbqm_solve_for_value_mul_cuts(self, count=0):
        bqm, invert = dimod.cqm_to_bqm(self.cqm)
        # OBJ   # Select a solver
        sampler = EmbeddingComposite(DWaveSampler(token = self.dwave_token))
        st = time.time()
        sampleset = sampler.sample(bqm, num_reads=self.num_of_read, label="HQCBD-mul-cuts_bqm")
        et = time.time()
        exe_time = (et - st) * 1000
        
        data_length = len(sampleset.record.energy)        
        if self.mcut_num > data_length:
            print(f"numberof reads from DWave is less than MCUT, We will use {data_length} cut in this round")
        self.final_mcut_num = min(self.mcut_num , data_length)
        self.obj_value = np.min(sampleset.record.energy[:self.final_mcut_num])
        best_data = sampleset.data()
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))

        Dwave_data = {}
        Dwave_data["info"] = sampleset.info
        Dwave_data["wait_id"] = sampleset.wait_id()
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Dwave_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Dwave_qbqm_info-round-{count}.json")
        create_config_file(dest_path, Dwave_data, count)

        lambda_lower_list = []
        i = 0
        for datum in best_data:
            local_lambda_lower = 0
            for key, val in datum.sample.items():
                if key.startswith("bin_"):
                    index = int(key.replace("bin_",""))
                    self.rearranged_bin_mat[i][index] = val
                elif key.startswith("lambda_bin_"):
                    index = int(key.replace("lambda_bin_",""))
                    local_lambda_lower += val * self.lambda_coeff[index]
            
            lambda_lower_list.append(local_lambda_lower)
            i += 1
            if i >= self.mcut_num:
                break
        self.lambda_lower = max(lambda_lower_list)
    
    def bifurcation_solve_for_value(self, count=0, penalty = 1):
        with open(f'LP_output_dwave\LP_output_MAP_{count}.lp', 'w') as f:
            dimod.lp.dump(self.cqm, f)
        bqm, inverter = cqm_to_bqm(self.cqm, penalty)
        bin_var_dictionary = inverter.to_dict()['binary']
        for index, key in enumerate(bin_var_dictionary.keys()):
            bin_var_dictionary[key] =index
        var_list = list(bin_var_dictionary.keys())
        self.bifurcation_var_list = var_list
        lin, (row, col, quad), _ = bqm.to_numpy_vectors(variable_order=var_list, sort_indices=False)
        size = len(lin)
        self.bifurcation_QUBO = np.zeros((size, size))
        np.fill_diagonal(self.bifurcation_QUBO, lin)
        self.bifurcation_QUBO[row, col] = quad
        #np.save(f"LP_output_dwave\bifurcation_round_{count}.npy", self.bifurcation_QUBO) # maybe for future use
        Q_tensor = torch.from_numpy(self.bifurcation_QUBO)

        st = time.time()
        best_vector, _ = sb.minimize(Q_tensor, max_steps=1e5 , agents = 2048, input_type="binary",device="cuda")
        et = time.time()
        exe_time = (et - st) * 1000
        
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))
        best_bin_vector = best_vector.cpu().numpy()
        np.save(f"LP_output_dwave\LP_output_BF_MAP_{count}.npy", best_bin_vector)
        x_keys = [key for key in bin_var_dictionary if key.startswith('bin_')]
        lambda_keys = [f"lambda_bin_{i}" for i in range(self.lambda_bits)]
        # Fill x matrix
        for key in x_keys:
            idx = bin_var_dictionary[key]
            self.rearranged_bin_mat[0][idx] = best_bin_vector[idx]
        # Extract lambda values
        # lambda_values = np.array([best_bin_vector[bin_var_dictionary[key]] for key in lambda_keys])
        # Calculate lambda_connection
        self.lambda_lower = np.sum(self.lambda_coeff[i] * best_bin_vector[bin_var_dictionary[key]] for i, key in enumerate(lambda_keys))

        self.obj_value = sum(best_bin_vector[i] * self.obj_c[i] for i in range(self.num_binvars)) + self.lambda_lower
                        
        Bifurcation_data = {}
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Bifurcation_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Bifurcation_info-round-{count}.json")
        create_config_file(dest_path, Bifurcation_data, count)
            
    def bifurcation_solve_for_value_classic(self, count=0, penalty = 10):
        #np.save(f"LP_output_dwave\bifurcation_round_{count}.npy", self.bifurcation_QUBO) # maybe for future use
        Q_tensor = torch.from_numpy(self.bifurcation_QUBO)

        st = time.time()
        best_vector, best_val = sb.minimize(Q_tensor, max_steps=5e5 , agents = 128, input_type="binary",device="cuda")
        et = time.time()
        exe_time = (et - st) * 1000

        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))
        best_bin_vector = best_vector.cpu().numpy()
        np.save(f"LP_output_dwave\LP_output_BF_MAP_{count}.npy", best_bin_vector)
        
        # Fill x matrix
        self.rearranged_bin_mat[0] = best_bin_vector[:self.num_binvars]
        # Extract lambda values
        lambda_values = np.array(best_bin_vector[self.num_binvars : self.num_binvars + self.lambda_bits])

        # Calculate lambda_connection
        self.lambda_lower = np.sum([self.lambda_coeff[i] * lambda_values[i] for i in range(self.lambda_bits)])
        self.obj_value = sum(best_bin_vector[i] * self.obj_c[i] for i in range(self.num_binvars)) + self.lambda_lower
        
        Bifurcation_data = {}    
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Bifurcation_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Bifurcation_info-round-{count}.json")
        create_config_file(dest_path, Bifurcation_data, count)

    def openjij_solve_for_value(self, count=0, penalty = 10):
        #np.save(f"LP_output_dwave\Openjij_round_{count}.npy", self.bifurcation_QUBO) # maybe for future use
        qubo_dict = {(i, j): self.bifurcation_QUBO[i, j]
             for i in range(self.bifurcation_QUBO.shape[0])
             for j in range(self.bifurcation_QUBO.shape[1])
             if self.bifurcation_QUBO[i, j] != 0.0}

        # create QUBO sampler in openjij
        sampler = oj.CSQASampler()

        st = time.time()
        response = sampler.sample_qubo(qubo_dict, num_reads=self.num_of_read)
        et = time.time()
        exe_time = (et - st) * 1000

        best_sample = response.first.sample
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))
        best_bin_vector = np.array(list(best_sample.values()))
        np.save(f"LP_output_dwave\LP_output_OJ_MAP_{count}.npy", best_bin_vector)
        
        # Fill x matrix
        self.rearranged_bin_mat[0] = best_bin_vector[:self.num_binvars]
        # Extract lambda values
        lambda_values = np.array(best_bin_vector[self.num_binvars : self.num_binvars + self.lambda_bits])
        # Calculate lambda_connection
        self.lambda_lower = np.sum([self.lambda_coeff[i] * lambda_values[i] for i in range(self.lambda_bits)])
        self.obj_value = sum(best_bin_vector[i] * self.obj_c[i] for i in range(self.num_binvars)) + self.lambda_lower
                        
        Bifurcation_data = {}
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Bifurcation_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Openjij_info-round-{count}.json")
        create_config_file(dest_path, Bifurcation_data, count)
    
    def cqm_solve_for_value(self, count=0):
        # OBJ # Select a solver
        sampler = LeapHybridCQMSampler(token = self.dwave_token)
        st = time.time()
        sampleset = sampler.sample_cqm(self.cqm, label="Master_problem_HQCBD")
        et = time.time()
        exe_time = (et - st) * 1000
        best = sampleset.filter(lambda row: row.is_feasible).first
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))

        self.obj_value = sampleset.filter(lambda row: row.is_feasible).first.energy        
        
        local_lambda_lower = 0
        for key, val in best.sample.items():
            if key.startswith("bin_"):
                index = int(key.replace("bin_",""))
                self.rearranged_bin_mat[0][index] = val
            elif key.startswith("lambda_bin_"):
                index = int(key.replace("lambda_bin_",""))
                local_lambda_lower += val * self.lambda_coeff[index]
        
        self.lambda_lower = local_lambda_lower

        Dwave_data = {}
        Dwave_data["info"] = sampleset.info
        Dwave_data["wait_id"] = sampleset.wait_id()
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Dwave_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Dwave_info-round-{count}.json")
        create_config_file(dest_path, Dwave_data, count)
    
    def cqm_solve_for_value_mul_cuts(self, count=0):
        # OBJ # Select a solver
        sampler = LeapHybridCQMSampler(token = self.dwave_token)
        st = time.time()
        sampleset = sampler.sample_cqm(self.cqm, label="HQCBD-mul-cuts")
        et = time.time()
        exe_time = (et - st) * 1000
        
        data_length = len(sampleset.filter(lambda row: row.is_feasible).record.energy)
        
        if self.mcut_num > data_length:
            print(f"Feasible answer from DWave is less than MCUT, We will use {data_length} cut in this round")
            
        self.final_mcut_num = min(self.mcut_num , data_length)
        self.obj_value = np.min(sampleset.filter(lambda row: row.is_feasible).record.energy[:self.final_mcut_num])
        best_data = sampleset.filter(lambda row: row.is_feasible).data()
        self.rearranged_bin_mat = np.zeros((self.mcut_num, self.num_binvars))

        Dwave_data = {}
        Dwave_data["info"] = sampleset.info
        Dwave_data["wait_id"] = sampleset.wait_id()
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Dwave_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Dwave_info-round-{count}.json")
        create_config_file(dest_path, Dwave_data, count)
        
        lambda_lower_list = []
        i = 0
        for datum in best_data:
            local_lambda_lower = 0
            for key, val in datum.sample.items():
                if key.startswith("bin_"):
                    index = int(key.replace("bin_",""))
                    self.rearranged_bin_mat[i][index] = val
                elif key.startswith("lambda_bin_"):
                    index = int(key.replace("lambda_bin_",""))
                    local_lambda_lower += val * self.lambda_coeff[index]
            lambda_lower_list.append(local_lambda_lower)
            i += 1
            if i >= self.mcut_num:
                break
        self.lambda_lower = max(lambda_lower_list)

    def solve_gurobi_master_problem(self, count=0): 
        if self.flag_print:
            LP_path = os.path.join(self.LP_folder, f"MAP_gurobi_round_{count}.lp")
            self.MAP.write(LP_path)

        st = time.time()
        # Optimize the model
        self.MAP.optimize()
        et = time.time()
        exe_time = (et - st) * 1000
      
        Gurobi_data = {}
        Gurobi_data["Runtime"] = self.MAP.Runtime
        Gurobi_data["RunVtime"] = self.MAP.RunVtime
        time_data = {
            "time_start": st,
            "time_end": et,
            "time_exec": exe_time
        }
        Gurobi_data["time_data"] = time_data
        dest_path = os.path.join(os.getcwd(), "data_output", f"Gurobi_info-round-{count}.json")
        create_config_file(dest_path, Gurobi_data, count)
        
        x_sol_MAP = self.MAP.getVars()
        self.rearranged_bin = np.zeros(len(self.Bin_varname))
        for index, item in enumerate(x_sol_MAP):
            self.rearranged_bin[index] = item.X
        self.lambda_lower = sum(np.array([self.rearranged_bin[i] for i, item in enumerate(self.Bin_varname) if item.startswith("t_bits")]) * self.lambda_coeff)
        self.obj_value = self.MAP.ObjVal

    def build_gurobi_sub_problem(self, counter = 0):
        
        if self.Hybrid_mode:
            if self.dwave_solver == "bifurcation":
                answer_filename = os.path.join(self.data_folder, f"QBifurcation_MAP_solution-round-{counter}.npy")
                np.save(answer_filename, self.rearranged_bin_mat)
                self.rearranged_x = self.rearranged_bin_mat[0]
                if self.sub_method == "normal":
                    self.subproblem_normal(counter)
                    self.Add_master_constraint_bifurcation(counter)
                elif self.sub_method == "l_shape":
                    self.subproblem_Lshape(counter)
                    self.Add_master_constraint_bifurcation(counter)                   
                else:
                    print("wrong subproblem method, please check config submethod")
            else:
                if self.MC_flag:
                    answer_filename = os.path.join(self.data_folder, f"QMC-{self.final_mcut_num}-MAP_solution-round-{counter}.npy")
                    np.save(answer_filename, self.rearranged_bin_mat)
                    for i in range(self.final_mcut_num):
                        self.rearranged_x = self.rearranged_bin_mat[i]
                        if self.sub_method == "normal":
                            self.subproblem_normal(counter, i)
                            self.Add_master_constraint_Q(counter, i)
                        elif self.sub_method == "l_shape":
                            self.subproblem_Lshape(counter, i)
                            self.Add_master_constraint_Q(counter, i)
                        else:
                            print("wrong subproblem method, please check config submethod")
                else:
                    answer_filename = os.path.join(self.data_folder, f"QMAP_solution-round-{counter}.npy")
                    np.save(answer_filename, self.rearranged_bin_mat)
                    self.rearranged_x = self.rearranged_bin_mat[0]
                    if self.sub_method == "normal":
                        self.subproblem_normal(counter)
                        self.Add_master_constraint_Q(counter)
                    elif self.sub_method == "l_shape":
                        self.subproblem_Lshape(counter)
                        self.Add_master_constraint_Q(counter)                   
                    else:
                        print("wrong subproblem method, please check config submethod")
        else:  
            self.rearranged_x = np.array([self.rearranged_bin[i] for i, item in enumerate(self.Bin_varname) if not item.startswith("t_bits")])
            answer_filename = os.path.join(self.data_folder, f"MAP_solution-round-{counter}.npy")
            np.save(answer_filename, self.rearranged_x)       
            if self.sub_method == "normal":
                self.subproblem_normal(counter)
                self.Add_master_constraint_gurobi(counter)
            elif self.sub_method == "l_shape":
                self.subproblem_Lshape(counter)
                self.Add_master_constraint_gurobi(counter)            
            else:
                print("error, no such method")
                pass
                 
    def subproblem_normal(self, counter = 0, index = 0):
        Ax = self.A_sub @ self.rearranged_x
        sub_rhs_vector = (self.rhs_sub.flatten() - Ax.flatten())
        # create one model instance, with a name
        self.Normal_sub_dual = gp.Model("Sub_normal_dual")
        self.Normal_sub_dual.setParam('LogToConsole', 0)  
        self.Normal_sub_dual.Params.OutputFlag = 0
        self.Normal_sub_dual.setParam("OutputFlag", 0)
        self.Normal_sub_dual.Params.LogToConsole = 0
        self.Normal_sub_dual.Params.InfUnbdInfo = 1        
        u_dual_sub = self.Normal_sub_dual.addMVar( shape= len(sub_rhs_vector), lb = 0, vtype=GRB.CONTINUOUS, name='u_dual_sub')
        # objective function
        self.Normal_sub_dual.setObjective(sub_rhs_vector @ u_dual_sub , GRB.MAXIMIZE)
        # eqConstriants has no upper/lower bound\ 
        for item in self.eq_constraint:
            u_dual_sub[item].lb = -float("inf")

        G_transpose = self.G_sub.T

        # np.dot(G.T, u) < h
        self.Normal_sub_dual.addMConstr(G_transpose, u_dual_sub, "<=", self.obj_d, name=f'Subproblem_constraints')
                    
        self.Normal_sub_dual.optimize()
        if self.flag_print:
            normal_sub_dual_path = os.path.join(self.LP_folder, f"normal_sub_dual_model-round-{counter}-cut-{index}.lp")
            self.Normal_sub_dual.write(normal_sub_dual_path)

        u_sol = self.Normal_sub_dual.getVars()
        result_ray = np.array([item.X for item in u_sol])

        self.MAP_next_lhs = np.dot(result_ray.T, self.A_sub)
        self.MAP_next_rhs = np.dot(self.rhs_sub, result_ray)
        #equality = "<="
        self.lambda_upper = self.Normal_sub_dual.objVal

    def subproblem_Lshape(self, counter = 0, index = 0):
        Ax = self.A_sub @ self.rearranged_x
        sub_rhs_vector = (self.rhs_sub.flatten() - Ax.flatten())  #b-Ax
        # create one model instance, with a name
        self.Lshape_sub = gp.Model("Sub_Lshape")
        self.Lshape_sub.setParam("LogToConsole", 0)
        self.Lshape_sub.setParam("OutputFlag", 0)
        self.Lshape_sub.setParam("InfUnbdInfo", 1)
        
        u_sub = self.Lshape_sub.addMVar(len(self.eq_constraint), lb = 0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='u_sub')
        vr_sub = self.Lshape_sub.addMVar(len(sub_rhs_vector), lb = 0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='vr_sub')
        y_sub = self.Lshape_sub.addMVar(len(self.obj_d), lb = 0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='y_sub')
        
        # Objective function #  sum 1u + 1v + 1r
        self.Lshape_sub.setObjective(sum(u_sub) + sum(vr_sub) , GRB.MINIMIZE)

        for index_ in range(len(sub_rhs_vector)):
            if index_ in self.eq_constraint:
                u_sub_pos = self.eq_constraint.index(index_)
                self.Lshape_sub.addConstr(quicksum(self.G_sub[index_,i] * y_sub[i] for i in range(len(self.obj_d))) + u_sub[u_sub_pos] - vr_sub[index_] == sub_rhs_vector[index_] , name=f'Sub_Ls_problem_eq_constraints_{index_}')
            else:
                self.Lshape_sub.addConstr(quicksum(self.G_sub[index_,i] * y_sub[i] for i in range(len(self.obj_d))) + vr_sub[index_] >= sub_rhs_vector[index_] , name=f'Sub_Ls_problem_ieq_constraints_{index_}')

        self.Lshape_sub.optimize()
        
        if self.flag_print:
            Lshape_sub_path = os.path.join(self.LP_folder, f"sub_Lshape_model-round-{counter}-cut-{index}.lp")
            self.Lshape_sub.write(Lshape_sub_path)
        if self.Lshape_sub.ObjVal > 0:
            self.subproblem_dual_Lshape(counter, index)
            self.L_shape_flag = 1
        else:            
            self.subproblem_normal(counter, index)
            self.L_shape_flag = 0
            
    def subproblem_dual_Lshape(self, counter = 0, cut_index = 0):
        Ax = self.A_sub @ self.rearranged_x
        sub_rhs_vector = (self.rhs_sub.flatten() - Ax.flatten())
        b = self.rhs_sub.flatten()
        e = np.ones_like(b)

        # create one model instance, with a name
        self.Lshape_sub_dual = gp.Model("Sub_Lshape_dual")
        self.Lshape_sub_dual.setParam("LogToConsole", 0)
        self.Lshape_sub_dual.setParam("OutputFlag", 0)
        self.Lshape_sub_dual.setParam("InfUnbdInfo", 1)
        
        w_sub = self.Lshape_sub_dual.addMVar(len(sub_rhs_vector), lb = 0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='w_sub')
        #Objective function
        self.Lshape_sub_dual.setObjective(sub_rhs_vector @ w_sub , GRB.MAXIMIZE)
        G_transpose = self.G_sub.T
        self.Lshape_sub_dual.addConstrs((G_transpose[i,:] @ w_sub <= 0 for i in range(len(G_transpose)) ), name=f'Sub_L_problem_constraints')

        for index in range(len(sub_rhs_vector)):
            #Lshape constraints "<=" or "=="
            self.Lshape_sub_dual.addConstr(e[index] * w_sub[index] <= e[index],  name=f'Subproblem_eq_leq_constraints_{index}')
            if index in self.eq_constraint:
                #Lshape constraints "="
                w_sub[index].lb = -float("inf")
                self.Lshape_sub_dual.addConstr(-e[index] * w_sub[index] <= e[index], name=f'Subproblem_leq_constraints_{index}')
        self.Lshape_sub_dual.optimize()
        if self.flag_print:
            Lshape_sub_dual_path = os.path.join(self.LP_folder, f"sub_Lshape_dual_model-round-{counter}-cut-{cut_index}.lp")
            self.Lshape_sub_dual.write(Lshape_sub_dual_path)
        Sub_L_sol = self.Lshape_sub_dual.getVars()
        result_ray = np.array([item.X for item in Sub_L_sol])
        self.MAP_next_lhs = np.dot(result_ray.T, self.A_sub)   #sigma * A
        self.MAP_next_rhs = np.dot(self.rhs_sub, result_ray)   #sigma * b
        #equality = ">="
        #self.lambda_upper = self.Lshape_sub_dual.objVal

    def Add_master_constraint_gurobi(self, counter = 0, index = 0):
            equality = ">="
            if self.sub_method == "normal":
                statuscode = self.Normal_sub_dual.getAttr(GRB.Attr.Status)
            elif self.sub_method == "l_shape":
                if self.L_shape_flag:
                    statuscode = 5 #self.Lshape_sub_dual.getAttr(GRB.Attr.Status)
                else:
                    statuscode = 2
            else:
                print("error, no such method")
            
            if statuscode == 5 :   # unbounded
                print("create feasibility cut")
                self.MAP.addConstr(self.MAP_next_lhs @ self.bin_x >= self.MAP_next_rhs, name=f'constraint_add_round_{counter}_cut_{index}')
                self.master_constraint_dict.update({f"c-round-{counter}-cut-{index}":  [[self.MAP_next_lhs[m] for m in range(len(self.MAP_next_lhs))] , equality , self.MAP_next_rhs] })
                
            elif statuscode == 2: # Optimal
                print("create optimality cut")
                self.MAP.addConstr(self.MAP_next_lhs @ self.bin_x + self.lambda_coeff @ self.bin_lambda >= self.MAP_next_rhs, \
                    name=f'constraint_add_round_{counter}_cut_s{index}')
                self.master_constraint_dict.update({f"c-round-{counter}-cut-{index}":  [[self.bin_x[m] * self.MAP_next_lhs[m] for m in range(len(self.MAP_next_lhs))] \
                    + [self.bin_lambda[n] * self.lambda_coeff[n] for n in range(self.lambda_bits)] , equality, self.MAP_next_rhs] })

    def Add_master_constraint_Q(self, counter = 0, index = 0):
        equality = ">="
        if self.sub_method == "normal":
            statuscode = self.Normal_sub_dual.getAttr(GRB.Attr.Status)
        elif self.sub_method == "l_shape":
            if self.L_shape_flag:
                statuscode = 5 #self.Lshape_sub_dual.getAttr(GRB.Attr.Status)
            else:
                statuscode = 2
        else:
            print("error, no such method") 
        if statuscode == 5 :   # unbounded
            print("create feasibility cut")
            QUBO_next_lhs = np.concatenate([self.MAP_next_lhs, np.zeros(self.lambda_bits), np.zeros(len(self.qubo_slack_dict.keys()))])
            QUBO_next_rhs = self.MAP_next_rhs
            self.bifurcation_QUBO, self.qubo_slack_dict = QUBO_Cons_selection(self.bifurcation_QUBO, QUBO_next_lhs, equality, QUBO_next_rhs, round_num = counter, penalty = 10, dictionary = self.qubo_slack_dict)
            self.master_constraint_dict.update({f"c-round-{counter}-cut-{index}":  [[self.MAP_next_lhs[m] for m in range(len(self.MAP_next_lhs))] , equality , self.MAP_next_rhs] })
        elif statuscode == 2: # Optimal
            print("create optimality cut")
            QUBO_next_lhs = np.concatenate([self.MAP_next_lhs, self.lambda_coeff, np.zeros(len(self.qubo_slack_dict.keys()))])
            QUBO_next_rhs = self.MAP_next_rhs
            self.bifurcation_QUBO, self.qubo_slack_dict = QUBO_Cons_selection(self.bifurcation_QUBO, QUBO_next_lhs, equality, QUBO_next_rhs, round_num = counter, penalty = 10, dictionary = self.qubo_slack_dict)       
            self.master_constraint_dict.update({f"c-round-{counter}-cut-{index}":  [[self.MAP_next_lhs[m] for m in range(len(self.MAP_next_lhs))], [self.lambda_coeff], equality, self.MAP_next_rhs] })
                         
    def Add_master_constraint_bifurcation(self, counter = 0):
        equality = ">="
        if self.sub_method == "normal":
            statuscode = self.Normal_sub_dual.getAttr(GRB.Attr.Status)
        elif self.sub_method == "l_shape":
            if self.L_shape_flag:
                statuscode = 5 #self.Lshape_sub_dual.getAttr(GRB.Attr.Status)
            else:
                statuscode = 2
        else:
            print("error, no such method")
        if statuscode == 5 :   # unbounded
            print("create feasibility cut")
            QUBO_next_lhs = np.concatenate([self.MAP_next_lhs, np.zeros(self.lambda_bits), np.zeros(len(self.qubo_slack_dict.keys()))])
            QUBO_next_rhs = self.MAP_next_rhs
            self.bifurcation_QUBO, self.qubo_slack_dict = QUBO_Cons_selection(self.bifurcation_QUBO, QUBO_next_lhs, equality, QUBO_next_rhs, round_num = counter, penalty = 10, dictionary = self.qubo_slack_dict)
            self.master_constraint_dict.update({f"c-round-{counter}": [[self.MAP_next_lhs[m] for m in range(len(self.MAP_next_lhs))], equality , self.MAP_next_rhs] })        
        elif statuscode == 2: # Optimal
            print("create optimality cut")
            QUBO_next_lhs = np.concatenate([self.MAP_next_lhs, self.lambda_coeff, np.zeros(len(self.qubo_slack_dict.keys()))])
            QUBO_next_rhs = self.MAP_next_rhs
            self.bifurcation_QUBO, self.qubo_slack_dict = QUBO_Cons_selection(self.bifurcation_QUBO, QUBO_next_lhs, equality, QUBO_next_rhs, round_num = counter, penalty = 10, dictionary = self.qubo_slack_dict)
            self.master_constraint_dict.update({f"c-round-{counter}":  [[self.MAP_next_lhs[m] for m in range(len(self.MAP_next_lhs))] \
                + [self.lambda_coeff[n] for n in range(self.lambda_bits)] , equality, self.MAP_next_rhs] })
                
    def run(self):
        lambda_upper_list = []
        lambda_lower_list = []
        obj_value_list = []
        ralative_gap = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3)
        abs_gap = abs(self.lambda_upper - self.lambda_lower)
        cunt = 1
        
        if self.threshold_type == "relative":
            gap = ralative_gap
        elif self.threshold_type == "absolute":
            gap = abs_gap
        
        if self.Hybrid_mode:       
            if self.dwave_solver in ["bifurcation", "openjij"]:
                self.build_QUBO_master_problem()
                while gap >= self.threshold_gap:
                    print(f"Benders decomposition Round - {cunt}")
                    self.solve_master_problem(cunt)
                    self.build_gurobi_sub_problem(cunt)
                    ratio = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3)
                    if self.Msense:
                        print(f"Round {cunt}: \n \
                            Current Objective value is {-1* self.obj_value}; \n \
                            lambda_upper is {-1*self.lambda_lower}; \n \
                            lambda_lower is {-1*self.lambda_upper}; \n \
                            Relative gap is {ratio}%. \n \
                            Absolute gap is {abs(self.lambda_upper - self.lambda_lower)}")
                        obj_value_list.append(-1* self.obj_value)
                        lambda_upper_list.append(-1*self.lambda_lower)
                        lambda_lower_list.append(-1*self.lambda_upper)                
                    else:
                        print(f"Round {cunt}: \n \
                            Current Objective value is {self.obj_value}; \n \
                            lambda_upper is {self.lambda_upper}; \n \
                            lambda_lower is {self.lambda_lower}; \n \
                            Relative gap is {ratio}%. \n \
                            Absolute gap is {abs(self.lambda_upper - self.lambda_lower)}")
                        obj_value_list.append(self.obj_value)
                        lambda_upper_list.append(self.lambda_upper)
                        lambda_lower_list.append(self.lambda_lower)
                    
                    ralative_gap = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3)
                    abs_gap = abs(self.lambda_upper - self.lambda_lower)
                    if self.threshold_type == "relative":
                        gap = ralative_gap
                    elif self.threshold_type == "absolute":
                        gap = abs_gap
                    if gap < self.threshold_gap:
                        print("optimal found, it takes", cunt, " Rounds.")
                        break                    
                    cunt += 1
                    if  cunt > self.max_steps:
                        print("Max iteration reached, it takes", cunt, " Rounds.")
                        break
            else:
                self.build_cqm_master_problem()
                while gap >= self.threshold_gap:
                    print(f"Benders decomposition Round - {cunt}")
                    self.solve_master_problem(cunt)
                    self.build_gurobi_sub_problem(cunt)
                    ratio = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3) 
                    if self.Msense:
                        print(f"Round {cunt}: \n \
                            Current Objective value is {-1* self.obj_value}; \n \
                            lambda_upper is {-1*self.lambda_lower}; \n \
                            lambda_lower is {-1*self.lambda_upper}; \n \
                            Relative gap is {ratio}%. \n \
                            Absolute gap is {abs(self.lambda_upper - self.lambda_lower)}")
                        obj_value_list.append(-1* self.obj_value)
                        lambda_upper_list.append(-1*self.lambda_lower)
                        lambda_lower_list.append(-1*self.lambda_upper)                
                    else:
                        print(f"Round {cunt}: \n \
                            Current Objective value is {self.obj_value}; \n \
                            lambda_upper is {self.lambda_upper}; \n \
                            lambda_lower is {self.lambda_lower}; \n \
                            Relative gap is {ratio}%. \n \
                            Absolute gap is {abs(self.lambda_upper - self.lambda_lower)}")
                        obj_value_list.append(self.obj_value)
                        lambda_upper_list.append(self.lambda_upper)
                        lambda_lower_list.append(self.lambda_lower)
                
                    ralative_gap = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3)
                    abs_gap = abs(self.lambda_upper - self.lambda_lower)
                    
                    if self.threshold_type == "relative":
                        gap = ralative_gap
                    elif self.threshold_type == "absolute":
                        gap = abs_gap
                    
                    if gap < self.threshold_gap:
                        print("optimal found, it takes", cunt, " Rounds.")
                        break
                    cunt += 1
                    if  cunt > self.max_steps:
                        print("Max iteration reached, it takes", cunt, " Rounds.")
                        break
            
        else:
            self.build_gurobi_master_problem()
            while gap >= self.threshold_gap:
                
                print(f"Benders decomposition Round - {cunt}")
                
                self.solve_gurobi_master_problem(cunt)
                self.build_gurobi_sub_problem(cunt)
                
                ratio = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3)
                
                if self.Msense:
                    print(f"Round {cunt}: \n \
                        Current Objective value is {-1* self.obj_value}; \n \
                        lambda_upper is {-1*self.lambda_lower}; \n \
                        lambda_lower is {-1*self.lambda_upper}; \n \
                        Relative gap is {ratio}%. \n \
                        Absolute gap is {abs(self.lambda_upper - self.lambda_lower)}")
                    obj_value_list.append(-1* self.obj_value)
                    lambda_upper_list.append(-1*self.lambda_lower)
                    lambda_lower_list.append(-1*self.lambda_upper)                
                else:
                    print(f"Round {cunt}: \n \
                        Current Objective value is {self.obj_value}; \n \
                        lambda_upper is {self.lambda_upper}; \n \
                        lambda_lower is {self.lambda_lower}; \n \
                        Relative gap is {ratio}%. \n \
                        Absolute gap is {abs(self.lambda_upper - self.lambda_lower)}")
                    obj_value_list.append(self.obj_value)
                    lambda_upper_list.append(self.lambda_upper)
                    lambda_lower_list.append(self.lambda_lower)
                
                ralative_gap = np.round(abs(self.lambda_upper - self.lambda_lower) / abs(self.lambda_upper) *100 , 3)
                abs_gap = abs(self.lambda_upper - self.lambda_lower)
                
                if self.threshold_type == "relative":
                    gap = ralative_gap
                elif self.threshold_type == "absolute":
                    gap = abs_gap
                if gap < self.threshold_gap:
                    print("optimal found, it takes", cunt, f" Rounds. The Binary results are stored at {self.data_folder}")
                    break
                cunt += 1
                if  cunt > self.max_steps:
                    print("Max iteration reached, it takes", cunt, " Rounds.")
                    break

        lambda_upper_filepath = os.path.join(self.data_folder, "lambda_upper_list.json")
        lambda_lower_filepath = os.path.join(self.data_folder, "lambda_lower_list.json")
        obj_value_filepath = os.path.join(self.data_folder, "obj_value_list.json")
            
        with open(lambda_upper_filepath, 'w') as json_file:
            json.dump(lambda_upper_list, json_file)    
        with open(lambda_lower_filepath, 'w') as json_file:
            json.dump(lambda_lower_list, json_file)
        with open(obj_value_filepath, 'w') as json_file:
            json.dump(obj_value_list, json_file)

        print(f"lambda_upper list has been saved to {lambda_upper_filepath}")
        print(f"lambda_lower list has been saved to {lambda_lower_filepath}")
        print(f"obj_value list has been saved to {obj_value_filepath}")
        