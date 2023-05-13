# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:51:22 2023

@author: Ravindranath Nemani
"""

import numpy as np

from random import sample

def get_basis_states():
    psi_00 = [1,0]
    psi_10 = [0, 1]
    psi_01 = [1/np.sqrt(2), 1/np.sqrt(2)]
    psi_11 = [1/np.sqrt(2), -1/np.sqrt(2)]
    return psi_00, psi_10, psi_01, psi_11


def get_Alice_basis_states(Alice_string, Alice_basis):
    psi_Alice = []
    psi_00, psi_10, psi_01, psi_11 = get_basis_states()
    for i in range(len(Alice_basis)):
        if Alice_string[i] == 0 and Alice_basis[i] == 0:
            psi_Alice.append(psi_00)
        elif Alice_string[i] == 1 and Alice_basis[i] == 0:
            psi_Alice.append(psi_10)
        elif Alice_string[i] == 0 and Alice_basis[i] == 1:
            psi_Alice.append(psi_01)
        else:
            psi_Alice.append(psi_11)
    return psi_Alice
            

def get_Eve_basis(n):
    basis_Eve = np.random.randint(2, size=4*n)
    return basis_Eve


def get_Eve_basis_states(Alice_string, n):
    psi_00, psi_10, psi_01, psi_11 = get_basis_states()
    basis_Eve = get_Eve_basis(n)
    psi_Eve = []
    for i in range(len(Alice_string)):
        if Alice_string[i] == 0 and basis_Eve[i] == 0:
            psi_Eve.append(psi_00)
        elif Alice_string[i] == 1 and basis_Eve[i] == 0:
            psi_Eve.append(psi_10)
        elif Alice_string[i] == 0 and basis_Eve[i] == 1:
            psi_Eve.append(psi_01)
        else:
            psi_Eve.append(psi_11)
    return psi_Eve


def measure_a_state_computational_basis(psi):
    psi_00, psi_10, psi_01, psi_11 = get_basis_states()    
    
    zero = np.array([[psi_00[0]], [psi_00[1]]])
    one = np.array([[psi_10[0]], [psi_10[1]]])
    
    M0 = np.dot(zero, zero.T)
    M1 = np.dot(one, one.T)
    
    psi_after_measurement_state1_numerator = np.dot(M0, psi)
    M0_1 = np.dot(M0.T, M0)
    psi_after_measurement_state1_denominator = np.sqrt(np.dot(psi.T, np.dot(M0_1, psi)))
    probability_of_state1 = np.dot(psi.T, np.dot(M0_1, psi))
    psi_after_measurement_state1_numerator = list(psi_after_measurement_state1_numerator)
    if psi_after_measurement_state1_denominator != 0:
        psi_after_measurement_state1 = [x*(1/psi_after_measurement_state1_denominator) for x in psi_after_measurement_state1_numerator]
    else:
        psi_after_measurement_state1 = psi_after_measurement_state1_numerator
    
    psi_after_measurement_state2_numerator = np.dot(M1, psi)
    M1_1 = np.dot(M1.T, M1)
    psi_after_measurement_state2_denominator = np.sqrt(np.dot(psi.T, np.dot(M1_1, psi)))
    probability_of_state2 = np.dot(psi.T, np.dot(M1_1, psi))
    psi_after_measurement_state2_numerator = list(psi_after_measurement_state2_numerator)
    if psi_after_measurement_state2_denominator != 0:
        psi_after_measurement_state2 = [x*(1/psi_after_measurement_state2_denominator) for x in psi_after_measurement_state2_numerator]
    else:
        psi_after_measurement_state2 = psi_after_measurement_state2_numerator
    '''
    if (np.random.rand() < probability_of_state1):
        measurement_outcome = 0
        psi_after_measurement = psi_after_measurement_state1
    else:
        measurement_outcome = 1
        psi_after_measurement = psi_after_measurement_state2
    '''
    outcomes = [0, 1]
    probabilities = [probability_of_state1,1-probability_of_state1]
    measurement_outcome = np.random.choice(outcomes, 1, p=probabilities)
    if measurement_outcome == 0:
        psi_after_measurement = psi_after_measurement_state1
    else:
        psi_after_measurement = psi_after_measurement_state2
    return [measurement_outcome, psi_after_measurement]


def measure_a_state_hadamard_basis(psi):
    psi_00, psi_10, psi_01, psi_11 = get_basis_states()    
        
    H0 = np.array([[psi_01[0]], [psi_01[1]]])
    H1 = np.array([[psi_11[0]], [psi_11[1]]])
    
    M0 = np.dot(H0, H0.T)
    M1 = np.dot(H1, H1.T)
    
    psi_after_measurement_state1_numerator = np.dot(M0, psi)
    M0_1 = np.dot(M0.T, M0)
    psi_after_measurement_state1_denominator = np.sqrt(np.dot(psi.T, np.dot(M0_1, psi)))
    probability_of_state1 = np.dot(psi.T, np.dot(M0_1, psi))
    psi_after_measurement_state1_numerator = list(psi_after_measurement_state1_numerator)
    
    if psi_after_measurement_state1_denominator != 0:
        psi_after_measurement_state1 = [x*(1/psi_after_measurement_state1_denominator) for x in psi_after_measurement_state1_numerator]
    else:
        psi_after_measurement_state1 = psi_after_measurement_state1_numerator
    
    psi_after_measurement_state2_numerator = np.dot(M1, psi)
    M1_1 = np.dot(M1.T, M1)
    psi_after_measurement_state2_denominator = np.sqrt(np.dot(psi.T, np.dot(M1_1, psi)))
    probability_of_state2 = np.dot(psi.T, np.dot(M1_1, psi))
    psi_after_measurement_state2_numerator = list(psi_after_measurement_state2_numerator)
    if psi_after_measurement_state2_denominator != 0: 
        psi_after_measurement_state2 = [x*(1/psi_after_measurement_state2_denominator) for x in psi_after_measurement_state2_numerator]
    else:
        psi_after_measurement_state2 = psi_after_measurement_state2_numerator
    '''
    if (np.random.rand() < probability_of_state1):
        measurement_outcome = 0
        psi_after_measurement = psi_after_measurement_state1
    else:
        measurement_outcome = 1
        psi_after_measurement = psi_after_measurement_state2
    '''
    outcomes = [0, 1]
    probabilities = [probability_of_state1, 1-probability_of_state1]
    measurement_outcome = np.random.choice(outcomes, 1, p=probabilities)
    if measurement_outcome == 0:
        psi_after_measurement = psi_after_measurement_state1
    else:
        psi_after_measurement = psi_after_measurement_state2

    return [measurement_outcome, psi_after_measurement]


def get_Bob_basis(n):
    basis_Bob = np.random.randint(2, size=4*n)
    return basis_Bob


def get_Bob_basis_states(Alice_string):
    psi_00, psi_10, psi_01, psi_11 = get_basis_states()
    basis_Bob = get_Bob_basis(n)
    psi_Bob = []
    for i in range(len(Alice_string)):
        if Alice_string[i] == 0 and basis_Bob[i] == 0:
            psi_Bob.append(psi_00)
        elif Alice_string[i] == 1 and basis_Bob[i] == 0:
            psi_Bob.append(psi_10)
        elif Alice_string[i] == 0 and basis_Bob[i] == 1:
            psi_Bob.append(psi_01)
        else:
            psi_Bob.append(psi_11)
    return psi_Bob


def Eve_measures_and_resends_to_Bob(psi_Eve):
    lst = range(4*n)
    random_subset_computational = sample(lst, 2*n)
    random_subset_hadamard = [elem for elem in lst if elem not in random_subset_computational]
    psi_Eve_measured = []
    for i in range(len(psi_Eve)):
        psi_Eve = np.array(psi_Eve)
        if i in random_subset_computational:
            psi_Eve_measured.append(measure_a_state_computational_basis(psi_Eve[i])[1])
        else:
            psi_Eve_measured.append(measure_a_state_hadamard_basis(psi_Eve[i])[1])
    return psi_Eve_measured


def Bob_recieves_qubits_from_Eve_and_measures(basis_Bob, psi_Eve_measured):
    Bob_measurement_outcomes = []
    for i in range(len(basis_Bob)):
        psi_Eve_measured = np.array(psi_Eve_measured)
        if basis_Bob[i] == 0:
            Bob_measurement_outcomes.append(measure_a_state_computational_basis(psi_Eve_measured[i])[0])
        else:
            Bob_measurement_outcomes.append(measure_a_state_hadamard_basis(psi_Eve_measured[i])[0])
    return Bob_measurement_outcomes


def discarding_bits_after_Alice_announces_her_basis(Alice_string, Alice_basis, basis_Bob, Bob_measurement_outcomes):
    indices_of_the_bits_not_discarded = []
    Alice_bits_after_discarding = []
    Bob_bits_after_discarding = []
    for i in range(len(Alice_basis)):
        if Alice_basis[i] == basis_Bob[i]:
            indices_of_the_bits_not_discarded.append(i)
            Alice_bits_after_discarding.append(Alice_string[i])
            Bob_bits_after_discarding.append(Bob_measurement_outcomes[i])
    return indices_of_the_bits_not_discarded, Alice_bits_after_discarding, Bob_bits_after_discarding


def error_estimation_and_decision(n, Alice_bits_after_discarding, Bob_bits_after_discarding):
    bit_error_count = 0
    l = np.random.choice(2*n, size=n, replace=False)
    for i in range(len(l)):
        if Alice_bits_after_discarding[i] != Bob_bits_after_discarding[i]:
            bit_error_count = bit_error_count + 1
    error_percent = bit_error_count/len(l)
    if error_percent > 0.25:
        decision = "Eve has intercepted the communication. So ABORT the protocol."
    else:
        decision = "Eve has NOT intercepted the communication. So DO NOT ABORT the protocol."
    return decision


#main

n = 5

Alice_string = np.random.randint(2, size=4*n)

Alice_string = list(Alice_string)

Alice_basis = np.random.randint(2, size=4*n)

Alice_basis = list(Alice_basis)

get_basis_states()

psi_Alice = get_Alice_basis_states(Alice_string, Alice_basis)

psi_Eve = get_Eve_basis_states(Alice_string, n)

psi_Eve_measured = Eve_measures_and_resends_to_Bob(psi_Eve)

basis_Bob = get_Bob_basis(n)

Bob_measurement_outcomes = Bob_recieves_qubits_from_Eve_and_measures(basis_Bob, psi_Eve_measured)

indices_of_the_bits_not_discarded, Alice_bits_after_discarding, Bob_bits_after_discarding = discarding_bits_after_Alice_announces_her_basis(Alice_string, Alice_basis, basis_Bob, Bob_measurement_outcomes)

decision = error_estimation_and_decision(n, Alice_bits_after_discarding, Bob_bits_after_discarding)

print("Decision is : ", decision)

input("Done!")