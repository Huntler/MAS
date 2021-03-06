import random
import itertools
import numpy as np
from sympy.utilities.iterables import multiset_permutations



def get_voting_situation(number_voters, number_candidates):
    candidates = number_candidates
    voters = number_voters
    votings = []
    votings_template_list = []
    for i in range(0, voters):
        votings_template_list.append([])
        for j in range(0, candidates):
            votings_template_list[i].append(j+1)
    for i in range(0, voters):
        votings.append([])
        for j in range(0, candidates):
            temp_len = len(votings_template_list[i])
            random_index = random.randint(0,temp_len-1)
            votings[i].append(votings_template_list[i].pop(random_index))
    return votings

def get_winners(votings):
    dict = {}
    for i in range(len(votings[0])):
        dict[i+1] = 0
    for voting in votings:
        rev_voting = list(reversed(voting))
        for i in range(len(voting)):
            dict[rev_voting[i]] += i+1
    sorted_winners = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    sorted_winners_list = []
    for i in sorted_winners:
        sorted_winners_list.append(i[0])
    return sorted_winners_list

def happiness(i: np.array, j: np.array, s: float=0.9) -> float:
    m = len(i)
    s1 = np.power([s for l in range(m)], np.power(range(m), 2))
    s2 = np.power([s for l in range(m)], np.power(m - np.array(range(m)), 2))
    d = np.abs(j - i)
    h_hat = np.sum(d * (s1 + s2)) / m
    return np.exp(-h_hat)

def single_voter_manipulation(votings):
    preferred_preferences = []
    for voting in range(len(votings)):
        origin_preference = votings[voting]
        origin_winners = get_winners(votings)
        origin_happyness = happiness(np.array(origin_winners), np.array(voting))
        all_permutations = list(multiset_permutations(votings[voting]))
        for permutation in range(1, len(all_permutations)):
            votings[voting] = all_permutations[permutation]
            winners = get_winners(votings)
            happyness_value = happiness(np.array(winners), np.array(origin_preference))
            if happyness_value > origin_happyness:
                new_ordering = [voting, all_permutations[permutation], happyness_value]
                preferred_preferences.append(new_ordering)
        votings[voting] = origin_preference
    return preferred_preferences

if __name__ == '__main__':
    voters = 100
    candidates = 5
    votings = get_voting_situation(voters, candidates)
    possible_manipulations = single_voter_manipulation(votings)
    print(possible_manipulations)

    # problem: wenn preference_permutation gar nichts mehr mit den urspr??nglichen preferencen zu tun hat
    #          => preferences permutation f??r alle voter gleich, da sie vom ganzen ergebnis abh??ngt
    #          => und immer die beste permutation genommen wird (die abh??ngig von dem gesamtergebnis f??r all gleich ist)
    #          => L??sung: preference nur leicht ah??ndern (1. bleibt gleich, 2. bleibt gleich ...)

    # similar voters -> list of list (different among similar voters -> compare happyness to origin preference)
    #                -> same manipulation (for each happier)
    #
    #










