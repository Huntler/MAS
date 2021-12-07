import random
import itertools
import numpy as np
from typing import Set, Dict, List
from main import *


def get_voting_situation(number_voters, number_candidates):
    candidates = number_candidates
    voters = number_voters
    votings = []
    votings_template_list = []
    for i in range(0, voters):
        votings_template_list.append([])
        for j in range(0, candidates):
            votings_template_list[i].append(j + 1)
    for i in range(0, voters):
        votings.append([])
        for j in range(0, candidates):
            temp_len = len(votings_template_list[i])
            random_index = random.randint(0, temp_len - 1)
            votings[i].append(votings_template_list[i].pop(random_index))
    return votings


def get_winners(votings):
    dict = {}
    for i in range(len(votings[0])):
        dict[i + 1] = 0
    for voting in votings:
        rev_voting = list(reversed(voting))
        for i in range(len(voting)):
            dict[rev_voting[i]] += i + 1
    sorted_winners = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    sorted_winners_list = []
    for i in sorted_winners:
        sorted_winners_list.append(i[0])
    return sorted_winners_list


def happyiness(i: np.array, j: np.array, s: float = 0.9) -> float:
    m = len(i)
    s1 = np.power([s for l in range(m)], np.power(range(m), 2))
    s2 = np.power([s for l in range(m)], np.power(m - np.array(range(m)), 2))
    d = np.abs(j - i)
    h_hat = np.sum(d * (s1 + s2)) / m
    return np.exp(-h_hat)


def single_voter_manipulation(votings):
    preferred_preferences = []
    all_permutations = list(itertools.permutations(votings[0]))
    for voting in range(len(votings)):
        origin_preference = votings[voting]
        origin_winners = get_winners(votings)
        origin_happyness = happyiness(np.array(origin_winners), np.array(voting))
        for permutation in range(1, len(all_permutations)):
            votings[voting] = all_permutations[permutation]
            winners = get_winners(votings)
            happyness_value = happyiness(np.array(winners), np.array(origin_preference))
            if happyness_value > origin_happyness:
                new_ordering = [voting, all_permutations[permutation], happyness_value]
                preferred_preferences.append(new_ordering)
        votings[voting] = origin_preference
    return preferred_preferences


def get_similar_voters():
    raise NotImplementedError("")


def get_voter_compatibility(voter_a_preferences, voter_b_preferences):
    differences = 0
    for i in range(len(voter_a_preferences)):
        if voter_a_preferences[i] != voter_b_preferences[i]:
            difference = i
            for j in range(len(voter_a_preferences)):
                if voter_a_preferences[i] != voter_b_preferences[j]:
                    break
                if j < i:
                    difference -= 1
                else:
                    difference += 1
            differences += difference
    worstcase = len(voter_a_preferences) * (len(voter_a_preferences) + 1) / 2
    return differences / worstcase * 100


def get_multiple_voter_compatibility(preferences):
    combinations = set()
    for voter_1 in preferences:
        for voter_2 in preferences:
            if not voter_1 == voter_2:
                combinations.add({voter_1, voter_2})
    ret = 0
    for voter in combinations:
        voter = list(voter)
        ret += get_voter_compatibility(voter[0], voter[1])
    ret /= len(combinations)


def lengths_sufficient(teams) -> bool:
    for team in teams:
        if len(team) == 5:
            return True
    return False


def multiple_voter_manipulations(origin_votings, coalition, all_permutations):
    multiple_voter_manipulations = []
    coalition_votings = []
    coalition_happiness_i = []
    winners = get_winners(origin_votings)
    for permutation in range(len(all_permutations)):
        temp_votings = origin_votings
        for member in coalition:
            coalition_votings.append(temp_votings[member])
            coalition_happiness_i.append(happiness(np.array(winners), np.array(temp_votings[member])))
            temp_votings[member] = all_permutations[permutation]
        temp_winners = get_winners(temp_votings)
        all_happier = 1
        for member in range(len(coalition)):
            temp_happiness = happiness(np.array(temp_winners), np.array(coalition_votings[member]))
            if temp_happiness <= coalition_happiness_i[member]:
                all_happier = 0
                break
        if all_happier == 1:
            multiple_voter_manipulations.append(permutation)
    return multiple_voter_manipulations


def create_groups_onlybiggest(votings, threshold: int):
    all_permutations = list(multiset_permutations(votings[0]))
    # collect all possible likely voter groups:
    teams = list()
    for i, voting in enumerate(votings):
        for j, friend in enumerate(votings):
            if voting != friend:
                teams.append([i, j])
    while not lengths_sufficient(teams):
        new_teams = list()
        for team in teams:
            for j, friend in enumerate(votings):
                if j not in team:
                    team_idxs = [i for i in team]
                    new_teams.append(team_idxs + [j])
        teams = new_teams
    new_teams = list()
    for team in teams:
        new_teams.append([team, multiple_voter_manipulations(votings, [i, j], all_permutations)])
    return new_teams


def create_groups(votings, threshold: int):
    all_permutations = list(multiset_permutations(votings[0]))
    # collect all possible likely voter groups:
    voting_groups = list()
    teams = list()
    for i, voting in enumerate(votings):
        for j, friend in enumerate(votings):
            if voting != friend:
                friendship_value = get_voter_compatibility(voting, friend)
                if friendship_value <= threshold:
                    teams.append([[i, j], multiple_voter_manipulations(votings, [i, j], all_permutations)])
    voting_groups += teams
    while not lengths_sufficient(teams):
        new_teams = list()
        for team in teams:
            for j, friend in enumerate(votings):
                if j not in team[0]:
                    # check if they should cooperate
                    # to reduce the complexity, the new contributor is only compared to one of the already existing team
                    if get_voter_compatibility(votings[team[0][0]], friend) <= threshold:
                        team_idxs = [i for i in team[0]]
                        new_teams.append(
                            [team_idxs + [j], multiple_voter_manipulations(votings, team_idxs + [j], all_permutations)])
        voting_groups += new_teams
        teams = new_teams
    return voting_groups


def get_manipulation_probability(all_happinesses_manipulations):
    probs = []
    total = sum(all_happinesses_manipulations, key=lambda x: x[1])
    for manipulation, happiness in all_happinesses_manipulations:
        probs.append((manipulation, happiness / total))
    return probs


def get_coalition_probablity(origin_votings, coalitions):
    probs = []
    for coalition in coalitions:
        # Dear Wiebke, please: return coalitions = [ coalition1, coalition2, coalition3 ] where coalition [ [ members ], [manipulations] ]
        all_happinesses_manipulations = []
        for manipulation in coalition[1]:
            temp_votings = origin_votings
            for member_index in coalition[0]:
                temp_votings[member_index] = manipulation
            temp_winners = get_winners(temp_votings)
            temp_overall_happiness = overall_happiness(coalition[0], temp_winners)
            manipulation_obj = (manipulation, temp_overall_happiness)
            all_happinesses_manipulations.append(manipulation_obj)
        probs_coalition = get_manipulation_probability(all_happinesses_manipulations)
        probs.append((coalition, probs_coalition))
    return probs


def counter_voting(origin_votings, coalitions, i, j):
    colation_probs = get_coalition_probablity(origin_votings, coalitions)


def create_groups_final(votings, threshold):
    copiedvotings = votings.copy()
    new_groups = []
    indices = []
    while len(votings) != 0:
        start = votings.pop(0)
        indexgroup = []
        indexgroup.append(copiedvotings.index(start))
        group = []
        group.append(start)
        new_votings = []
        for i, voting in enumerate(votings):
            if get_voter_compatibility(voting, start) <= threshold:
                # del voting, votings
                group.append(voting)
                indexgroup.append(copiedvotings.index(voting))
            else:
                new_votings.append(voting)
        votings = new_votings
        new_groups.append(group)
        indices.append(indexgroup)
    votings = copiedvotings
    return new_groups, indices


if __name__ == '__main__':
    voters = 10
    candidates = 5
    votings = get_voting_situation(voters, candidates)
    possible_manipulations = single_voter_manipulation(votings)

    groups, indices = create_groups_final(votings, 20)

    tactical_votings = []
    for i,group in enumerate(groups):
        tactical_votings.append([indices[i], multiple_voter_manipulations(votings, indices[i], list(multiset_permutations(votings[0])))])
        #tactical_votings.append(multiple_voter_manipulations(votings, group, possible_manipulations)

    # tactical_votings -> [ coalition1, coalition2, coalition3 ] where coalition [ [ members ], [manipulations] ]

    # groups = create_groups_onlybiggest(votings, 20)

    # similar voters -> list of list (different voting preferences among similar voters -> compare happyness to origin preference)
    #                -> for each group manipulation needs to be better than for each voters origin preference

    #
