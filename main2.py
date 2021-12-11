from abc import ABC, abstractmethod
import math
import tqdm
import numpy as np
import sys, getopt, os
from typing import Set, Dict, List
from sympy.utilities.iterables import multiset_permutations
import matplotlib.pyplot as plt


class VotingScheme(ABC):
    def __init__(self, mapping=None):
        self._mapping = np.asarray(mapping)

    @abstractmethod
    def compute_res(self, preferences) -> list:
        raise NotImplemented


class VotingForOne(VotingScheme):
    def compute_res(self, preferences: np.array):
        # get the voted candidates and how often they were voted
        (unique, counts) = np.unique(preferences[:, 0], return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        max_vote = max(counts)
        counts = [max_vote - int(l) for l in counts]

        frequencies = [l for l in zip(unique, counts)]
        frequencies = np.array(frequencies, dtype=np.dtype(
            [('x', object), ('y', int)])).T

        # sort those candidates based on the voting count
        # if two candidates have the same counting, then sort by name
        sorted_freqs = frequencies[np.argsort(frequencies, order=('y', 'x'))]
        sorted_freqs = [l[0] for l in sorted_freqs]

        # if there are some candidates without any votes, then append them to the end of the ranking
        sorted_freqs = np.hstack(
            (sorted_freqs, np.setxor1d(sorted_freqs, self._mapping)))

        return sorted_freqs


class VotingForTwo(VotingScheme):
    def compute_res(self, preferences: np.array):
        # get the voted candidates and how often they were voted
        (unique, counts) = np.unique(preferences[:, :2], return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        max_vote = max(counts)
        counts = [max_vote - int(l) for l in counts]

        frequencies = [l for l in zip(unique, counts)]
        frequencies = np.array(frequencies, dtype=np.dtype(
            [('x', object), ('y', int)])).T

        # sort those candidates based on the voting count
        # if two candidates have the same counting, then sort by name
        sorted_freqs = frequencies[np.argsort(frequencies, order=('y', 'x'))]
        sorted_freqs = [l[0] for l in sorted_freqs]

        # if there are some candidates without any votes, then append them to the end of the ranking
        sorted_freqs = np.hstack(
            (sorted_freqs, np.setxor1d(sorted_freqs, self._mapping)))

        return sorted_freqs


class AntiPluralityVoting(VotingScheme):
    def compute_res(self, preferences: np.array):
        # get the voted candidates and how often they were voted
        (unique, counts) = np.unique(preferences[:, :-1], return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        max_vote = max(counts)
        counts = [max_vote - int(l) for l in counts]

        frequencies = [l for l in zip(unique, counts)]
        frequencies = np.array(frequencies, dtype=np.dtype(
            [('x', object), ('y', int)])).T

        # sort those candidates based on the voting count
        # if two candidates have the same counting, then sort by name
        sorted_freqs = frequencies[np.argsort(frequencies, order=('y', 'x'))]
        sorted_freqs = [l[0] for l in sorted_freqs]

        # if there are some candidates without any votes, then append them to the end of the ranking
        sorted_freqs = np.hstack(
            (sorted_freqs, np.setxor1d(sorted_freqs, self._mapping)))

        return sorted_freqs


class BordaVoting(VotingScheme):
    def compute_res(self, preferences: np.array):
        m = preferences.shape[-1]
        d = {}
        for i in range(1, m):
            # get the voted candidates and how often they were voted
            (unique, counts) = np.unique(preferences[:, i], return_counts=True)
            for j, c in enumerate(unique):
                d[c] = d.get(c, 0) + counts[j] * (m - i)

        max_vote = max(d.values())
        unique = list(d.keys())
        counts = [max_vote - int(l) for l in d.values()]

        frequencies = [l for l in zip(unique, counts)]
        frequencies = np.array(frequencies, dtype=np.dtype(
            [('x', object), ('y', int)])).T

        # sort those candidates based on the voting count
        # if two candidates have the same counting, then sort by name
        sorted_freqs = frequencies[np.argsort(frequencies, order=('y', 'x'))]
        sorted_freqs = [l[0] for l in sorted_freqs]

        # if there are some candidates without any votes, then append them to the end of the ranking
        sorted_freqs = np.hstack(
            (sorted_freqs, np.setxor1d(sorted_freqs, self._mapping)))

        return sorted_freqs


def create_voting_situation(n_voters, n_candidates):
    mapping = [str(chr(i)) for i in range(65, 65 + n_candidates)]

    # shuffle the mapping in order to create random
    # preferences for each voter
    votings = []
    preference = np.asarray(mapping)
    for voter in range(n_voters):
        np.random.shuffle(preference)
        votings.append(preference.copy())

    return np.asarray(votings), mapping


def create_voting_situation_from_file(file):
    #open file from path
    with open('./votingsituation.txt', 'r') as f:
        first_line = f.readline().split(',')
        votings = []
        #fill with first preferences
        for c in first_line:
            votings.append([c])
        #append with following preferences
        for l in f:
            tmp = l.split(',')
            for i, c in enumerate(tmp):
                votings[i].append(c)
    return votings, np.asarray([str(chr(i)) for i in range(65, 65 + len(first_line))])


def happiness(i: np.array, j: np.array, s: float = 0.9) -> float:
    i_index = {val: index for index, val in enumerate(i)}
    j_index = {val: index for index, val in enumerate(j)}
    d1 = np.abs([i_index[val] - j_index[val] for val in i])
    d2 = np.abs([i_index[val] - j_index[val] for val in j])
    d = d1 + d2
    m = len(i)
    # TODO s1+s2 doesn't work for small m (m<7)
    s1 = np.power(s, np.power(range(m), 2))
    h_hat = np.sum(d * (s1 + s1[::-1])) / m
    return np.exp(-h_hat)


def s_voter_manipulation(votings):
    _votings = votings.copy()
    manipulated_preferences = []

    # temporary save the original outcome to calculate the original happiness for
    # each voter to compare the manipulation with
    for i, voting in enumerate(tqdm.tqdm(_votings)):
        o_outcome = active_scheme.compute_res(_votings)
<<<<<<< HEAD
        o_happiness = hf(o_outcome, voting)
        overall_o_happiness = o_happiness / len(_votings)
=======
        o_happiness = happiness(o_outcome, voting)
>>>>>>> 4e06b3cb6d1b38f7f3bebc4e9d0209ce28bbd557

        for j, manipulation in enumerate(multiset_permutations(voting)):
            # set the manipulation into the votings array to test it
            _votings[i] = np.asarray(manipulation)
            outcome = active_scheme.compute_res(_votings)
<<<<<<< HEAD
            h_val = hf(outcome, voting)
            overall_h_val = h_val / len(_votings)

=======
            h_val = happiness(outcome, voting)
>>>>>>> 4e06b3cb6d1b38f7f3bebc4e9d0209ce28bbd557

            # if the h_val is higher (better) than before, then store this manipulation
            if h_val > o_happiness:
                ordering = [i, np.asarray(manipulation), outcome, h_val, o_happiness, overall_h_val, overall_o_happiness]
                manipulated_preferences.append(ordering)

        # revert the votings to the original, in order to test other manipulations
        _votings[i] = voting

    return np.asarray(manipulated_preferences)


def get_winners(votings):
    return active_scheme.compute_res(np.asarray(votings))


def overall_happiness(origin_votings, group, result):
    overall_happiness = 0
    for member in group:
        member_happiness = happiness(
            np.array(result), np.array(origin_votings[member]))
        overall_happiness += member_happiness
    overall_happiness /= len(group)
    return overall_happiness


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
    for permutation in tqdm.tqdm(range(len(all_permutations))):
        temp_votings = origin_votings.copy()
        for member in coalition:
            coalition_votings.append(temp_votings[member])
            coalition_happiness_i.append(
                happiness(np.array(winners), np.array(temp_votings[member])))
            temp_votings[member] = all_permutations[permutation]
        temp_winners = get_winners(temp_votings)
        all_happier = 1
        for member in range(len(coalition)):
            temp_happiness = happiness( np.array(temp_winners), np.array(coalition_votings[member]))
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
        new_teams.append([team, multiple_voter_manipulations(
            votings, [i, j], all_permutations)])
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
                    teams.append([[i, j], multiple_voter_manipulations(
                        votings, [i, j], all_permutations)])
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
            temp_overall_happiness = overall_happiness(
                origin_votings, coalition[0], temp_winners)
            manipulation_obj = (manipulation, temp_overall_happiness)
            all_happinesses_manipulations.append(manipulation_obj)
        probs_coalition = get_manipulation_probability(
            all_happinesses_manipulations)
        probs.append((coalition, probs_coalition))
    return probs


def counter_voting(origin_votings, coalition1, coalition2, permutations):
    origin_winners = get_winners(origin_votings)
    origin_happiness = overall_happiness(
        origin_votings, coalition1, origin_winners)

    # generate best tactical voting of opponent
    coalition_probs = get_coalition_probablity(
        origin_votings, list(coalition2))
    max_coalition = max(coalition_probs, key=lambda x: x[1])
    for member in coalition2[0]:
        origin_votings[member] = max_coalition[0]

    temp_new_origin_votings = origin_votings.copy()

    # counter this tactical voting

    all_counter_tactical_votings = []

    for permutation in permutations:
        for member in coalition1[0]:
            temp_new_origin_votings[member] = permutation

        temp_winners = get_winners(temp_new_origin_votings)
        temp_happiness = overall_happiness(temp_new_origin_votings, coalition1, temp_winners)

        if temp_happiness >= origin_happiness:
            all_counter_tactical_votings.append((permutation, temp_happiness))

    return max(all_counter_tactical_votings, key=lambda x: x[1])


def create_groups_final(votings, threshold):
    copiedvotings = [v for v in votings]
    helpvotingscopy = [v for v in votings]
    new_groups = []
    indices = []
    idx = 0
    while len(helpvotingscopy) != 0:
        start = helpvotingscopy.pop(0)
        idx += 1
        indexgroup = []
        indexgroup.append(idx)
        group = []
        group.append(start)
        new_votings = []
        for i, voting in enumerate(helpvotingscopy):
            if get_voter_compatibility(voting, start) <= threshold:
                # del voting, votings
                group.append(voting)
                indexgroup.append(idx + i)
            else:
                new_votings.append(voting)
        helpvotingscopy = new_votings
        new_groups.append(group)
        indices.append(indexgroup)
    return new_groups, indices


def visualize_manipulations(manipulations: np.array, voter: int = -1, title="", p=plt):
    # if no voter was specified, then draw the graph for every voter
    # available in manipulations
    n_voters = manipulations[:, 0].max() + 1
    if voter == -1:
        for i in range(n_voters):
            b = manipulations[manipulations[:, 0] == i][:, 2]
            p.bar(range(len(b)), b)

        p.legend([f"Voter {j}" for j in range(n_voters)])

    else:
        b = manipulations[manipulations[:, 0] == voter][:, 2]
        p.bar(range(len(b)), b)
        p.legend([f"Voter {voter}"])

    p.title.set_text(title)
    p.set_xlabel("Permutation")
    p.set_ylabel("Happiness")


if __name__ == '__main__':
    global active_scheme

    #file input
    inputfile = ''
    fileinput = False
    try:
        opts, args = getopt.getopt(sys.argv, "hi:")
    except getopt.GetoptError:
        print('tva.py [-h] [-i <inputfile>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('tva.py [-h] [-i <inputfile>]')
            sys.exit()
        elif opt == '-i':
            inputfile = arg
            fileinput = True
        else:
            print('tva.py [-h] [-i <inputfile>]')
            sys.exit()
    if (not os.path.exists(inputfile)) and (inputfile != ''):
        print('The provided inputfile is not valid!')
        sys.exit()

    #voting schemes
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    schemes = [(VotingForOne, "Voting for 1"), (VotingForTwo, "Voting for 2"),
               (AntiPluralityVoting, "Anti-Plural Voting"), (BordaVoting, "Borda Voting")]

    scheme_titles = []
    for i, scheme in enumerate(schemes):
        scheme_titles.append(f"({i + 1}) {scheme[1]};")

    print(" ".join(scheme_titles))

    try:
        scheme_idx = int(input("Select the Voting Scheme: ")) - 1
    except:
        print("Wrong input type given. Exit.")
        quit()

    if scheme_idx < 0 or scheme_idx > len(schemes) - 1:
        print(f"The Voting scheme {scheme_idx} does not exist.")
        quit()

    #voting sitations
    try:
        voters = int(input("Insert amount of Voters: "))
        candidates = int(input("Insert amount of Candidates: "))
    except:
        print("Wrong input type given. Exit.")
        quit()
    print()

    if fileinput:
        votings, mapping = create_voting_situation_from_file(inputfile)
    else:
        votings, mapping = create_voting_situation(voters, candidates)

    active_scheme = schemes[scheme_idx][0](mapping)

    print("Active Voting Scheme: ", schemes[scheme_idx][1])
    possible_manipulations = s_voter_manipulation(votings)

    original_winner = get_winners(votings)
    print("\tNon-strategic voting outcome: ", original_winner)
    original_happinesses = happiness(np.array(original_winner), np.array(votings))
    print("\thppiness level for each voter i: ", original_happinesses)
    print("\toverall happiness: ", sum(original_happinesses)/len(original_happinesses))

    for possible_manipulation in possible_manipulations:
        print(f"\tstrategic voting option for voter {possible_manipulation[0]}: "
              f"\n \t manipulation: {possible_manipulation[1]},"
              f"\n \t outcome: {possible_manipulation[2]},"
              f"\n \t manipulated happiness: {possible_manipulation[3]}"
              f"\n \t true happiness: {possible_manipulation[4]} "
              f"\n \t overall manipulated happiness: {possible_manipulation[5]}"
              f"\n \t overall true happiness: {possible_manipulation[6]}")

    p = ax[idx[i][0]][idx[i][1]]
    visualize_manipulations(possible_manipulations, title=scheme[1], p=p)

    groups, indices = create_groups_final(votings, 20)

    tactical_votings = []
    singleGroupCount = 0
    for i, group in enumerate(groups):
        if len(group) == 1:
            singleGroupCount += 1
        tactical_votings.append([indices[i], multiple_voter_manipulations(
            votings, indices[i], list(multiset_permutations(votings[0])))])
        # tactical_votings.append(multiple_voter_manipulations(votings, group, possible_manipulations)

    risk_groups = 100 - (singleGroupCount / voters * 100)

    print("\tAmount of Manipulations: ", len(possible_manipulations))

    # # risk function
    pmcount = len(possible_manipulations)
    risk_single = (pmcount / (voters * math.factorial(candidates))) * 100
    print("\tRisk of Single Manipulation: ", risk_single)
    print("\tRisk of Group Manipulation: ", risk_groups)
    print()

    fig.show()
    input("Showing graphs. Waiting... (press ENTER to continue)")
    # tactical_votings -> [ coalition1, coalition2, coalition3 ] where coalition [ [ members ], [manipulations] ]

    # groups = create_groups_onlybiggest(votings, 20)

    # similar voters -> list of list (different voting preferences among similar voters -> compare happyness to origin preference)
    #                -> for each group manipulation needs to be better than for each voters origin preference

    #
    # inputfile structure:
    # C,B,C,B,B
    # A,D,D,D,C
    # D,C,A,D,C
    # B,A,B,A,A
