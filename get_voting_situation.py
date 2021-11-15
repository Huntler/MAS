import random

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

if __name__ == '__main__':

    votings = get_voting_situation(100, 10)
    print(votings)



