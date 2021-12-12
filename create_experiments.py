from main2 import create_voting_situation
from numpy import savetxt, transpose


def create_experiment(n_voters, n_candidates, filename):
    voting_situation, _ = create_voting_situation(n_voters, n_candidates)
    voting_situation = transpose(voting_situation)
    savetxt(filename, voting_situation, delimiter=',', fmt='%s')


if __name__ == '__main__':
    # head boy/girl, 3 candidates, 600 voter

    create_experiment(15, 3, 'head_boy_girl.txt')

    # vote for party location, 4 friends, 4 locations

    create_experiment(3, 3, 'vote_for_party_location.txt')

    # 2 friends decide on one out of 20 restaurants

    create_experiment(2, 5, 'restaurant_vote.txt')
