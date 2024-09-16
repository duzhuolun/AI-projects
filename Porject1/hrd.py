import numpy as np
import sys

left = np.array([0, -1])
up = np.array([-1, 0])
right = np.array([0, 1])
down = np.array([1, 0])
DIRECTIONS = {'left': left, 'up': up, 'right': right, 'down': down}


def find_locations(state, piece):
    locations = []
    for i in range(5):
        for j in range(4):
            if state[i, j] == piece:
                locations.append([i, j])
    return np.array(locations)


def move2new_locations(state, piece, direction, location=None):
    if location is not None:
        if piece == 7:
            new_state = state.copy()
            new_state[location[0], location[1]] = 0
            new_sev_loc = location + DIRECTIONS[direction]
            new_state[new_sev_loc[0], new_sev_loc[1]] = piece
            return True, new_state
        else:
            raise Exception('invalid input')

    old_locations = find_locations(state, piece)
    new_locations = []
    new_locations = old_locations + DIRECTIONS[direction]

    # overlapping positions
    old_set = set(tuple(i) for i in old_locations)
    new_set = set(tuple(i) for i in new_locations)
    overlap_set = old_set & new_set
    new_space = new_set - overlap_set
    old_space = old_set - overlap_set

    # check if new space is all 0
    for s in new_space:
        if state[s[0], s[1]] != 0:
            return False, None

    # create new state
    new_state = state.copy()
    for s in new_space:
        new_state[s[0], s[1]] = piece
    for s in old_space:
        new_state[s[0], s[1]] = 0
    return True, new_state


def get_oppo_direction(direction):
    if direction == 'right':
        return 'left'
    elif direction == 'left':
        return 'right'
    elif direction == 'up':
        return 'down'
    elif direction == 'down':
        return 'up'
    else:
        raise Exception('invalid input')


def BoardState_h_manhattan(BoardState):
    locations_one = find_locations(BoardState.state, 1)
    top_left_corner = locations_one[0]
    goal_loc = [3, 1]
    dist = abs(top_left_corner[0] - goal_loc[0]) + abs(top_left_corner[1] - goal_loc[1])
    return dist


def BoardState_h_cus(BoardState):
  locations_zero = find_locations(BoardState.state, 0)
  loc1 = locations_zero[0]
  loc2 = locations_zero[1]
  dist_btw_zeros = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
  if dist_btw_zeros > 1:
    return 1 + BoardState_h_manhattan(BoardState)
  else:
    return BoardState_h_manhattan(BoardState)


class BoardState():

    def __init__(self, action, state, parent=None):
        self.action = action
        self.parent = parent
        self.state = state
        self.gval = 0 if self.parent == None else parent.gval + 1

    def successors(self):
        successors_list = []
        actions_bigP = []
        zero_locs = find_locations(self.state, 0)
        for zero_loc in zero_locs:
            for direc_name, direc in DIRECTIONS.items():
                curr_loc = zero_loc + direc
                if -1 < curr_loc[0] < 5 and -1 < curr_loc[1] < 4:
                    current_piece = self.state[curr_loc[0], curr_loc[1]]
                    move_direction = get_oppo_direction(direc_name)
                    if 0 < current_piece < 7:
                        action_str = move_direction + str(current_piece)
                        if action_str in actions_bigP:
                            continue
                        else:
                            is_movable, new_state = move2new_locations(self.state, current_piece, move_direction,
                                                                       location=None)
                            actions_bigP.append(action_str)
                    elif current_piece == 7:
                        is_movable, new_state = move2new_locations(self.state, current_piece, move_direction,
                                                                   location=curr_loc)
                    else:
                        continue
                    if is_movable:
                        successors_list.append(
                            BoardState('#{} move {}'.format(current_piece, move_direction), new_state, self))

        return successors_list

    def is_goal(self):
        return self.state[3, 1] == 1 and self.state[4, 2] == 1

    def get_fval(self, heuristic_func):
        return self.gval + heuristic_func(self)


def dfs(initial_boardstate, hash_func):
    expanded_set = set()
    frontier = []
    frontier.append(initial_boardstate)
    expanded_set.add(hash_func(initial_boardstate))

    while (len(frontier) != 0):
        selected = frontier.pop()
        successors = selected.successors()
        for succ in successors:
            if (succ.is_goal()):
                return succ
            if (hash_func(succ) not in expanded_set):
                frontier.append(succ)
                expanded_set.add(hash_func(succ))

    return None


def astar(initial_boardstate, hash_func, heuristic_func):
    expanded_set = set()
    frontier = []
    frontier_score = []

    frontier.append(initial_boardstate)
    expanded_set.add(hash_func(initial_boardstate))
    frontier_score.append(initial_boardstate.get_fval(heuristic_func))

    while (len(frontier) != 0):
        min_score = min(frontier_score)
        selected_index = frontier_score.index(min_score)
        selected = frontier.pop(selected_index)
        _ = frontier_score.pop(selected_index)

        successors = selected.successors()
        for succ in successors:
            if (succ.is_goal()):
                return succ
            if (hash_func(succ) not in expanded_set):
                frontier.append(succ)
                expanded_set.add(hash_func(succ))
                frontier_score.append(succ.get_fval(heuristic_func))

    return None


def find_h_v_2(state):
    h_2 = []
    v_2 = []
    for i in range(2, 7):
        locs = find_locations(state, i)
        locs_diff = locs[0] - locs[1]
        if locs_diff[0] == 0:
            h_2.append(i)
        else:
            v_2.append(i)

    return h_2, v_2


class AdvanceHash():

    def __init__(self, initial_boardstate):
        h_v_2 = find_h_v_2(initial_boardstate.state)
        self.h_2 = h_v_2[0]
        self.v_2 = h_v_2[1]

    def get_hashable(self, boardstate):

        state = boardstate.state
        new_state = state.flatten()

        for h in self.h_2:
            new_state[new_state == h] = 8
        for v in self.v_2:
            new_state[new_state == v] = 9

        return tuple(new_state)


class WriteFormat():

    def __init__(self, initial_boardstate):
        h_v_2 = find_h_v_2(initial_boardstate.state)
        self.h_2 = h_v_2[0]
        self.v_2 = h_v_2[1]

    def get_printable(self, boardstate):
        state = boardstate.state
        new_state = state.copy()

        for h in self.h_2:
            new_state[new_state == h] = 8
        for v in self.v_2:
            new_state[new_state == v] = 9

        new_state[new_state == 8] = 2
        new_state[new_state == 9] = 3
        new_state[new_state == 7] = 4

        print_str = ''
        for r in new_state:
            print_str += str('\n')
            for i in r:
                print_str += str(i)

        return print_str


def search(initial_boardstate, heuristic_func):
    ah = AdvanceHash(initial_boardstate)
    hash_func = ah.get_hashable
    return dfs(initial_boardstate, hash_func), astar(initial_boardstate, hash_func, heuristic_func)


def read_puzzle(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        nums = []
        for line in lines:
            for char in line:
                if char != '\n':
                    nums.append(int(char))

    board = np.array(nums).reshape((5, 4))
    initial_boardstate = BoardState("START", board)
    return initial_boardstate


def write_result(dfs_boardstate, astar_boardstate, dfs_out_filename, astar_out_filename):
    wf = WriteFormat(dfs_boardstate)
    for boardstate, filename in zip([dfs_boardstate, astar_boardstate], [dfs_out_filename, astar_out_filename]):
        curr_bs = boardstate
        path = []
        while curr_bs != None:
            path.append(curr_bs)
            curr_bs = curr_bs.parent
        with open(filename, 'w') as f:
            f.write('Cost of the solution: {}'.format(len(path) - 1))
            first_line = True
            for bs in reversed(path):
                if not first_line:
                    f.write('\n')
                else:
                    first_line = False
                f.write(wf.get_printable(bs))


def main(input_filename, dfs_out_filename, astar_out_filename):
    init_bs = read_puzzle(input_filename)
    a, b = search(init_bs, BoardState_h_manhattan)
    # a, b = search(init_bs, BoardState_h_cus)
    write_result(a, b, dfs_out_filename, astar_out_filename)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
