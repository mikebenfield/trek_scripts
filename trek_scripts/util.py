def select_from_list(rand, lst, count):
    if count >= len(lst):
        return (lst, [])
    indices = rand.choice(len(lst), size=count, replace=False)
    indices = set(indices)
    selected = []
    others = []
    for i, item in enumerate(lst):
        if i in indices:
            selected.append(item)
        else:
            others.append(item)
    return (selected, others)

def batch_iter(rand, batch_size, lst):
    visited = []
    not_visited = lst
    use = []
    while not_visited:
        visited += use
        (use, not_visited) = select_from_list(rand, not_visited, batch_size)
        yield use

def train_test_split(rand, suffix, data_directory, shows, test_proportion):
    '''Returns (train_episodes, test_episodes).
    
    These are paths relative to the data directory.
    '''
    import pathlib

    episodes = []
    for show in shows:
        for child in pathlib.Path(data_directory, show).iterdir():
            if child.suffix == suffix:
                episodes.append(pathlib.Path(*child.parts[-2:]))
    test_size = int(test_proportion * len(episodes))
    test_episodes, train_episodes = select_from_list(rand, episodes, test_size)
    return test_episodes, train_episodes
