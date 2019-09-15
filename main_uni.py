
import numpy as np
import random
import argparse


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

class EmpModel:
    def __init__(self, shape, root=None):
        self.shape = shape
        self.trans = np.zeros((4,) + shape, dtype=bool)
        self.root = root or [np.random.randint(s) for s in shape]
        self.build_tree(self.root)

    def build_tree(self, root):
        reachable = np.zeros(self.shape, dtype=bool)
        reachable[root[0], root[1]] = True
        while reachable.sum() < np.prod(self.shape):
            # sample random direction
            a = random.sample({UP, DOWN, LEFT, RIGHT}, 1)[0]
            next_states = np.zeros(self.shape, dtype=bool)
            #new_trans = np.zeros((4,) + self.shape, dtype=bool)
            if a == UP:
                next_states[:-1] = np.roll(reachable, -1, 0)[:-1]
                new_states = next_states & ~reachable
                self.trans[UP, 1:] |= np.roll(new_states, +1, 0)[1:]
            if a == DOWN:
                next_states[1:] = np.roll(reachable, +1, 0)[1:]
                new_states = next_states & ~reachable
                self.trans[DOWN, :-1] |= np.roll(new_states, -1, 0)[:-1]
            if a == LEFT:
                next_states[:, :-1] = np.roll(reachable, -1, 1)[:, :-1]
                new_states = next_states & ~reachable
                self.trans[LEFT, :, 1:] |= np.roll(new_states, +1, 1)[:, 1:]
            if a == RIGHT:
                next_states[:, 1:] = np.roll(reachable, +1, 1)[:, 1:]
                new_states = next_states & ~reachable
                self.trans[RIGHT, :, :-1] |= np.roll(new_states, -1, 1)[:, :-1]
            reachable |= next_states

    def extend_reach(self, init_set):
        new_trans = np.zeros_like(self.trans)
        # up
        next_states = np.roll(self.trans[UP] & init_set, -1, 0)[:-1]
        new_trans[UP, 1:] = ~init_set[:-1] & next_states
        # dwn
        next_states = np.roll(self.trans[DOWN] & init_set, +1, 0)[1:]
        new_trans[DOWN, :-1] = ~init_set[1:] & next_states
        # lft
        next_states = np.roll(self.trans[LEFT] & init_set, -1, 1)[:, :-1]
        new_trans[LEFT, :, 1:] = ~init_set[:, :-1] & next_states
        # rgt
        next_states = np.roll(self.trans[RIGHT] & init_set, +1, 1)[:, 1:]
        new_trans[RIGHT, :, :-1] = ~init_set[:, 1:] & next_states
        return new_trans


class NaiveModel:
    def __init__(self, shape, p_known=0.1, root=None):
        self.shape = shape
        self.trans = np.random.binomial(1, p_known, size=(4,) + shape).astype(bool)
        self.trans[UP, 0, :] = False
        self.trans[DOWN, -1, :] = False
        self.trans[LEFT, :, 0] = False
        self.trans[RIGHT, :, -1] = False

        self.init_state = root or [np.random.randint(s) for s in shape]
        self.init_mask = np.zeros(shape, dtype=bool)
        self.init_mask[self.init_state[0], self.init_state[1]] = True

    @property
    def reachable_states(self):
        #init_states = self.trans.sum(0) > 0 # all inital states we know of
        reachable_states = self.init_mask
        n_pre = 0
        while reachable_states.sum() > n_pre:
            n_pre = reachable_states.sum()
            trans = self.trans & reachable_states
            next_states = np.zeros_like(reachable_states)
            next_states[:-1] |= np.roll(trans[UP], -1, 0)[:-1]
            next_states[1:] |= np.roll(trans[DOWN], +1, 0)[1:]
            next_states[:, :-1] |= np.roll(trans[LEFT], -1, 1)[:, :-1]
            next_states[:, 1:] |= np.roll(trans[RIGHT], +1, 1)[:, 1:]

            reachable_states = reachable_states | next_states
        return reachable_states

    def update_model(self, new_transitions):
        self.trans |= new_transitions



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    parser.add_argument('--size', default=8, type=int)
    parser.add_argument('--p_init', default=0.2, type=float)
    args = parser.parse_args()

    size = args.size, args.size
    x, y, = np.meshgrid(np.arange(size[1]), np.arange(size[0]))

    # set up naive agent
    b = NaiveModel(size, p_known=args.p_init)

    # set up empowered agent
    # hack: make a's tree root one of the states known by b
    root = (
        np.random.choice(y[b.reachable_states]),
        np.random.choice(x[b.reachable_states])
        )
    a = EmpModel(size, root=root)

    b_reach = [b.reachable_states]
    b_model = [True & b.trans]
    while b_reach[-1].sum() < np.prod(size):
        print('empowerment of B: %s' % b_reach[-1].sum())
        new_transitions = a.extend_reach(b_reach[-1])
        print('communicating %s transitions' % new_transitions.sum())
        b.update_model(new_transitions)
        b_reach.append(b.reachable_states)
        b_model.append(True & b.trans)

    print('final empowerment of B: %s' % b_reach[-1].sum())

    x, y = x + 0.5, y + 0.5

    def plot_model(trans):
        plt.xticks(np.arange(size[1]))
        plt.yticks(np.arange(size[0]))
        plt.grid(lw=1)
        plt.xlim(0, size[1])
        plt.ylim(0, size[0])
        px, py = x[trans[UP]], y[trans[UP]]
        plt.plot(np.stack((px, px)), np.stack((py, py-1)), 'k', lw=3)
        px, py = x[trans[DOWN]], y[trans[DOWN]]
        plt.plot(np.stack((px, px)), np.stack((py, py+1)), 'k', lw=3)
        px, py = x[trans[LEFT]], y[trans[LEFT]]
        plt.plot(np.stack((px, px-1)), np.stack((py, py)), 'k', lw=3)
        px, py = x[trans[RIGHT]], y[trans[RIGHT]]
        plt.plot(np.stack((px, px+1)), np.stack((py, py)), 'k', lw=3)

    # plot empowered agent model

    plt.figure(figsize=size)
    plt.title('Empowered model')
    plot_model(a.trans)

    n = len(b_reach)
    plt.figure(figsize=(3 * n, 3))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(np.flipud(b_reach[i]), cmap='bone', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, n, n + i + 1)
        plot_model(b_model[i])

    plt.show()

