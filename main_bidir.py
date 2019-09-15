
import numpy as np
import random
import argparse


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

class EmpModel:
    def __init__(self, shape, root=None):
        self.shape = shape
        self.tr_y = np.zeros((shape[0] - 1, shape[1]), dtype=bool)
        self.tr_x = np.zeros((shape[0], shape[1] - 1), dtype=bool)
        self.root = root or [np.random.randint(s) for s in shape]
        self.build_tree(self.root)

    def build_tree(self, root):
        reachable = np.zeros(self.shape, dtype=bool)
        reachable[root[0], root[1]] = True
        while reachable.sum() < np.prod(self.shape):
            # sample random direction
            a = np.random.choice([UP, DOWN, LEFT, RIGHT])
            next_states = np.zeros(self.shape, dtype=bool)
            if a == UP:
                next_states[:-1] = reachable[1:]
                new_states = next_states & ~reachable
                self.tr_y |= new_states[:-1]
            if a == DOWN:
                next_states[1:] = reachable[:-1]
                new_states = next_states & ~reachable
                self.tr_y |= new_states[1:]
            if a == LEFT:
                next_states[:, :-1] = reachable[:, 1:]
                new_states = next_states & ~reachable
                self.tr_x |= new_states[:, :-1]
            if a == RIGHT:
                next_states[:, 1:] = reachable[:, :-1]
                new_states = next_states & ~reachable
                self.tr_x |= new_states[:, 1:]
            reachable |= next_states

    def extend_reach(self, init_set):
        new_tr_x = np.zeros_like(self.tr_x)
        new_tr_y = np.zeros_like(self.tr_y)
        for a in np.random.permutation([UP, DOWN, LEFT, RIGHT]):
            if a == UP:
                next_states = self.tr_y & init_set[1:]
                new_tr_y |= ~init_set[:-1] & next_states
            if a == DOWN:
                next_states = self.tr_y & init_set[:-1]
                new_tr_y |= ~init_set[1:] & next_states
            if a == LEFT:
                next_states = self.tr_x & init_set[:, 1:]
                new_tr_x |= ~init_set[:, :-1] & next_states
            if a == RIGHT:
                next_states = self.tr_x & init_set[:, :-1]
                new_tr_x |= ~init_set[:, 1:] & next_states
        return new_tr_x, new_tr_y

    @property
    def trans(self):
        return self.tr_x, self.tr_y


class NaiveModel:
    def __init__(self, shape, p_known=0.1, root=None):
        self.shape = shape
        self.tr_x = np.random.binomial(1, p_known, size=(shape[0], shape[1] - 1)).astype(bool)
        self.tr_y = np.random.binomial(1, p_known, size=(shape[0] - 1, shape[1])).astype(bool)

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
            next_states = np.zeros_like(reachable_states)
            next_states[:-1] |= self.tr_y & reachable_states[1:]
            next_states[1:] |= self.tr_y & reachable_states[:-1]
            next_states[:, :-1] |= self.tr_x & reachable_states[:, 1:]
            next_states[:, 1:] |= self.tr_x & reachable_states[:, :-1]

            reachable_states = reachable_states | next_states
        return reachable_states

    def update_model(self, new_transitions):
        self.tr_x |= new_transitions[0]
        self.tr_y |= new_transitions[1]


def plot_model(trans, lw=3):
    tr_x, tr_y = trans
    x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    x, y = x + 0.5, y + 0.5
    plt.xticks(np.arange(size[1]))
    plt.yticks(np.arange(size[0]))
    plt.grid(lw=lw/3)
    plt.xlim(0, size[1])
    plt.ylim(0, size[0])
    px, py = x[:-1][tr_y], y[:-1][tr_y]
    plt.plot(np.stack((px, px)), np.stack((py, py+1)), 'k', lw=lw)
    px, py = x[:, :-1][tr_x], y[:, :-1][tr_x]
    plt.plot(np.stack((px, px+1)), np.stack((py, py)), 'k', lw=lw)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    parser.add_argument('--size', default=8, type=int)
    parser.add_argument('--p_init', default=0.2, type=float)
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--savefig', default=False, action='store_true')
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
    b_model = [(b.tr_x.copy(), b.tr_y.copy())]
    n_comm_tot = 0
    n_init = b_model[0][0].sum() + b_model[0][1].sum()
    while b_reach[-1].sum() < np.prod(size):
        print('empowerment of B: %s' % b_reach[-1].sum())
        tr_new = a.extend_reach(b_reach[-1])
        n_comm = tr_new[0].sum() + tr_new[1].sum()
        n_comm_tot += n_comm
        print('communicating %s transitions' % n_comm)
        b.update_model(tr_new)
        b_reach.append(b.reachable_states)
        b_model.append((b.tr_x.copy(), b.tr_y.copy()))

    print('final empowerment of B: %s' % b_reach[-1].sum())
    print('transitions known initially: %s' % n_init)
    print('transitions communicated: %s' % n_comm_tot)

    if args.savefig:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plot_model(a.trans, lw=1)
        plt.savefig('a.pdf')

        plt.figure(figsize=(10,10))
        plot_model(b_model[-1], lw=1)
        plt.savefig('b.pdf')

    if args.plot:
        import matplotlib.pyplot as plt

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

