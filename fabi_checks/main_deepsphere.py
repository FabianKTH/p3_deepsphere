import numpy as np
import torch
from torch.utils import data
import torch.optim as optim

from so3_actor import SO3Actor


class SyntheticData:
    def __init__(self, actor, size=42):
        self.actor = actor
        self.size = size
        self.it_count = 0

    def __next__(self):
        if self.it_count < self.size:
            self.it_count += 1
            rnd_vec = self.actor._tt(np.random.randn(3))
            rnd_sph = self.actor._dirs_to_shsignal(rnd_vec[None, None, :])
            rnd_sph = rnd_sph[0]
            rnd_sp = self.actor._sph_to_sp(rnd_sph)
            rnd_vec = torch.nn.functional.normalize(rnd_vec, dim=-1)

            return rnd_sp, rnd_vec
        else:
            raise StopIteration

    def __iter__(self):
        return self


class IterDset(data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


def main():
    _actor      = SO3Actor()
    _actor.to(device=torch.device("cuda"))
    synth_data  = SyntheticData(_actor, 4096 * 8)
    dset        = IterDset(synth_data)
    criterion   = torch.nn.MSELoss()
    optimizer   = optim.SGD(_actor.parameters(), lr=0.01, momentum=0.9)

    loss_hist = list()

    for xi, yi in data.DataLoader(dset, batch_size=512):

        optimizer.zero_grad()
        out, _ = _actor(xi, stochastic=False)
        # out = _actor._so3_deepsphere_net(xi)
        loss = criterion(out, yi)
        # loss = criterion(out[:, 0, None], xi)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.detach().cpu().numpy().item())
        print(loss)

    pass

    # for _ in range(12):
    #     print(next(synth_data))


if __name__ == '__main__':
    main()
