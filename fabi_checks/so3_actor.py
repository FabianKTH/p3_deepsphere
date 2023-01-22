import numpy as np
import pygsp as pg
import torch
import torch.nn.functional as F
from dipy.direction import Sphere
from dipy.reconst.shm import sh_to_sf_matrix
from torch import nn
from torch.distributions.normal import Normal

from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.layers.samplings.icosahedron_pool_unpool import Icosahedron
from deepsphere.models.spherical_unet.utils import SphericalChebBNPool
# deepsphere imports
from deepsphere.utils.laplacian_funcs import get_icosahedron_laplacians
# user imports
from so3_helper import _init_antipod_dict
from sphere import vertex_area, spharm_real

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = -1  # (2)
LOG_STD_MIN = -20  # (-20)


class SO3Actor(nn.Module):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: str,
    ):

        super(SO3Actor, self).__init__()

        self.l_max = 6
        self._antipod_dict = _init_antipod_dict(l_max=self.l_max)
        self.antipod_idx = self._tt(np.array(self._antipod_dict[self.l_max]),
                                    dtype=torch.int64)

        self.callcount = 0  # TODO remove, this is really dirty

        self._init_spharm_basis()  # inits self.Y, self.Y_inv, self.area
        # self._init_spharm_conv() # inits self.isht, self.sht
        self._init_deepsphere()

    @staticmethod
    def _tt(x, dtype=torch.float32):
        """
        tt: to torch, shorthand for converting numpy to torch
        """

        return torch.from_numpy(x).to(device=device, dtype=dtype)

    def _init_spharm_basis(self):
        # sphere_dir = '/fabi_project/sphere/ico4.vtk' # TODO: dont code so hard (ico_low2.vtk)

        # v, f = read_mesh(sphere_dir)
        v = pg.graphs.SphereIcosahedral(2 ** 4).coords
        f = Sphere(xyz=v).faces

        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, self.l_max, threads=1)
        self.area = self._tt(area)
        self.B_spha = self._tt(Y)
        self.v = self._tt(v)
        self.invB_spha = self.B_spha.T
        self.sphere = Sphere(xyz=v)
        self._init_dipy_basis()

    def _init_dipy_basis(self):
        B_desc, invB_desc = sh_to_sf_matrix(self.sphere, sh_order=self.l_max,
                                            basis_type='descoteaux07', full_basis=True,
                                            legacy=False, return_inv=True, smooth=0)

        B_tour, invB_tour = sh_to_sf_matrix(self.sphere, sh_order=self.l_max,
                                            basis_type='tournier07', full_basis=True,
                                            legacy=False, return_inv=True, smooth=0)

        self.B_desc, self.invB_desc = self._tt(B_desc), self._tt(invB_desc)
        self.B_tour, self.invB_tour = self._tt(B_tour), self._tt(invB_tour)

    def _init_deepsphere(self):
        self.laps = get_icosahedron_laplacians(nodes=2562, depth=4, laplacian_type="combinatorial")
        self.pooling_class = Icosahedron()

        self.conv1 = SphericalChebBNPool(1, 8, self.laps[2], self.pooling_class.pooling, kernel_size=3)
        self.conv2 = FinalSphericalChebBN(8, 2, self.laps[2], kernel_size=3)

    # @staticmethod
    def _reformat_state(self, state):
        """
        takes state from original TTL and converts it to a spherical harmonics representation
        """
        sh_end = 7 * 29  # 7 neigbourhood_size * (28 shcoeff + 1 mask)
        sh_part, dir_part = state[:, :sh_end], state[:, sh_end:]

        # sh_part
        sh_part = sh_part.reshape([-1, 7, 29])
        sh_part = self._remove_mask(sh_part)
        sh_part = self._pad_antipod(sh_part)
        # sh_part = self._change_basis(sh_part, conversion='descoteaux->spharm')
        sh_part = self._sph_to_sp(sh_part, sph_basis='descoteaux')

        # dir_part
        dir_part = dir_part.reshape([-1, 4, 3])  # 4 directions, 3 point coefficients
        dir_part = self._dirs_to_shsignal(dir_part, self.l_max)
        dir_part = self._sph_to_sp(dir_part, sph_basis='tournier')  # spharm?

        # concat channels
        out = torch.cat([sh_part, dir_part], 1)

        # normalize (TODO: spharm normalization instead of vector norm)
        # out = torch.nn.functional.normalize(out, dim=-1)

        return out

    @staticmethod
    def _remove_mask(signal):
        """
        removes the tracking mask part from the sh-signal
        """
        return signal[..., :-1]

    def _dirs_to_shsignal(self, dirs, l_max=6):
        """
        converts direction vectors to sh-signals with maximum at direction
        """
        assert dirs.shape[-1] == 3

        N, no_channels, P = dirs.shape

        sh_ = torch.zeros(size=[N, no_channels, self.nocoeff_from_l(l_max)], device=device)
        sh_[..., 1] = dirs[..., 1]
        sh_[..., 2] = dirs[..., 2]
        sh_[..., 3] = dirs[..., 0]

        return sh_

    @staticmethod
    def nocoeff_from_l(l_max):
        return (l_max + 1) ** 2

    def _pad_antipod(self, signal):
        """
        takes antipodal spharm signal and fills up 0 to shape of full spectrum spharm signal
        # TODO check and test
        """

        N, no_channels, no_coeff = signal.shape

        # zero-padding for all even degree sph harm (antipodal -> podal)
        # l_max = -1.5 + np.sqrt(0.25 + 2 * no_coeff)  # from no_sph = (l + 1)(l/2 + 1)
        # antipod_idx = self._antipod_idx # self._antipod_dict[int(l_max)]
        new_no_coeff = len(self.antipod_idx)
        idx_expanded = self.antipod_idx.expand([N, no_channels, new_no_coeff])
        signal = torch.nn.functional.pad(signal, (0, new_no_coeff - no_coeff))
        signal = torch.gather(signal.view([-1, new_no_coeff]),
                              1,
                              idx_expanded.view([-1, new_no_coeff])
                              ).view(N, no_channels, new_no_coeff)

        return signal

    def _change_basis(self, signal, conversion='tournier->descoteaux'):
        if conversion == 'tournier->descoteaux':
            return torch.matmul(torch.matmul(signal, self.B_tour), self.invB_desc)
        elif conversion == 'descoteaux->tournier':
            return torch.matmul(torch.matmul(signal, self.B_desc), self.invB_tour)
        else:
            return ValueError

    def _sph_to_sp(self, signal, sph_basis='tournier'):
        if sph_basis == 'tournier':
            return torch.matmul(signal, self.B_tour)
        elif sph_basis == 'descoteaux':
            return torch.matmul(signal, self.B_desc)
        else:
            return ValueError

    def _sp_to_sph(self, signal, sph_basis='tournier'):
        if sph_basis == 'tournier':
            return torch.matmul(signal, self.invB_tour)
        elif sph_basis == 'descoteaux':
            return torch.matmul(signal, self.invB_desc)
        else:
            return ValueError

    def _get_direction(self, signal, is_sp_signal=False):
        """
        expects output sh-signals, samples/extracts directions.
        """

        # check if signal is already in spherical domain (not spharm)
        if not is_sp_signal:
            # odf = self.isht(signal)
            raise ValueError('isph, sph from spharmnet removed!')
        else:
            odf = signal

        peak_idx = torch.argmax(odf, dim=-1)

        peak_dir = self.v[peak_idx]

        # import ipdb; ipdb.set_trace()

        return peak_dir

    def _so3_deepsphere_net(self, signal):
        # x = torch.cat([signal[:, 0, None], signal[:, -1, None]], 1)
        x = signal[:, 0, None]
        x = torch.swapaxes(x, 1, 2)

        x = self.conv1(x)  # self.laps[3])
        x = self.conv2(x)  # self.laps[3])

        return x

    def forward(
            self,
            state: torch.Tensor,
            stochastic: bool,
            with_logprob: bool = False,
    ) -> (torch.Tensor, torch.Tensor):

        # extract state data
        state = self._reformat_state(state)

        # convert state to spharm basis
        # state = self._change_basis(state, 'descoteaux->spharm')

        # below: test, TODO remove
        stochastic = False
        # state = self._change_basis(state, 'tournier->spharm') # should be descoteaux->spharm

        p = self._so3_deepsphere_net(state)
        p = self._get_direction(p, is_sp_signal=True)

        mu = p[:, 0]
        log_std = p[:, 1]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action -
                         F.softplus(-2 * pi_action))).sum(axis=1)

        # pi_action = self.output_activation(pi_action) # TODO

        return pi_action, logp_pi

    def logprob(
            self,
            state: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:

        p = self._so3_conv_net(state)
        mu = p[:, 0]
        log_std = p[:, 1]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action -
                         F.softplus(-2 * action))).sum(axis=1)

        return logp_pi


class FinalSphericalChebBN(nn.Module):
    """Building Block with a Chebyshev Convolution, Batchnormalization, and ReLu activation.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.
        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]
        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        return x
