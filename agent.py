import jax
import jax.numpy as jnp
from flax import linen as nn  # Linen API


class Encoder(nn.Module):
    """Encoder module that embeds obs and a_t-1"""

    @nn.compact
    def __call__(self, obs, action):
        # obs [n, h, w], action [n, 1]
        obs = nn.Embed(4, 4)(obs)  # [n, h, w, 4]
        # obs = jnp.transpose(obs, axes=[0, 3, 1, 2])  # fixme: bug here
        x = nn.Conv(features=8, kernel_size=(3, 3), strides=2)(obs)
        x = nn.tanh(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=2)(x)
        x = nn.tanh(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        xa = nn.Embed(4, 4)(action)  # [n, 1, 4]
        xa = nn.tanh(xa)
        xa = xa.reshape((xa.shape[0], -1))  # [n, 4]

        return x, xa


class Policy(nn.Module):
    """A simple pfc model."""
    output_size: int
    theta_hidden_size: int
    theta_fast_size: int
    bottleneck_size: int
    drop_out_rate: float

    @nn.compact
    def __call__(self, theta, obs_embeds, action_embeds, hipp_hidden, noise_key, outside_hipp_info):
        # obs_embeds [n, d](-1~1), theta[n, h](-1~1), hipp_hidden[n, h] (often zero, (-1~1))
        # return new_theta[n, h](-1~1), output(policy logits)[n, output_size], value[n, 1], to_hipp[n, output_size]
        # during walk, theta is fixed; during replay, theta is updated
        hipp_info = nn.tanh(nn.Dense(self.bottleneck_size)(hipp_hidden))  # [64->4]
        # jax.debug.print('noise_key:{a}', a=noise_key)
        noise = jax.random.normal(noise_key, hipp_info.shape)
        # noise = jnp.zeros_like(hipp_info)
        # hipp_info = noise
        # hipp_info = jnp.where((jnp.isclose(noise_key,0)).all(), hipp_info, noise)

        outside_info_not_integrate_indicator = jnp.isclose(outside_hipp_info, 0).all(axis=-1).reshape(-1, 1)
        # jax.debug.print('outside_hipp_info:{a}',a=outside_hipp_info.sum(axis=0))
        # jax.debug.print('outside_info_not_integrate_indicator:{a}', a=outside_info_not_integrate_indicator)
        # hipp_info = jnp.where(outside_info_not_integrate_indicator, hipp_info, outside_hipp_info)

        ## TODO dropout之后是否形状改变
        input = jnp.concatenate((obs_embeds, action_embeds, hipp_info), axis=-1)
        # Original ===================================================================================================
        # input = nn.Dropout(rate=self.drop_out_rate, deterministic=not training)(input)
        # new_theta, output = nn.GRUCell(features=self.theta_hidden_size)(theta, input)
        # output = nn.Dropout(rate=self.drop_out_rate, deterministic=not training)(output)
        # todo: Large theta, vanilla RNN ------------------------------------------------------------------------------
        # input = nn.Dropout(rate=self.drop_out_rate, deterministic=not training)(input)
        input = jnp.concatenate((input, theta), axis=-1)
        # jax.debug.print('old_theta:{a}',a=theta.shape)
        fast_theta = nn.tanh(nn.Dense(self.theta_fast_size)(input)) #[104->4]
        fast_new_theta = jnp.concatenate((fast_theta, theta[:, self.theta_fast_size:]), axis=-1)
        new_theta = nn.tanh(nn.Dense(self.theta_hidden_size)(input))#[104->32]
        # new_theta = jnp.where(integrated_indicator, new_theta, fast_new_theta)
        output = nn.tanh(nn.Dense(self.output_size)(new_theta))#[32->4]
        # output = nn.Dropout(rate=self.drop_out_rate, deterministic=not training)(output)
        # =============================================================================================================
        policy = nn.Dense(self.output_size)(output)# [4->4]
        value = nn.Dense(1)(
            nn.tanh(nn.Dense(64)(new_theta)))# [32->64->1]
              # fixme: separate actor and critic networks can achieve better results

        to_hipp = nn.tanh(nn.Dense(8)(nn.tanh(nn.Dense(self.bottleneck_size)(new_theta))))  # [32->4->8]
        # to_hipp = jnp.where(noise_key is None, to_hipp, jax.random.normal(noise_key, to_hipp.shape))
        # noise = jnp.zeros_like(to_hipp)
        # to_hipp = noise
        # to_hipp = nn.tanh(nn.Dense(self.bottleneck_size)(new_theta))

        # new_theta = jnp.where(hipp_hidden.sum(axis=-1).reshape(-1, 1) > 0, new_theta, theta)  # fixme:
        return new_theta, (policy, value, to_hipp, hipp_info)


# class Hippo(nn.Module):
#     """A simple hippo model.(rnn)"""
#     hidden_size: int
#     output_size: int

#     @nn.compact
#     def __call__(self, hipp_hidden, pfc_input, encoder_inputs, rewards):
#         # pfc_input [n, d](-1~1), encoder_input(obs_embed[n, d], action_embed[n, d])(-1~1)), hipp_hidden[n, d](-1~1)
#         # rewards[n, 1]
#         obs_embed, action_embed = encoder_inputs
#         new_hidden = nn.Dense(features=self.hidden_size)(jnp.concatenate(
#             (obs_embed, action_embed, pfc_input, hipp_hidden, rewards), axis=-1))
#         new_hidden = nn.tanh(new_hidden)
#         output = nn.Dense(self.output_size)(new_hidden)
#         return new_hidden, output

class Hippo(nn.Module):
    """A simple hippo model.(rnn)"""
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, hipp_hidden, pfc_input, encoder_inputs, rewards):
        # pfc_input [n, d](-1~1), encoder_input(obs_embed[n, d], action_embed[n, d])(-1~1)), hipp_hidden[n, d](-1~1)
        # rewards[n, 1]
        obs_embed, action_embed = encoder_inputs
        new_hidden, output = nn.GRUCell()(hipp_hidden, jnp.concatenate((obs_embed, action_embed, pfc_input, rewards), axis=-1))
        # new_hidden, output = nn.GRUCell(features=self.hidden_size)(hipp_hidden, jnp.concatenate((obs_embed, action_embed, pfc_input, rewards), axis=-1))
        output = nn.sigmoid(nn.Dense(self.output_size)(output))
        return new_hidden, output


if __name__ == '__main__':
    pass
