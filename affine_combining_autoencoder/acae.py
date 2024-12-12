# Copyright (C) 2023 István Sárándi
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

import numpy as np
import tensorflow as tf

import fleras
from fleras.util.easydict import EasyDict


def main():
    n_latent_sided = 40
    n_latent_center = 8
    batch_size = 32
    regul_lambda = 6e-1
    training_epochs = 30

    poses_train = np.load('poses_train.npy')  # Shape [n_train_examples, n_joints, 3]
    poses_test = np.load('poses_test.npy')  # Shape [n_test_examples, n_joints, 3]

    # Get name of each joint, left ones start with l, right with r
    # This will be a list like ['lhip', 'lknee', 'lankle', 'rhip', 'rknee', 'rankle', 'spine', ...]
    # Left-right pairs should have the same name except for the first letter, which is l for left
    # and r for right joints. Central joints (e.g., spine, pelvis, head, neck) should not start
    # with either l or r.
    joint_names = list(np.load('joint_names.npy'))
    w1, w2 = train_acae(
        poses_train=poses_train, poses_test=poses_test, joint_names=joint_names,
        n_latent_sided=n_latent_sided, n_latent_center=n_latent_center, batch_size=batch_size,
        regul_lambda=regul_lambda, training_epochs=training_epochs)
    np.savez('result.npz', w1=w1, w2=w2)


def train_acae(
        poses_train, poses_test, joint_names, n_latent_sided, n_latent_center, batch_size,
        regul_lambda, training_epochs):
    left_ids = [i for i, name in enumerate(joint_names) if name[0] == 'l']
    right_ids = [joint_names.index('r' + name[1:])
                 for i, name in enumerate(joint_names) if name[0] == 'l']
    center_ids = [i for i, name in enumerate(joint_names) if name[0] not in 'lr']
    permutation = left_ids + right_ids + center_ids
    inv_permutation = tf.math.invert_permutation(permutation)

    # Permute the joints, such that left ones come first, then right ones, then center ones
    poses_train = poses_train[:, permutation]
    poses_test = poses_test[:, permutation]

    train_ds = tf.data.Dataset.from_tensor_slices(dict(pose3d=poses_train)).shuffle(
        len(poses_train), reshuffle_each_iteration=True).repeat().batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(dict(pose3d=poses_test)).batch(batch_size)

    model = AffineCombiningAutoencoder(
        len(left_ids) + len(right_ids), len(center_ids), n_latent_sided, n_latent_center,
        chiral=True)
    trainer = AffineCombiningAutoencoderTrainer(
        model, regul_lambda=regul_lambda, use_projected_loss=True, random_seed=0)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule()))
    trainer.fit(
        train_ds, validation_data=val_ds, steps_per_epoch=len(poses_train) / batch_size,
        epochs=training_epochs, verbose=1)

    w1 = model.encoder.get_w().numpy()
    w2 = model.decoder.get_w().numpy()
    w1, w2 = permute_weights(w1, w2, inv_permutation)
    return w1, w2


class AffineCombiningAutoencoder(tf.keras.Model):
    def __init__(
            self, n_sided_joints, n_center_joints, n_latent_points_sided, n_latent_points_center,
            chiral=True):
        super().__init__()

        self.encoder = AffineCombinationLayer(
            n_sided_joints, n_center_joints, n_latent_points_sided, n_latent_points_center,
            transposed=False, chiral=chiral)

        self.decoder = AffineCombinationLayer(
            n_sided_joints, n_center_joints, n_latent_points_sided, n_latent_points_center,
            transposed=True, chiral=chiral)

    def call(self, inp):
        return self.decoder(self.encoder(inp))


class AffineCombinationLayer(tf.keras.layers.Layer):
    def __init__(
            self, n_sided_points, n_center_points, n_latent_points_sided, n_latent_points_center,
            transposed, chiral=True, **kwargs):
        super().__init__(dtype='float32', **kwargs)
        self.n_sided_points = n_sided_points
        self.n_center_points = n_center_points
        self.n_latent_points_sided = n_latent_points_sided
        self.n_latent_points_center = n_latent_points_center
        self.transposed = transposed
        self.chiral = chiral

    def get_blocks(self):
        return [
            [self.w_s[0], self.w_q[0], self.w_x[0]],
            [self.w_q[1], self.w_s[1], self.w_x[1]],
            [self.w_c[0], self.w_c[1], self.w_z[0]]]

    def build(self, input_shape):
        self.w_s = self.make_weight(
            'affine_weights_s',
            shape=(self.n_sided_points // 2, self.n_latent_points_sided // 2),
            initializers=tf.keras.initializers.random_uniform(-0.1, 1))
        self.w_c = self.make_weight(
            'affine_weights_c',
            shape=(self.n_center_points, self.n_latent_points_sided // 2),
            initializers=tf.keras.initializers.random_uniform(-0.1, 1))
        self.w_q = self.make_weight(
            'affine_weights_q',
            shape=(self.n_sided_points // 2, self.n_latent_points_sided // 2),
            initializers=tf.keras.initializers.random_uniform(-0.1, 1))
        self.w_x = self.make_weight(
            'affine_weights_x',
            shape=(self.n_sided_points // 2, self.n_latent_points_center),
            initializers=tf.keras.initializers.random_uniform(-0.1, 1))
        self.w_z = self.make_weight(
            'affine_weights_z',
            shape=(self.n_center_points, self.n_latent_points_center),
            initializers=tf.keras.initializers.random_uniform(-0.1, 1), single=True)

    def call(self, inputs):
        return tf.einsum('bjc,jJ->bJc', inputs, self.get_w())

    def get_w(self):
        w = block_concat(self.get_blocks())
        w = tf.transpose(w, (1, 0)) if self.transposed else w
        return normalize_weights(w)

    def make_weight(self, name, shape, initializers, dtype='float32', single=False):
        w1 = self.add_weight(
            name, shape=shape, dtype=dtype, initializer=initializers[0])

        if self.chiral or single:
            w2 = w1
        else:
            w2 = self.add_weight(f'{name}_2', shape=shape, dtype=dtype, initializer=initializers[1])
        return w1, w2


class AffineCombiningAutoencoderTrainer(fleras.ModelTrainer):
    def __init__(self, model, regul_lambda, use_projected_loss, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.regul_lambda = regul_lambda
        self.use_projected_loss = use_projected_loss

    def forward_train(self, inps, training):
        return dict(pose3d=self.model(inps.pose3d))

    def compute_losses(self, inps, preds):
        losses = EasyDict()
        if self.use_projected_loss:
            x, y = splat(inps.pose3d, preds.pose3d)
        else:
            x, y = inps.pose3d / 1000, preds.pose3d / 1000
        losses.main_loss = tf.reduce_mean(tf.abs(x - y))

        w1 = self.model.encoder.get_w()
        w2 = self.model.decoder.get_w()
        mean_w1 = tf.reduce_mean(tf.abs(w1))
        mean_w2 = tf.reduce_mean(tf.abs(w2))

        losses.regul = mean_w1 + mean_w2
        losses.loss = losses.main_loss + self.regul_lambda * losses.regul
        return losses


@fleras.optimizers.schedules.wrap(jit_compile=True)
def lr_schedule(step):
    if step < 150000:
        return 3e-2
    if step < 300000:
        return 3e-3
    else:
        return 3e-4

def splat(x, y):
    z_mean = tf.reduce_mean(x[..., 2:], axis=1, keepdims=True)
    return x[..., :2] / x[..., 2:] * z_mean / 1000, y[..., :2] / y[..., 2:] * z_mean / 1000


def normalize_weights(w):
    return w / tf.reduce_sum(w, axis=0, keepdims=True)


def block_concat(inp):
    return tf.concat([tf.concat(arrs, axis=1) for arrs in inp], axis=0)


def block_split(inp, row_sizes, col_sizes):
    return [tf.split(row, col_sizes, axis=1) for row in tf.split(inp, row_sizes, axis=0)]


def permute_weights(w1, w2, permutation):
    return w1[..., permutation, :], w2[..., :, permutation]




if __name__ == '__main__':
    main()
