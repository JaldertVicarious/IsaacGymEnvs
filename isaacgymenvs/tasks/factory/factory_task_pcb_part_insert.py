# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for pcb part insertion.

Inherits pcb environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskPcbPartInsert
"""

import hydra
import math
import omegaconf
import os
import torch


import roma
import numpy as np
from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_pcb import FactoryEnvPcb
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
from isaacgymenvs.utils import torch_jit_utils


class FactoryTaskPcbPartInsert(FactoryEnvPcb, FactoryABCTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        asset_info_path = "../../assets/factory/yaml/factory_asset_info_pcb.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_pcb = hydra.compose(config_name=asset_info_path)
        self.asset_info_pcb = self.asset_info_pcb[""][""][""][""][""][""]["assets"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting

        ppo_path = "train/FactoryTaskPcbPartInsertPPO.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # part/board tensors
        self.part_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.board_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        # Keypoint tensors
        # TODO: Add keypoints on prongs and below board at the correct locations.
        # The keypoints are then also not lines but an array of 2x4 pins.

        # TODO: offsets might need to be different for part and board.
        self.part_keypoint_offsets = torch.tensor(
            [
                [-0.05, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.0, -0.05],
                [0.0, 0.0, 0.05],
                [-0.00381, 0.00381, -0.00575],
                # [-0.00127, 0.00381, -0.004],
                # [0.00127, 0.00381, -0.004],
                # [0.00381, 0.00381, -0.004],
                # [-0.00381, -0.00381, -0.004],
                # [-0.00127, -0.00381, -0.004],
                # [0.00127, -0.00381, -0.004],
                [0.00381, -0.00381, -0.00575],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        delta = (0.0035 + 0.00157) / 2

        self.board_keypoint_offsets = torch.tensor(
            [
                [-0.05, 0.0, delta],
                [0.05, 0.0, delta],
                [0.0, 0.0, -0.05 + delta],
                [0.0, 0.0, 0.05 + delta],
                [-0.00381, 0.00381, -0.00575 + delta],
                # [-0.00127, 0.00381, -0.004],
                # [0.00127, 0.00381, -0.004],
                # [0.00381, 0.00381, -0.004],
                # [-0.00381, -0.00381, -0.004],
                # [-0.00127, -0.00381, -0.004],
                # [0.00127, -0.00381, -0.004],
                [0.00381, -0.00381, -0.00575 + delta],
            ],
            dtype=torch.float32,
            device=self.device
        )



        # NOTE: Key points seem to be aligned along z-axis in part-board example.
        # This would ensure an alignment of top-down grasp.

        # self.keypoint_offsets = (
        #     self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints)
        #     * self.cfg_task.rl.keypoint_scale
        # )
        self.keypoints_part = torch.zeros(
            (self.num_envs, 6, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_board = torch.zeros_like(self.keypoints_part, device=self.device)

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pos of keypoints on gripper, part, and board in world frame
        for idx, keypoint_offset in enumerate(self.part_keypoint_offsets):
            self.keypoints_part[:, idx] = torch_jit_utils.tf_combine(
                self.part_quat,
                self.part_pos,
                self.identity_quat,
                (keypoint_offset + self.part_pos_local),
            )[1]
        
        
        for idx, keypoint_offset in enumerate(self.board_keypoint_offsets):
            self.keypoints_board[:, idx] = torch_jit_utils.tf_combine(
                self.board_quat,
                self.board_pos,
                self.identity_quat,
                (keypoint_offset + self.board_pos_local ),
            )[1]

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        # In this policy, episode length is constant
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
        if is_last_step:
            self._open_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
            self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # This is a (very ugly) way to render key points
        self.gym.clear_lines(self.viewer)

        # Render key points

        # TODO: Ensure that the right keypoints line up.
        eps = 0.001 / 2

        for idx in range(6):
            # P_W
            x, y, z = self.keypoints_part[0, idx].cpu()

            self.gym.add_lines(
                self.viewer,
                self.gym.get_env(self.sim, 0),
                3,
                np.array(
                    [
                        [x - eps, y, z],
                        [x + eps, y, z],
                        [x, y - eps, z],
                        [x, y + eps, z],
                        [x, y, z - eps],
                        [x, y, z + eps],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [[255.0, 0.0, 0.0], [255.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
            )

            x, y, z = self.keypoints_board[0, idx].cpu()

            self.gym.add_lines(
                self.viewer,
                self.gym.get_env(self.sim, 0),
                3,
                np.array(
                    [
                        [x - eps, y, z],
                        [x + eps, y, z],
                        [x, y - eps, z],
                        [x, y + eps, z],
                        [x, y, z - eps],
                        [x, y, z + eps],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [[0.0, 255.0, 0.0], [0.0, 255.0, 0.0], [0.0, 255.0, 0.0]],
                    dtype=np.float32,
                ),
            )

        # TODO: This seems pretty noisy, and the norm seems large. Is my interpretation correct?
        # 
        # Render contact forces on fingers
        # env x 3
        fx, fy, fz = self.left_finger_force[0, :].cpu()
        x, y, z = self.left_finger_pos[0, :].cpu()
        tx = gymapi.Transform(
            gymapi.Vec3(*self.left_finger_pos[0, :].cpu()),
            gymapi.Quat(*self.left_finger_quat[0, :].cpu()),
        )

        vec = tx.transform_point(gymapi.Vec3(fx, fy, fz))

        self.gym.add_lines(
            self.viewer,
            self.gym.get_env(self.sim, 0),
            1,
            np.array(
                [
                    [x, y, z],
                    [x + vec.x, y + vec.y, z + vec.z],
                ],
                dtype=np.float32,
            ),
            np.array(
                [[0.0, 0.0, 255.0]],
                dtype=np.float32,
            ),
        )


        # Shallow copies of tensors
        obs_tensors = [
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_linvel,
            self.fingertip_midpoint_angvel,
            self.part_pos,
            self.part_quat,
            self.board_pos,
            self.board_quat,
        ]

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)

        return self.obs_buf

    def compute_reward(self):
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = (
            torch.norm(self.actions, p=2, dim=-1)
            * self.cfg_task.rl.action_penalty_scale
        )

        self.rew_buf[:] = (
            keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
            - action_penalty * self.cfg_task.rl.action_penalty_scale
        )

        # In this policy, episode length is constant across all envs
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        if is_last_step:
            # Check if part is close enough to board
            is_part_close_to_board = self._check_part_close_to_board()
            self.rew_buf[:] += is_part_close_to_board * self.cfg_task.rl.success_bonus
            self.extras["successes"] = torch.mean(is_part_close_to_board.float())

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        # Close gripper onto part
        self.disable_gravity()  # to prevent part from falling
        for _ in range(self.cfg_task.env.num_gripper_close_sim_steps):
            self.ctrl_target_dof_pos[env_ids, 7:9] = 0.0
            delta_hand_pose = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )  # no arm motion
            self._apply_actions_as_ctrl_targets(
                actions=delta_hand_pose, ctrl_target_gripper_dof_pos=0.0, do_scale=False
            )
            self.gym.simulate(self.sim)
            self.render()
        self.enable_gravity(gravity_mag=self.cfg_base.sim.gravity_mag)

        # self._randomize_gripper_pose(
        #     env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        # )

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = (
            torch.cat(
                (
                    torch.tensor(
                        self.cfg_task.randomize.franka_arm_initial_dof_pos,
                        device=self.device,
                    ),
                    # NOTE: this depends on width of part to-be-grasped.
                    torch.tensor(
                        [0.007 * 0.5 * 1.1],
                        device=self.device,
                    ),
                    torch.tensor(
                        [0.007 * 0.5 * 1.1],
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
            .unsqueeze(0)
            .repeat((self.num_envs, 1))
        )  # shape = (num_envs, num_dofs)
        # self.dof_pos[env_ids] = torch.cat(
        #     (
        #         torch.tensor(
        #             self.cfg_task.randomize.franka_arm_initial_dof_pos,
        #             device=self.device,
        #         ).repeat((len(env_ids), 1)),
        #         torch.tensor([[(0.01 * 0.5) * 1.1]], device=self.device),  # buffer on gripper DOF pos to prevent initial contact
        #         torch.tensor([[(0.01 * 0.5) * 1.1]], device=self.device),
        #     ),  # buffer on gripper DOF pos to prevent initial contact
        #     dim=-1,
        # )  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _reset_object(self, env_ids):
        """Reset root states of part and board."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of part within gripper
        self.root_pos[env_ids, self.part_actor_id_env, 0] = 0.0
        self.root_pos[env_ids, self.part_actor_id_env, 1] = 0.0
        fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset
        self.root_pos[env_ids, self.part_actor_id_env, 2] = (
            fingertip_midpoint_pos_reset + 0.004
        )

        part_noise_pos_in_gripper = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        part_noise_pos_in_gripper = part_noise_pos_in_gripper @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.part_noise_pos_in_gripper, device=self.device
            )
        )
        # self.root_pos[env_ids, self.part_actor_id_env, :] += part_noise_pos_in_gripper[
        #     env_ids
        # ]

        # part_rot_euler = torch.tensor([0.0, 0.0, math.pi * 0.5], device=self.device).repeat(len(env_ids), 1)
        # part_noise_rot_in_gripper = \
        #     2 * (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        # part_noise_rot_in_gripper *= self.cfg_task.randomize.part_noise_rot_in_gripper
        # part_rot_euler[:, 2] += part_noise_rot_in_gripper
        # part_rot_quat = torch_utils.quat_from_euler_xyz(part_rot_euler[:, 0], part_rot_euler[:, 1], part_rot_euler[:, 2])
        # NOTE: hardcoded to lie orthogonal to gripper closing direction.
        self.root_quat[env_ids, self.part_actor_id_env] = torch.tensor(
            [0.0, 0.0, 0.707106781186547, 0.707106781186547],
            dtype=torch.float32,
            device=self.device,
        ).repeat(len(env_ids), 1)

        # Randomize root state of board
        board_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        board_noise_xy = board_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.board_pos_xy_noise, device=self.device)
        )
        self.root_pos[env_ids, self.board_actor_id_env, 0] = (
            self.cfg_task.randomize.board_pos_xy_initial[0] + board_noise_xy[env_ids, 0]
        )
        self.root_pos[env_ids, self.board_actor_id_env, 1] = (
            self.cfg_task.randomize.board_pos_xy_initial[1] + board_noise_xy[env_ids, 1]
        )
        self.root_pos[env_ids, self.board_actor_id_env, 2] = 0.57
        
        # # [0, 2 PI]
        # yaws = torch.rand(
        #     (self.num_envs, 1), dtype=torch.float32, device=self.device
        # ) * (np.pi * 2)

        # rotvecs = (
        #     torch.tensor(
        #         [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        #     ).repeat(len(env_ids), 1)
        #     * yaws
        # )

        # qyaws = roma.rotvec_to_unitquat(rotvecs)

        # self.root_quat[env_ids, self.board_actor_id_env] = qyaws
        self.root_quat[env_ids, self.board_actor_id_env] = torch.tensor([0., 0.,  0.707106781186547, 0.70710678118654], device=self.device).repeat(len(env_ids), 1)

        self.root_linvel[env_ids, self.board_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.board_actor_id_env] = 0.0

        part_board_actor_ids_sim = torch.cat(
            (self.part_actor_ids_sim[env_ids], self.board_actor_ids_sim[env_ids]), dim=0
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(part_board_actor_ids_sim),
            len(part_board_actor_ids_sim),
        )

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.fingertip_midpoint_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        if self.cfg_ctrl["do_force_ctrl"]:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.force_action_scale, device=self.device
                    )
                )

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.torque_action_scale, device=self.device
                    )
                )

            self.ctrl_target_fingertip_contact_wrench = torch.cat(
                (force_actions, torque_actions), dim=-1
            )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )  # no arm motion
        self._apply_actions_as_ctrl_targets(
            delta_hand_pose, gripper_dof_pos, do_scale=False
        )

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self, gripper_dof_pos=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance  # lift along z

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, do_scale=False
            )
            self.render()
            self.gym.simulate(self.sim)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = (
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        )

        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distances."""

        keypoint_dist = torch.sum(
            torch.norm(self.keypoints_board - self.keypoints_part, p=2, dim=-1), dim=-1
        )

        return keypoint_dist

    def _check_part_close_to_board(self):
        """Check if part is close to board."""

        keypoint_dist = torch.norm(
            self.keypoints_board - self.keypoints_part, p=2, dim=-1
        )

        is_part_close_to_board = torch.where(
            torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
            torch.ones_like(self.progress_buf),
            torch.zeros_like(self.progress_buf),
        )

        return is_part_close_to_board

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions, ctrl_target_gripper_dof_pos=0.0, do_scale=False
            )

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
