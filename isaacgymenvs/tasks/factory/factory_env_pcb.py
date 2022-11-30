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

"""Factory: class for part-board env.

Inherits base class and abstract environment class. Inherited by part-board task classes. Not directly executed.

Configuration defined in FactoryEnvPcb.yaml. Asset info defined in factory_asset_info_part_board.yaml.
"""

import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgymenvs.tasks.factory.factory_base import FactoryBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvPcb(FactoryBase, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/FactoryEnvPcb.yaml'  # relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_pcb.yaml'
        self.asset_info_pcb = hydra.compose(config_name=asset_info_path)
        self.asset_info_pcb = self.asset_info_pcb['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        part_asset, board_asset = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, part_asset, board_asset, table_asset)

    def _import_env_assets(self):
        """Set part and board asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'urdf')

        part_options = gymapi.AssetOptions()
        part_options.flip_visual_attachments = False
        part_options.fix_base_link = False
        part_options.thickness = 0.0  # default = 0.02
        part_options.armature = 0.0  # default = 0.0
        part_options.use_physx_armature = True
        part_options.linear_damping = 0.0  # default = 0.0
        part_options.max_linear_velocity = 1000.0  # default = 1000.0
        part_options.angular_damping = 0.0  # default = 0.5
        part_options.max_angular_velocity = 64.0  # default = 64.0
        part_options.disable_gravity = False
        part_options.enable_gyroscopic_forces = True
        part_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        part_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            part_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        board_options = gymapi.AssetOptions()
        board_options.flip_visual_attachments = False
        board_options.fix_base_link = True
        board_options.thickness = 0.0  # default = 0.02
        board_options.armature = 0.0  # default = 0.0
        board_options.use_physx_armature = True
        board_options.linear_damping = 0.0  # default = 0.0
        board_options.max_linear_velocity = 1000.0  # default = 1000.0
        board_options.angular_damping = 0.0  # default = 0.5
        board_options.max_angular_velocity = 64.0  # default = 64.0
        board_options.disable_gravity = False
        board_options.enable_gyroscopic_forces = True
        board_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        board_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            board_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        part_assets = []
        board_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_pcb[subassembly])
            part_file = self.asset_info_pcb[subassembly][components[0]]['urdf_path'] + '.urdf'
            board_file = self.asset_info_pcb[subassembly][components[1]]['urdf_path'] + '.urdf'
            part_options.density = self.cfg_env.env.part_board_density
            board_options.density = self.cfg_env.env.part_board_density
            part_asset = self.gym.load_asset(self.sim, urdf_root, part_file, part_options)
            board_asset = self.gym.load_asset(self.sim, urdf_root, board_file, board_options)
            part_assets.append(part_asset)
            board_assets.append(board_asset)

        return part_assets, board_assets

    def _create_actors(self, lower, upper, num_per_row, franka_asset, part_assets, board_assets, table_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.part_handles = []
        self.board_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.part_actor_ids_sim = []  # within-sim indices
        self.board_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        # self.part_heights = []
        # self.part_widths_max = []
        # self.board_widths = []
        # self.board_head_heights = []
        # self.board_shank_lengths = []
        # self.thread_pitches = []

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs,
                                                      0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_pcb[subassembly])

            part_pose = gymapi.Transform()
            part_pose.p.x = 0.0
            part_pose.p.y = self.cfg_env.env.part_lateral_offset
            part_pose.p.z = self.cfg_base.env.table_height
            part_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            part_handle = self.gym.create_actor(env_ptr, part_assets[j], part_pose, 'part', i, 0, 0)
            self.part_actor_ids_sim.append(actor_count)
            actor_count += 1

            # part_height = self.asset_info_pcb[subassembly][components[0]]['height']
            # part_width_max = self.asset_info_pcb[subassembly][components[0]]['width_max']
            # self.part_heights.append(part_height)
            # self.part_widths_max.append(part_width_max)

            board_pose = gymapi.Transform()
            board_pose.p.x = 0.0
            board_pose.p.y = 0.0
            board_pose.p.z = self.cfg_base.env.table_height
            board_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            board_handle = self.gym.create_actor(env_ptr, board_assets[j], board_pose, 'board', i, 0, 0)
            self.board_actor_ids_sim.append(actor_count)
            actor_count += 1

            # board_width = self.asset_info_pcb[subassembly][components[1]]['width']
            # board_head_height = self.asset_info_pcb[subassembly][components[1]]['head_height']
            # board_shank_length = self.asset_info_pcb[subassembly][components[1]]['shank_length']
            # self.board_widths.append(board_width)
            # self.board_head_heights.append(board_head_height)
            # self.board_shank_lengths.append(board_shank_length)

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            part_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, part_handle)
            part_shape_props[0].friction = self.cfg_env.env.part_board_friction
            part_shape_props[0].rolling_friction = 0.0  # default = 0.0
            part_shape_props[0].torsion_friction = 0.0  # default = 0.0
            part_shape_props[0].restitution = 0.0  # default = 0.0
            part_shape_props[0].compliance = 0.0  # default = 0.0
            part_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, part_handle, part_shape_props)

            board_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, board_handle)
            board_shape_props[0].friction = self.cfg_env.env.part_board_friction
            board_shape_props[0].rolling_friction = 0.0  # default = 0.0
            board_shape_props[0].torsion_friction = 0.0  # default = 0.0
            board_shape_props[0].restitution = 0.0  # default = 0.0
            board_shape_props[0].compliance = 0.0  # default = 0.0
            board_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, board_handle, board_shape_props)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.part_handles.append(part_handle)
            self.board_handles.append(board_handle)
            self.table_handles.append(table_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.part_actor_ids_sim = torch.tensor(self.part_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.board_actor_ids_sim = torch.tensor(self.board_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.part_actor_id_env = self.gym.find_actor_index(env_ptr, 'part', gymapi.DOMAIN_ENV)
        self.board_actor_id_env = self.gym.find_actor_index(env_ptr, 'board', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.part_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, part_handle, 'part', gymapi.DOMAIN_ENV)
        self.board_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, board_handle, 'board', gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)


    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # Note: the transforms for part and board are set up s.t. they are at the approximate CoM
        self.part_pos = self.root_pos[:, self.part_actor_id_env, 0:3]
        self.part_quat = self.root_quat[:, self.part_actor_id_env, 0:4]
        self.part_linvel = self.root_linvel[:, self.part_actor_id_env, 0:3]
        self.part_angvel = self.root_angvel[:, self.part_actor_id_env, 0:3]

        self.board_pos = self.root_pos[:, self.board_actor_id_env, 0:3]
        self.board_quat = self.root_quat[:, self.board_actor_id_env, 0:4]

        self.part_force = self.contact_force[:, self.part_body_id_env, 0:3]

        self.board_force = self.contact_force[:, self.board_body_id_env, 0:3]


    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.
        ...