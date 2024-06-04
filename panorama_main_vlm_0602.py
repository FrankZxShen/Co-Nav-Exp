from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import json
import sys
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import quaternion
import pickle
import io
import re

from skimage import measure
import skimage.morphology
from PIL import Image

import math
import cv2
import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

# from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
# from constants import coco_categories, color_palette, category_to_id
from agents.panorama_vlm_agents import LLM_Agent
# from agents.llm_agents import LLM_Agent
from constants import color_palette, coco_categories, hm3d_category, category_to_id, object_category

from envs.habitat.multi_agent_env_vlm import Multi_Agent_Env
# from envs.habitat.multi_agent_env import Multi_Agent_Env

# from src.habitat import (
#     make_simple_cfg,
#     pos_normal_to_habitat,
#     pos_habitat_to_normal,
#     pose_habitat_to_normal,
#     pose_normal_to_tsdf,
# )
# from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM, CogVLM2
from src.SystemPrompt import form_prompt_for_VLM, Perception_weight_decision
# from src.tsdf import TSDFPlanner
import utils.pose as pu

import utils.visualization as vu

from arguments import get_args

from detect import Detect

@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:1" if args.cuda else "cpu")

    # logging.info(f"stride:{stride}")
    # logging.info(f"names:{names}")
    # logging.info(f"pt:{pt}")


    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()

    config_env.TASK.POSSIBLE_ACTIONS = config_env.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    config_env.freeze()

    # Load VLM
    # vlm = VLM(args.vlm_model_id, args.hf_token, device)
    base_url = args.base_url 
    cogvlm2 = CogVLM2(base_url) 
    # Load Yolo
    yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)

    # print(config_env)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # exit(0)


    # TSDF需要init_pts（初始三维坐标）、pathfinder（habitat_sim）
    # sim_cfg = config_env
    # simulator = habitat_sim.Simulator(sim_cfg)
    # print(simulator.pathfinder)
    
    env = Multi_Agent_Env(config_env=config_env)

    num_episodes = env.number_of_episodes

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS

    agent = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, i, device))

    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'output.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    # logging.info(args)
    # ------------------------------------------------------------------

    # print("num_episodes:",num_episodes)# 1000

    agg_metrics: Dict = defaultdict(float)

    count_episodes = 0
    count_step = 0
    
    goal_points = []
    log_start = time.time()
    last_decision = []
    total_usage = []

    # logging.info(f"num agents: {num_agents}")

    # 存在巨大漏洞：设置问题->FMM反向导航

    while count_episodes < num_episodes:
        observations = env.reset()
        for i in range(num_agents):
            agent[i].reset()

        while not env.episode_over:
            if agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0:
                goal_points.clear()
                all_rgb = [] # 用于保存每个智能体的
                all_objs = [] # 记录每个时间步每个智能体的目标检测信息
                all_VLM_Pred = [] # 记录每个时间步中每个智能体的VLM预测结果
                all_VLM_PR = [] # 记录每个时间步中每个智能体的PR分数
                start = time.time()
                count_rotating = 0

                for j in range(num_agents):
                    agent[i].EXIT = False
                    agent[i].Perception_PR = 0
                    agent[i].Max_Perception_Angle = 360
                    agent[i].count_rerotation = 0
                
                #### 这个是画图测试用的，现在是没有决策策略的
                # while agent[0].l_step < 25:
                #     goal_points = [[275, 288]]

                    # action = [0]
                    # pose_pred = []
                    # full_map = []
                    # visited_vis = []
                    # for i in range(num_agents):
                    #     agent[i].mapping(observations[i])
                    #     local_map1, _ = torch.max(agent[i].local_map.unsqueeze(0), 0)
                        # full_map.append(agent[i].local_map)
                        # visited_vis.append(agent[i].visited_vis)
                        # start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                        # gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                        # pos = (
                        #     (start_x * 100. / args.map_resolution - gy1)
                        #     * 480 / agent[i].visited_vis.shape[0],
                        #     (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                        #     * 480 / agent[i].visited_vis.shape[1],
                        #     np.deg2rad(-start_o)
                        # )
                    #     pose_pred.append(pos)
                    
                    # # full_map2 = torch.cat((full_map[0].unsqueeze(0), full_map[1].unsqueeze(0)), 0)
                    # full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
                    # # logging.info(f"full_map2: {full_map2.shape}") #[x,20,480,480]

                    # full_map_pred, _ = torch.max(full_map2, 0)
                    # Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred)


                    # for i in range(num_agents):
                    #     action[i] = agent[i].act(goal_points[i], False)
                    
                    # if len(target_point_map) > 0:
                    #     Frontiers_dict = {}
                    #     for j in range(len(target_point_map)):
                    #         Frontiers_dict['frontier_' + str(j)] = f"<centroid: {target_point_map[j][0], target_point_map[j][1]}, number: {Frontier_list[j]}>"
                    # logging.info(f'=====> Exit Frontier: {Frontiers_dict}')
                #     observations = env.step(action)
                #     if args.visualize or args.print_images: 
                #         Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                #                     agent[0].goal_id, visited_vis, target_edge_map, Frontiers_dict, goal_points)
                # exit(0)
                for ags in range(360 // args.turn_angle): # 旋转
                    action = []
                    returnsteps = []
                    for j in range(num_agents):
                        returnsteps.append(0)
                        action.append(0)
                    full_map = []
                    visited_vis = []
                    pose_pred = []
                    agent_objs = {} # 记录单个时间步内每个智能体的目标检测信息
                    agents_VLM_Rel = {} # 记录单个时间步内（每个角度）每个智能体的VLM预测分数
                    agents_VLM_Pred = {} # 记录单个时间步内（每个角度）每个智能体的VLM预测结果
                    agents_VLM_PR = {} # 记录单个时间步内（每个角度）每个智能体的VLM PR分数
                    for i in range(num_agents):
                        agent[i].mapping(observations[i])
                        local_map1, _ = torch.max(agent[i].local_map.unsqueeze(0), 0)
                        full_map.append(agent[i].local_map)
                        visited_vis.append(agent[i].visited_vis)
                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                        pos = (
                            (start_x * 100. / args.map_resolution - gy1)
                            * 480 / agent[i].visited_vis.shape[0],
                            (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                            * 480 / agent[i].visited_vis.shape[1],
                            np.deg2rad(-start_o)
                        )
                        pose_pred.append(pos)

                        if agent[i].Max_Perception_Angle != 360 or agent[i].Perception_PR != 0:
                            continue
                        rgb = observations[i]['rgb'].astype(np.uint8)
                        all_rgb.append(rgb)
                        # rgb_panorama.append(rgb)
                        # save panorama map
                        # plt.imsave(
                        #     os.path.join("{}/dump/agent_{}_{}_obs.png".format(args.dump_location, i, count_rotating)), 
                        #     rgb,
                        # )
                        goal_name = agent[i].goal_name

                        # 修改为多智能体版本
                        agent_objs[f"agent_{i}"] = yolo.run(rgb) # 记录一个时间步内每个智能体的目标检测信息
                        

                        agents_seg_list = Objects_Extract(local_map1, args.use_sam)

                        # np.set_printoptions(threshold=np.inf)
                        # logging.info(f"explorable_map.shape:{explorable_map}") #[480, 480]
                        # logging.info(f"local_map1.shape:{local_map1.shape}") #[20,480,480]


                        # logging.info(f"agent_objs:{agent_objs}")
                        # logging.info(f"agents_seg_list:{agents_seg_list}")

                        VLM_Perception_Prompt = form_prompt_for_VLM(goal_name, agent_objs[f'agent_{i}'], agents_seg_list)

                        # logging.info(f"VLM_Perception_Prompt:{VLM_Perception_Prompt}")
                        # ------------------------------------------------------------------
                        ##### COT
                        # ------------------------------------------------------------------
                        # User_Prompt1 = "You are a robot that is exploring an indoor environment. Please describe the scene you currently see."
                        # _, cot_pred1 = cogvlm2.simple_image_chat(User_Prompt=User_Prompt1, img=rgb)

                        # Perception_Rel, Perception_Pred = cogvlm2.COT2(User_Prompt1=User_Prompt1, User_Prompt2=VLM_Perception_Prompt, \
                        #                                                cot_pred1=cot_pred1, return_string_probabilities="[Yes, No]", \
                        #                                                 img=rgb)
                        Perception_Rel, Perception_Pred = cogvlm2.simple_image_chat(User_Prompt=VLM_Perception_Prompt, 
                                                                                    return_string_probabilities="[Yes, No]", img=rgb)
                        Perception_Rel = np.array(Perception_Rel)
                        
                        Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        logging.info(f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionRel: {Perception_Rel}")
                        logging.info(f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPred: {Perception_Pred}")
                        logging.info(f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPR: {Perception_PR}")


                        agents_VLM_Rel[f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionRel"] = Perception_Rel
                        agents_VLM_Pred[f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPred"] = Perception_Pred
                        agents_VLM_PR[f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPR"] = Perception_PR

                    ##### Making panaorama map
                    # ------------------------------------------------------------------
                    # img_width = rgb_panorama[0].shape[1]
                    # # 创建一个空的全景图
                    # panorama = np.zeros((rgb_panorama[0].shape[0], img_width * 12, 3), dtype=np.uint8)

                    # # 将图片拼接到全景图中
                    # for i in range(12):
                    #     panorama[:, i*img_width:(i+1)*img_width] = rgb_panorama[i]

                    # 保存全景图
                    # cv2.imwrite(f'{args.dump_location}/dump/agent_0_{agent[0].l_step}_obs_panorama.jpg', panorama)
                        
                    all_objs.append(agent_objs) 
                    all_VLM_Pred.append(agents_VLM_Pred)
                    all_VLM_PR.append(agents_VLM_PR)
                    

                    # full_map2 = torch.cat((full_map[0].unsqueeze(0), full_map[1].unsqueeze(0)), 0)
                    full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
                    # full_map2 = full_map[0].unsqueeze(0)
                    # logging.info(f"full_map2: {full_map2.shape}") #[x,20,480,480]

                    full_map_pred, _ = torch.max(full_map2, 0)
                    Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred)

                    # ------------------------------------------------------------------
                    ##### Vote regions 退出旋转后语义无法更新，待处理
                    # ------------------------------------------------------------------
                    # explorable_map = get_explorable_areas(full_map_pred, count_rotating)
                    # for i in range(num_agents):
                    #     if count_rotating == 0:
                    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            # local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
                            # explo_area_map = np.zeros_like(local_ob_map)
                            # agent[i].explo_area_map = explo_area_map
                        #     color_map = np.zeros((explo_area_map.shape[0], explo_area_map.shape[1], 3), dtype=np.uint8)
                        #     agent[i].color_map = color_map
                        # else:
                        #     explo_area_map = agent[i].explo_area_map
                        #     color_map = agent[i].color_map
                        # explorable_map_cur, color_map_cur = ExtractExplorableAreas(
                            # full_map_pred, explo_area_map, \
                            # all_VLM_PR[ags][f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPR"], \
                            # all_VLM_PR[ags-1][f"Agent_{i}-Angle_{(ags-1)*args.turn_angle}-VLM_PerceptionPR"] if count_rotating!=0 else None, \
                        #     color_map, count_rotating,
                        # )
                        # agent[i].explo_area_map = explorable_map_cur
                        # agent[i].color_map = color_map_cur

                    # ------------------------------------------------------------------


                        # np.savetxt('array.txt', agent[i].explo_area_map, fmt='%.2f', delimiter=' ')

                    # logging.info(f"object_list:{Objects_Extract(full_map_pred)}")
                    # logging.info(f"target_edge_map:{target_edge_map}")

                    # exit(0)

                    # ------------------------------------------------------------------
                    ##### #判断是否旋转结束 设置单个时间步 每个智能体的状态：继续旋转/本地策略/回转
                    # ------------------------------------------------------------------
                    for i in range(num_agents):
                        if agent[i].Max_Perception_Angle != 360 or agent[i].Perception_PR != 0:
                            continue
                        if all_VLM_PR[ags][f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPR"][0] > 0.8 or (ags+1)*args.turn_angle >= 360:
                            agent[i].EXIT = True
                            # logging.info(f"Agent_{i} EXIT!")
                            # logging.info(f"all_VLM_PR:{all_VLM_PR}")

                            #### 所有置信度过低 360
                            if (ags+1)*args.turn_angle >= 360: 
                                logging.info(f"Agent_{i} 360 EXIT!!!")
                                max_value = float('-inf')  #初始化最大值为负无穷大
                                max_angle = None  #初始化最大值对应的角度为None
                                for item in all_VLM_PR:
                                    for key, value in item.items():
                                        agentss, angle, _ = key.split('-')
                                        if agentss == f'Agent_{i}':
                                            if value[0] > max_value:
                                                max_value = value[0]
                                                max_angle = int(angle.split('_')[1])
                                
                                agent[i].Max_Perception_Angle = max_angle
                                agent[i].Perception_PR = max_value
                            
                            #### 置信度达标
                            else:
                                agent[i].Perception_PR = all_VLM_PR[ags][f"Agent_{i}-Angle_{(ags)*args.turn_angle}-VLM_PerceptionPR"][0]
                    
                    ##### 没有达到退出条件，继续转
                    if all_agents_exit_false(agent):
                        goal_points.clear()
                        for i in range(num_agents):
                            goal_points.append([9999,9999])
                            action[i] = agent[i].act(goal_points[i], True)
                        

                    ##### 全部达到退出条件，退出
                    elif all_agents_exit_true(agent):
                        goal_points.clear()
                        break
                    ##### 部分达到退出条件，先执行退出的，再回转没有退出的
                    else:
                        goal_points.clear()
                        for i in range(num_agents):
                            # if agent[i].Max_Perception_Angle != 360 or agent[i].Perception_PR != 0:
                            #     continue
                            #### 达到退出条件->不用回转->刚好PR最高的是退出点
                            #### 达到退出条件->不用回转->当前PR大于0.8
                            if agent[i].Perception_PR > 0.8 or agent[i].Max_Perception_Angle == ags*args.turn_angle:
                                # ------------------------------------------------------------------
                                ##### History TOT 
                                # ------------------------------------------------------------------
                                if len(target_point_map) > 0:
                                    Frontiers_dict = {}
                                    for j in range(len(target_point_map)):
                                        Frontiers_dict['frontier_' + str(j)] = f"<centroid: {target_point_map[j][0], target_point_map[j][1]}, number: {Frontier_list[j]}>"
                                    logging.info(f'=====> Exit Frontier: {Frontiers_dict}')
                                    for j in range(num_agents):
                                        # Agent States
                                        logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Exit: {agent[j].EXIT}; Perception_PR:{agent[j].Perception_PR}; Max_Perception_Angle: {agent[j].Max_Perception_Angle}; count_rerotation: {agent[j].count_rerotation}')
            
                                else:
                                    for j in range(num_agents):
                                        actions = np.random.rand(1, 2).squeeze()*(target_edge_map.shape[0] - 1)

                                        goal_points.append([int(actions[0]), int(actions[1])])
                                
                                #### 判断是否回转结束
                                for inner_i in range(num_agents):
                                    if agent[inner_i].Max_Perception_Angle != 360 and agent[inner_i].count_rerotation < returnsteps[inner_i]:
                                        action[inner_i] = 2
                                        agent[inner_i].count_rerotation += 1
                                    else:
                                        
                                        action[inner_i] = 1 ###TEST

                                ##### 目标：获取当前所有agent的action

                            #### 达到退出条件->需要回转
                            elif agent[i].Max_Perception_Angle < ags*args.turn_angle:
                                # for turnsteps in range((ags*args.turn_angle - agent[i].Max_Perception_Angle) // args.turn_angle): #智能体i转到对应角度
                                    ##### 退出旋转的智能体的操作
                                    # turnedagents = i
                                    # for allagents in range(num_agents):
                                    #     action[allagents] = agent[allagents].act(goal_points[allagents], True)
                                # 计数，转多久？？？
                                returnsteps[i] = (ags*args.turn_angle - agent[i].Max_Perception_Angle) // args.turn_angle
                                action[i] = 2 # Left
                                agent[i].count_rerotation += 1
                                if agent[i].count_rerotation == returnsteps[i]:
                                    agent[i].Max_Perception_Angle = 360
                                

                            #### 没有达到退出条件->继续旋转
                            else:
                                action[i] = 3

                                    
                    observations = env.step(action)

                    count_rotating += 1

            # ------------------------------------------------------------------
            ##### History TOT -> All Agents EXIT(Not All Rerotate)
            # ------------------------------------------------------------------
            #### 判断是否回转结束
            # for i in range(num_agents):
            #     if agent[i].Max_Perception_Angle != 360 and agent[i].count_rerotation < returnsteps[i]:
            #         action[i] = 2
            #         agent[i].count_rerotation += 1
                    
            if len(target_point_map) > 0:
                Frontiers_dict = {}
                for j in range(len(target_point_map)):
                    Frontiers_dict['frontier_' + str(j)] = f"<centroid: {target_point_map[j][0], target_point_map[j][1]}, number: {Frontier_list[j]}>"
                logging.info(f'=====> Exit Frontier: {Frontiers_dict}')
                for j in range(num_agents):
                    # Agent States
                    logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Exit: {agent[j].EXIT}; Perception_PR:{agent[j].Perception_PR}; Max_Perception_Angle: {agent[j].Max_Perception_Angle}; count_rerotation: {agent[j].count_rerotation}')
            
            else:
                for j in range(num_agents):
                    actions = np.random.rand(1, 2).squeeze()*(target_edge_map.shape[0] - 1)

                    goal_points.append([int(actions[0]), int(actions[1])])
            
            #### 判断是否回转结束
            for j in range(num_agents):
                if agent[j].Max_Perception_Angle != 360 and agent[j].count_rerotation < returnsteps[j]:
                    action[j] = 2
                    agent[j].count_rerotation += 1
            
            #### 这个是画图测试用的，现在是没有决策策略的
            while agent[0].l_step < 25:
                goal_points = [[275, 288]]
                for i in range(num_agents):
                    action[i] = agent[i].act(goal_points[i], False)
                
                observations = env.step(action)

                action = [0]
                pose_pred = []
                full_map = []
                visited_vis = []
                for i in range(num_agents):
                    agent[i].mapping(observations[i])
                    # local_map1, _ = torch.max(agent[i].local_map.unsqueeze(0), 0)
                    full_map.append(agent[i].local_map)
                    visited_vis.append(agent[i].visited_vis)
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                    pos = (
                        (start_x * 100. / args.map_resolution - gy1)
                        * 480 / agent[i].visited_vis.shape[0],
                        (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                        * 480 / agent[i].visited_vis.shape[1],
                        np.deg2rad(-start_o)
                    )
                    pose_pred.append(pos)
                
                # full_map2 = torch.cat((full_map[0].unsqueeze(0), full_map[1].unsqueeze(0)), 0)
                full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
                # full_map2 = full_map[0].unsqueeze(0)
                # logging.info(f"full_map2: {full_map2.shape}") #[x,20,480,480]

                full_map_pred, _ = torch.max(full_map2, 0)
                Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred)
                

                if args.visualize or args.print_images: 
                    Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                agent[0].goal_id, visited_vis, target_edge_map, Frontiers_dict, goal_points)
            
            exit(0)

            action = [1] ###TEST
            logging.info(f"Begin History TOT -> All Agents EXIT")
            observations = env.step(action)

            #### 更新环境
            action.clear()
            for j in range(num_agents):
                action.append(0)
            pose_pred = []
            full_map = []
            visited_vis = []
            for inner_i in range(num_agents):
                agent[inner_i].mapping(observations[inner_i])
                local_map1, _ = torch.max(agent[inner_i].local_map.unsqueeze(0), 0)
                full_map.append(agent[inner_i].local_map)
                visited_vis.append(agent[inner_i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[inner_i].planner_pose_inputs

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[inner_i].visited_vis.shape[0],
                    (agent[inner_i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[inner_i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)

            # full_map2 = torch.cat((full_map[0].unsqueeze(0), full_map[1].unsqueeze(0)), 0)
            full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
            # full_map2 = full_map[0].unsqueeze(0)
            # logging.info(f"full_map2: {full_map2.shape}") #[x,20,480,480]

            full_map_pred, _ = torch.max(full_map2, 0)

                        



                # if count_rotating == 2:
                #     exit(0)
                # ------------------------------------------------------------------

                    # Debug: 不画图
                    # if args.visualize or args.print_images: 
                    #     Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                    #             agent[0].goal_id, visited_vis, target_edge_map, goal_points)
                
                # exit(0)
            

                                
            


            # logging.info(f"full_map_pred.shape: {full_map_pred.shape}") # [20,480,480] HM-3D
            exit(0)

# def get_explorable_areas(full_map_pred, count):
#     # 获取可探索区域
#     explorable_areas = cv2.inRange(full_map_pred[1].cpu().numpy(), 0.1, 1)
    
    # # 使用形态学闭运算填充小的空洞
    # kernel = np.ones((5, 5), dtype=np.uint8)
    # explorable_areas = cv2.morphologyEx(explorable_areas, cv2.MORPH_CLOSE, kernel)
    
    # # 创建一个新的地图，用于存储可探索区域
    # explorable_map = np.zeros_like(full_map_pred[0].cpu().numpy())
    
    # # 在新地图上标记可探索区域
    # explorable_map[explorable_areas == 255] = 1

    # color_map = np.zeros((explorable_map.shape[0], explorable_map.shape[1], 3), dtype=np.uint8)
    
    # # 将可探索区域标记为绿色
    # color_map[explorable_map == 1] = [0, 255, 0]  # GR 值
    
    # # 显示彩色图像
    # fn = f'Vis-explore_{count}.png'
    # cv2.imwrite(fn, color_map)
    
    # return explorable_map

def Objects_Extract(full_map_pred, use_sam):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) > 30:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                if use_sam:
                    Object_list[object_category[i]] = Single_object_list
                else:
                    Object_list[hm3d_category[i]] = Single_object_list
    return Object_list

def all_agents_exit_false(agents):
    for agent in agents:
        if agent.EXIT:
            return False
    return True

def all_agents_exit_true(agents):
    for agent in agents:
        if not agent.EXIT:
            return False
    return True

def ExtractExplorableAreas(full_map_pred, explo_area_map, VLM_PR, VLM_PR_last, color_map, count):
    PR = VLM_PR[0]

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_w = full_map_pred.shape[1]

    # local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(), 0.1, 1)

    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
    explo_area_map_cur = np.zeros_like(local_ob_map)

    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) > 4: # 示例中将面积极小的区域排除
                cv2.drawContours(explo_area_map_cur, [contour], -1, PR, -1) # 填充内部为PR

    # 清除边界部分
    explo_area_map_cur[0:2, 0:full_w] = 0
    explo_area_map_cur[full_w-2:full_w, 0:full_w] = 0
    explo_area_map_cur[0:full_w, 0:2] = 0
    explo_area_map_cur[0:full_w, full_w-2:full_w] = 0

    if VLM_PR_last:
        # mask = np.logical_and(explo_area_map_cur != PR, explo_area_map == VLM_PR_last[0])
        coords = np.where(explo_area_map != 0)
        # PR_coords = list(zip(coords[0], coords[1]))
        explo_area_map_cur[coords] = explo_area_map[coords]

    
    # 将可探索区域标记为当前颜色
    intensity = int(PR * 100 * 2.55)
    intensity = max(0, min(intensity, 100))
    color_map[np.where(explo_area_map_cur == PR)] = [intensity, intensity, intensity]  #  RGB 值

    lipped_map = cv2.flip(color_map, 0)
    color_map__ = Image.fromarray(lipped_map)
    color_map__ = color_map__.convert("RGB")

    
    # 显示彩色图像
    # fn = f'Vis-explore2_{count}.png'
    # # cv2.imwrite(fn, color_map__)
    # color_map__.save(fn)

    return explo_area_map_cur, color_map

def Frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
    Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
            if i == 3:
                break
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Wall_lines, Goal_area_list, Goal_edge, Goal_point

# 画出所有的Frontier
def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, Frontiers_dict, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    no_cat_mask = sem_map == len(object_category) + 4
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    # vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)
    vis_image = vu.init_multi_vis_image(object_category[goal_name], color)
    

    vis_image[50:530, 15:495] = sem_map_vis

    color_black = (0,0,0)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    for keys, value in Frontiers_dict.items():
        match = re.match(pattern, value)
        if match:
            centroid_x = int(match.group(1)[1:])
            centroid_y = int(match.group(2)[:-1])
            number = float(match.group(3))
            # print(f"Centroid: ({centroid_x}, {centroid_y})")
            # print(f"Number: {number}")
            cv2.circle(vis_image, (centroid_x+28, centroid_y-35), 5, color_black, -1)
            label = f"{alpha[alpha0]} {number}"
            alpha0 += 1
            cv2.putText(vis_image, label, (centroid_x+28+5, centroid_y-35+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1)

    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)
    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Step-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        # print(fn)
        cv2.imwrite(fn, vis_image)    
    

if __name__ == "__main__":
    main()
