import re

# 废弃
Meta_Agent_Preprocess_Prompt = '''
You are a knowledgeable and skilled expert in indoor navigation planning. You are planning the navigation of multiple robots in an indoor environment, equipped with three specialized modules to help analyze and understand the connections between current semantic maps and navigation objectives:
(1) Scene image description module: This module records the current and historical navigation scene image description information, and can query any content related to the historical navigation image content. This module is especially useful when you don't know the scene information in the semantic map.
(2) Scene object detection module: This module identifies and locates the objects in the image, and provides the boundaries and confidence of these objects. This module is especially useful when you don't know the information about objects in a semantic map.
(3) Scene exploration and analysis module: This module records the exploration possibility of current and historical navigation scenes. This module is especially useful when you are not clear about the exploration possibilities of semantic graphs.
Your task is: based on the capabilities of each module, assign specific tasks as needed to gather the additional information needed to accurately answer the question.

Your output format is SIMPLY as follows **WITHOUT ANY OTHER WORDS**:

Tasks of the module:
(1) Scene image description module: <If necessary, please return Yes. Otherwise, return No.>
(2) Scene object detection module: <If necessary, please return Yes. Otherwise, return No.>
(3) Scenario exploration analysis module: <If required, please raise any specific questions you have about the need for further exploration of the agent that require more in-depth visual analysis. Otherwise return to No.>

Make sure your answers fit into this format, using available modules or appropriate direct analysis to systematically address the question.

**Here are some RIGHT EXAMPLES:**
Example 1:
(1) Scene image description module: No
(2) Scene object detection module: No
(3) Scenario exploration analysis module: No

Example 2:
(1) Scene image description module: Yes
(2) Scene object detection module: No
(3) Scenario exploration analysis module: No

Example 3:
(1) Scene image description module: No
(2) Scene object detection module: Yes
(3) Scenario exploration analysis module: No

Example 4:
(1) Scene image description module: No
(2) Scene object detection module: No
(3) Scenario exploration analysis module: Yes

Example 5:
(1) Scene image description module: Yes
(2) Scene object detection module: Yes
(3) Scenario exploration analysis module: No

Example 6:
(1) Scene image description module: Yes
(2) Scene object detection module: No
(3) Scenario exploration analysis module: Yes

Example 7:
(1) Scene image description module: No
(2) Scene object detection module: Yes
(3) Scenario exploration analysis module: Yes

Example 8:
(1) Scene image description module: No
(2) Scene object detection module: Yes
(3) Scenario exploration analysis module: Yes

Example 9:
(1) Scene image description module: Yes
(2) Scene object detection module: Yes
(3) Scenario exploration analysis module: Yes

'''
# 废弃
Module_Decision_Prompt = '''
You are an advanced semantic understanding agent, and you need to focus on the semantic information of the scene image on the right. Your task is: 

'''
# 废弃
Single_Agent_Decision_Prompt_mix = '''
Based on the above information, you will now explore in this indoor environment. You have access to an image of the current timestep's scene and a top-down semantic map. 

The right part of the image is the scene image of the current timestep. 

The left part of the image is the corresponding top-down semantic map. The black dots and corresponding uppercase letters represent the Frontiers awaiting your exploration: 
{FRONTIERS_RESULTS}
The arrow is your location: [[Coordinates: {CUR_LOCATION}].
The line behind you represent your historical movement path. 

Your goal is to find the {TARGET}. You need to comprehensively consider the relevance of the scene image and the top-down semantic map, and choose the navigation frontier for the next timestep based on their relationship with your navigation goal. You need to select the Frontier corresponding to the black capital letter [A, B, C, D] and answer the reason for your choice.

Your output format is as follows: 

My choice: 
<You need to select the Frontier corresponding to the black capital letter [A, B, C, D]>

The reasons why I choose this Frontier: 
<(1) Consider the pixel counting area size. You need to consider the front pixel count in relation to the navigation target. Usually larger exploration objects such as beds and sofas have larger pixel counts. 
(2) Consider the proximity and accessibility of the frontier. You need to consider the proximity and accessibility of the front and your location. Frontiers that are closer and without barriers tend to have a higher exploration priority. 
(3) Consider the objects in the scene. Use your knowledge of the location of typical objects (e.g., the bed in your bedroom, the TV in your living room) to assess the likelihood of finding the target object.
(4) Consider possible potential rooms in the scene. Areas in the scene image may lead to different rooms, and navigation targets may be located in different rooms. For example, the appearance of a door in the image may indicate that the target object may be located in a potential room behind the door, which makes the corresponding frontier worth exploring.>
'''







### 感知
Perception_System_Prompt = """
You are a knowledgeable and skilled expert in indoor navigation planning. You are planning the navigation of multiple robots in an indoor environment. You can access a scene image and top-down semantic graph of the current time step of one of the robots.
The right part of the image is the scene image of the current timestep. 

The left part of the image is the corresponding top-down semantic map.

Given the following information format:

Target of navigation: format-<Name>

Scene object (object detection): format-<Name, Confidence Score>

You need to determine the reasonableness of the existence of the above scene objects based on the image information. Your goal is to determine if the scene image is worth exploring by the robot. Consider the following points when making your decision:
(1) Examine the relationship between the target object and the scene. If the object detection model predicts the presence of the target object with high probability (e.g., >85%), it indicates that the scene is worth exploring.
(2) Analyze the image and consider the context of the scene, such as ceilings, walls, floors, and windows. Use your knowledge of typical object locations (e.g., beds in bedrooms, TVs in living rooms) to assess the likelihood of finding the target object.
(3) Consider whether the object target the robot is trying to find is close to the image scene, and make sure that the judgment you output does not violate rule (1). For example, a bathtub is unlikely to be found in a bedroom. However, the presence of a door in the image might suggest that the target object could be located behind it, making the scene worth exploring.
(4) Disregard objects that are commonly found in multiple rooms, such as light switches and doors, as they do not provide strong evidence for the presence of the target object.
Your output should be a simple "[Yes, No]" statement indicating whether the scene is worth exploring based on the given criteria.

Now we begin:
"""
### F/N
FN_System_Prompt = '''
If the current scene is worth exploring, you should prioritize exploring frontier points. If the current scene is not worth exploring, you should consider revisiting historical observation points. Based on the current top-down semantic map of an indoor environment provided (The left part of the image), you will see various points marked as either 'frontier points' or 'historical observation points'. Frontier points represent unexplored areas that the robot has yet to navigate, while historical observation points signify areas the robot has previously explored or observed.
Given the following information format:
Frontier Points (The black dots and corresponding black uppercase letters on the left image):
format-<Name: [Coordinates: <semantic map coordinates>]>
Historical Observation Points (The green dots and corresponding green lowercase letters on the left image):
format-<Name: [Coordinates: <semantic map coordinates>]>
The Arrow (Your location): format-<semantic map coordinates>
Previous Movement: format-<semantic map coordinates>
You need to guide the robot for exploration purposes based on the relationship between different objects, the structure of the explored area, the robot's position, the proximity of the exploration point, and the direction of previous movement.
Would you recommend:
A) Explore a frontier point? If so, answer ONLY Yes.

B) Revisit a historical observation point? If so, answer ONLY No.

Now we begin:
Frontier Points (The black dots and corresponding black uppercase letters on the left image):
{FRONTIERS_RESULTS}
Historical Observation Points (The green dots and corresponding green lowercase letters on the left image):
{HISTORY_NODES}
The Arrow: {CUR_LOCATION}.
Previous Movement: {PRE}

Your Answer MUST in 'Yes', 'No' **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''

### 决策1
Single_Agent_Decision_Prompt_Frontier = '''
Your choice is 'Frontier Points'. So you will now explore the frontiers in this indoor environment.

The black dots and corresponding uppercase letters on the left image represent the Frontier Points awaiting your exploration: 
{FRONTIERS_RESULTS}
The Arrow is Your Location: {CUR_LOCATION}.
Previous Movement: {PRE}

Your goal is to find the {TARGET}. You need to consider the following factors:
(1) Consider the proximity and accessibility of the frontier. You need to consider the proximity and accessibility of the front and your previous movement. Frontiers that are closer and without barriers tend to have a higher exploration priority. 
(2) Consider the objects in the scene. Use your knowledge of the location of typical objects (e.g., the bed in your bedroom, the TV in your living room) to assess the likelihood of finding the target object.
(3) Minimize frequent switches between frontiers. You should maintain its exploration direction unless an efficient switch is evident.
You need to comprehensively consider the relevance of the scene image and the top-down semantic map, and choose the navigation frontier for the next timestep based on their relationship with your navigation goal. 

Your choice MUST in 'A', 'B', 'C', 'D' **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''
### 决策2
Single_Agent_Decision_Prompt_History = '''
Your choice is 'Historical Observation Points'. So you will now re-explore historical nodes in this indoor environment. 

The green dots and corresponding green lowercase letters on the left image represent the Historical Observation Points awaiting your exploration: 
{HISTORY_NODES}
where the confidence level following each green lowercase letter represents the exploration likelihood that this point was recorded. Point with higher confidence have higher exploration priority.
The Arrow is Your Location: {CUR_LOCATION}
Previous Movement: {PRE}

Your goal is to find the {TARGET}. You need to consider the following factors:
(1) Consider the proximity and accessibility of the points. You need to consider the proximity and accessibility of the front and your previous movement. Points that are closer and without barriers tend to have a higher exploration priority. 
(2) Consider the objects in the scene. Use your knowledge of the location of typical objects (e.g., the bed in your bedroom, the TV in your living room) to assess the likelihood of finding the target object.
(3) Minimize frequent switches between points. Use centroid for frontier selection. You should maintain its exploration direction unless an efficient switch is evident.
You need to comprehensively consider the relevance of the scene image and the top-down semantic map, and choose the navigation point for the next timestep based on their relationship with your navigation goal. 

Your choice must be in 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' and 26 other lowercase letters representing historical observation points  **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''







# 废弃
# Meta_Agent_Decision_Prompt2 = '''
# Based on the above decisions and analysis, please combine your knowledge to choose the navigation frontier for each robots.

# Your output format is as follows: 

# Agent0: <You need to select the Frontier corresponding to the black capital letter [A, B, C, D]>

# Agent1: <You need to select the Frontier corresponding to the black capital letter [A, B, C, D]>

# ...

# ''' 
# Example:

# Target of navigation: 'bed'

# Scene object (object detection): 
# bench: 0.45, <(186, 325,), (587, 479)>
# bed: 0.65, <(466, 448), (640, 480)>
# dining table: 0.77, <(125, 244), (201, 325)>
# chair: 0.77, <(65, 231), (146, 328)> 
# couch: 0.87, <(169, 247) (640, 475)>

# Scene object (semantic segmentation):
# sofa: <(294, 202), (267, 216), (284, 217), (288, 258), (307, 235)>
# table: <(200, 150), (230, 150), (230, 170), (200, 170)> 

# Are you confident that the robot is worth exploring in this scenario? Please answer Yes or No.

def form_prompt_for_PerceptionVLM(target, objs, yolo) -> str:

    object_detection = ''
    if yolo == 'yolov9':
        if len(objs) < 1:
            object_detection = 'No Detections'
        else:
            for item in objs:
                name, confidence, coords = item
                # coord_pairs = coords[0].split(',')
                # coord1 = coord_pairs[0], coord_pairs[1]
                # coord2 = coord_pairs[2], coord_pairs[3]
                detection = f"{name}: {confidence})>\n"
                object_detection += detection
    else:
        # print(objs)
        if len(objs) < 1:
            object_detection = 'No Detections'
        else:
            for name, confidence in objs.items():
                detection = f"<{name}: {confidence})>\n"
                object_detection += detection

    # semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
    #                         for j in range(len(value[i]))]) + f">"
    #                         for i in range(len(value))])
    #                         for key, value in agents_seg_list.items()]) + "\n"

    Perception_Template = """
Target of navigation: {TARGET}

Scene object (object detection): 
{OBJECT_DETECTION}
Are you confident that the robot is worth exploring in this scenario? You only need to answer "Yes" or "No". You don't need to add punctuation at the end.
"""
    User_Prompt = Perception_Template.format(
                        TARGET = target,
                        OBJECT_DETECTION = object_detection,
                    )
    User_Prompt = Perception_System_Prompt + User_Prompt



    return User_Prompt

def form_prompt_for_FN(pre_goal_point, Frontier_list, cur_location, History_nodes) -> str:

    def convert_entry(entry):
        # 从字符串中提取坐标
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}]'
    
    if all(num == 0 for num in pre_goal_point):
        pre_goal_point = 'No Movements'

    # 遍历字典，并使用转换函数
    Frontiers = []
    for i, key in enumerate(Frontier_list):
        entry = Frontier_list[key]
        converted = convert_entry(entry)
        Frontiers.append(f'{chr(65+i)}: {converted}') 
    
    Frontiers_results = ''
    for i in range(len(Frontiers)):
        Frontiers_results += Frontiers[i]
        Frontiers_results += '\n'

    if len(History_nodes) > 0:
        History = []
        for i in range(len(History_nodes)):
            History.append(f'{chr(ord("a")+i)}: [Coordinates: {History_nodes[i]}]') 
        His_results = ''
        for i in range(len(History)):
            His_results += History[i]
            His_results += '\n'
    else:
        His_results = 'No historical observation points'

    User_Prompt = FN_System_Prompt.format(
                        FRONTIERS_RESULTS = Frontiers_results,
                        HISTORY_NODES = His_results,
                        CUR_LOCATION = cur_location,
                        PRE = pre_goal_point, 
                    )


    return User_Prompt

def form_prompt_for_DecisionVLM_Frontier(pre_goal_point, target, cur_location, Frontier_list) -> str:

    def convert_entry(entry):
        # 从字符串中提取坐标和像素计数
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}]'

    if all(num == 0 for num in pre_goal_point):
        pre_goal_point = 'No Movements'

    # 遍历字典，并使用转换函数
    Frontiers = []
    for i, key in enumerate(Frontier_list):
        entry = Frontier_list[key]
        converted = convert_entry(entry)
        Frontiers.append(f'{chr(65+i)}: {converted}') 
    
    Frontiers_results = ''
    for i in range(len(Frontiers)):
        Frontiers_results += Frontiers[i]
        Frontiers_results += '\n'
    
    User_Prompt = Single_Agent_Decision_Prompt_Frontier.format(
        FRONTIERS_RESULTS = Frontiers_results,
        TARGET = target,
        CUR_LOCATION = cur_location,
        PRE = pre_goal_point, 
    )
    

    return User_Prompt

def form_prompt_for_DecisionVLM_History(pre_goal_point, target, cur_location, confidence, History_nodes) -> str:

    # def convert_entry(entry):
    #     # 从字符串中提取坐标和像素计数
    #     centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
    #     number_str = entry.split('number: ')[1][:-1]
    #     return f'[Coordinates: {centroid_str}]'
    
    if all(num == 0 for num in pre_goal_point):
        pre_goal_point = 'No Movements'
    # 遍历字典，并使用转换函数
    if len(History_nodes) > 0:
        History = []
        for i in range(len(History_nodes)):
            History.append(f'{chr(ord("a") + i)}: [Coordinates: {History_nodes[i]}; Confidence: {confidence[i]}]') 
        His_results = ''
        for i in range(len(History)):
            His_results += History[i]
            His_results += '\n'
    else:
        His_results = 'No historical observation points'
    
    User_Prompt = Single_Agent_Decision_Prompt_History.format(
        TARGET = target,
        CUR_LOCATION = cur_location,
        PRE = pre_goal_point, 
        HISTORY_NODES = His_results
    )
    

    return User_Prompt







def extract_scene_image_description_results(text):
    pattern = re.compile(r'Scene image description module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
def extract_scene_object_detection_results(text):
    pattern = re.compile(r'Scene object detection module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
def extract_scenario_exploration_analysis_results(text):
    pattern = re.compile(r'Scenario exploration analysis module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches

def form_prompt_for_DecisionVLM_MetaPreprocess() -> str:
    return Meta_Agent_Preprocess_Prompt

def form_prompt_for_Module_Decision(target, Preprocess_prompt, objs) -> str:
    Use_prompt = ''
    object_detection = ''
    for item in objs:
        name, confidence, coords = item
        coord_pairs = coords[0].split(',')
        coord1 = coord_pairs[0], coord_pairs[1]
        coord2 = coord_pairs[2], coord_pairs[3]
        detection = f"{name}: {confidence}, <({int(float(coord1[0]))}, {int(float(coord1[1]))}), ({int(float(coord2[0]))}, {int(float(coord2[1]))})>\n"
        object_detection += detection
    task_prompt_decription = '''
Provide a detailed description of the scene image on the right. Requirements include, but are not limited to, the main content of the image, the objects present, the environment (such as living room, bedroom, etc.), and possible potential indoor objects. Please note that the description should be detailed enough for someone who has not seen the image to have a clear visual perception of your explanation.

'''

    task_prompt_detection = '''
This is the object detection results for the right scene image in the format - [Name, Confidence Score, Coordinates of the Bounding Box (The Upper left and lower right corners) <(x1, y1), (x2, y2)>]:
{OBJECT_DETECTION}
Considering the existing object recognition results and the right scene image, list the names of all recognizable objects of the image. Be sure to rank these objects by predicted confidence or importance and briefly describe each object and its context.

'''

#     task_prompt_analysis = '''
# Please analyze the following elements based on the image: 1. the spatial layout and passable areas; 2. any potential obstacles; and 3. identify which areas may contain critical information or objects based on the current mission objectives {TARGET}. In conjunction with the above analysis, assess whether the areas in the image are worth exploring further and give your reasons for your recommendation.

# '''

    task_prompt_analysis = '''
Consider the spatial layout and passable areas, any potential obstacles based on the scene image on the right. Evaluate whether the areas in the image are worth exploring further in light of the current mission objectives {TARGET}, taking into account the above considerations, and give your reasons for your recommendation.    

'''
    des = ", ".join(extract_scene_image_description_results(Preprocess_prompt))
    det = ", ".join(extract_scene_object_detection_results(Preprocess_prompt))
    ana = ", ".join(extract_scenario_exploration_analysis_results(Preprocess_prompt))
    is_description = True if 'yes' in des.lower() else False
    is_detection = True if 'yes' in det.lower() else False
    is_analysis = True if 'yes' in ana.lower()  else False
    
    if is_description and not is_detection and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + end_prompt
    elif is_detection and not is_description and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image object detection: <Input your scene image object detection>
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection)
    elif is_analysis and not is_description and not is_detection:
        end_prompt = '''
Your output format is as follows:
(1) rationale for scene exploration: <Input your rationale for scene exploration>
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_analysis.format(TARGET=target)
    elif is_description and is_detection and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
(2) scene image object detection: <Input your scene image object detection>
'''     
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + end_prompt
    elif is_description and is_analysis and not is_detection:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
(2) rationale for scene exploration: <Input your rationale for scene exploration>
'''   
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    elif is_detection and not is_analysis and not is_description:    
        end_prompt = '''
Your output format is as follows:
(1) scene image object detection: <Input your scene image object detection>
(2) rationale for scene exploration: <Input your rationale for scene exploration>
'''   
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + \
            '(2) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    elif is_detection and is_analysis and is_description:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
(2) scene image object detection: <Input your scene image object detection>
(3) rationale for scene exploration: <Input your rationale for scene exploration>
'''     
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + \
                '(3) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    else:
        ### 跳过Module Prompt这一阶段
        Use_prompt = None
    
    return Use_prompt    




# def form_prompt_for_DecisionVLM_Meta() -> str:
#     return Meta_Agent_Decision_Prompt




def contains_yes_or_no(VLM_Pred: str) -> str:
    if "Yes" in VLM_Pred:
        return "Yes"
    elif "No" in VLM_Pred:
        return "No"
    else:
        return "Neither"

def Perception_weight_decision(VLM_Rel: list, VLM_Pred: str) -> str:
    b_decision = contains_yes_or_no(VLM_Pred)
    if b_decision == "Neither":
        return b_decision
    
    x, y = VLM_Rel
    
    if b_decision == "Yes":
        weighted_yes_prob = x
        weighted_no_prob = y * (1 - x)
    else:  # b_decision == "No"
        weighted_yes_prob = x * (1 - y)
        weighted_no_prob = y
    
    total_prob = weighted_yes_prob + weighted_no_prob
    if total_prob == 0:  # 避免划分零的问题
        return b_decision
    
    weighted_yes_prob /= total_prob
    weighted_no_prob /= total_prob
    
    return weighted_yes_prob, weighted_no_prob

def contains_decision(VLM_Pred: str) -> str:
    if "A" in VLM_Pred:
        return "A"
    elif "B" in VLM_Pred:
        return "B"
    elif "C" in VLM_Pred:
        return "C"
    elif "D" in VLM_Pred:
        return "D"
    else:
        return "Neither"

def Perception_weight_decision4(VLM_Rel: list, VLM_Pred: str) -> str:
    decision = contains_decision(VLM_Pred)
    if decision == "Neither":
        return decision
    
    # 假定 VLM_Rel 的長度為 4，分別對應四種決策的相對權重
    assert len(VLM_Rel) == 4, "VLM_Rel must contain weights for four decisions (A, B, C, D)."
    
    # 分配對應的權重
    weights = {
        "A": VLM_Rel[0],
        "B": VLM_Rel[1],
        "C": VLM_Rel[2],
        "D": VLM_Rel[3]
    }

    # 加权权重
    weights[decision] = weights[decision] * 4
    
    # 根據決策計算加權概率
    total_weight = sum(VLM_Rel)
    weighted_probs = {key: val/total_weight for key, val in weights.items()}
    
    # 確保總概率為 1
    if total_weight == 0:  # 避免除以零
        return "Invalid weights provided."
    
    return weighted_probs["A"],weighted_probs["B"],weighted_probs["C"],weighted_probs["D"]


def contains_decision26(VLM_Pred: str) -> str:
    for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
        if char in VLM_Pred:
            return char
    return "Neither"

def Perception_weight_decision26(VLM_Rel: list, VLM_Pred: str) -> dict:
    decision = contains_decision26(VLM_Pred)
    if decision == "Neither":
        return {"Neither": 1.0}
    
    assert len(VLM_Rel) == 26, "VLM_Rel must contain weights for 26 decisions (1-26)."
    
    weights = {chr(ord("a")+i): VLM_Rel[i] for i in range(26)}  # Assign weights to each number
    
    # 加权权重
    weights[decision] = weights[decision] * 26

    total_weight = sum(VLM_Rel)
    if total_weight == 0:
        return {"Invalid weights provided.": 1.0}

    weighted_probs = [val/total_weight for _, val in weights.items()]
    
    return weighted_probs
