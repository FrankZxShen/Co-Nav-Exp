import re
Perception_System_Prompt = """
You are a robot that is exploring an indoor environment. Given the following information format:

Target of navigation: format-[Name]

Scene object (object detection): shif

Scene object (semantic segmentation): format-[Name, Boundary Coordinates defined by vertices in clockwise order <(x1, y1), (x2, y2)...>]

You need to determine the reasonableness of the existence of the above scene objects based on the image information. Your goal is to determine if the scene image is worth exploring by the robot. Consider the following points when making your decision:
(1) Examine the relationship between the target object and the scene. If the object detection model predicts the presence of the target object with high probability (e.g., >85%), it indicates that the scene is worth exploring.
(2) Analyze the image and consider the context of the scene, such as ceilings, walls, floors, and windows. Use your knowledge of typical object locations (e.g., beds in bedrooms, TVs in living rooms) to assess the likelihood of finding the target object.
(3) Consider whether the object target the robot is trying to find is close to the image scene, and make sure that the judgment you output does not violate rule (1). For example, a bathtub is unlikely to be found in a bedroom. However, the presence of a door in the image might suggest that the target object could be located behind it, making the scene worth exploring.
(4) Disregard objects that are commonly found in multiple rooms, such as light switches and doors, as they do not provide strong evidence for the presence of the target object.
Your output should be a simple "[Yes, No]" statement indicating whether the scene is worth exploring based on the given criteria.

Now we begin:
"""

Meta_Agent_Preprocess_Prompt = '''
You are a knowledgeable and skilled expert in indoor navigation planning. You are planning the navigation of multiple robots in an indoor environment, equipped with three specialized modules to help analyze and understand the connections between current semantic maps and navigation objectives:
(1) Scene image description module: This module records the current and historical navigation scene image description information, and can query any content related to the historical navigation image content. This module is especially useful when you don't know the scene information in the semantic map.
(2) Scene object detection module: This module identifies and locates the objects in the image, and provides the boundaries and confidence of these objects. This module is especially useful when you don't know the information about objects in a semantic map.
(3) Scene exploration and analysis module: This module records the exploration possibility of current and historical navigation scenes. This module is especially useful when you are not clear about the exploration possibilities of semantic graphs.
Your task is: based on the capabilities of each module, assign specific tasks as needed to gather the additional information needed to accurately answer the question.

Your output format is SIMPLY as follows **WITHOUT ANY OTHER WORDS**:

Tasks of the module:
(1) Scene image description module: [If necessary, please return Yes. Otherwise, return No.]
(2) Scene object detection module: [If necessary, please return Yes. Otherwise, return No.]
(3) Scenario exploration analysis module: [If required, please raise any specific questions you have about the need for further exploration of the agent that require more in-depth visual analysis. Otherwise return to No.]

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

Module_Decision_Prompt = '''
You are an advanced semantic understanding agent, and you need to focus on the semantic information of the scene image on the right. Your task is: 

'''

Single_Agent_Decision_Prompt = '''
You are a robot navigating in an indoor environment. You have access to an image of the current timestep's scene and a top-down semantic map. 

The right part of the image is the scene image of the current timestep. 

The left part of the image is the corresponding top-down semantic map. The black dots and corresponding uppercase letters represent the Frontiers awaiting your exploration, with their coordinates and pixel counts shown below: 
{FRONTIERS_RESULTS}
The arrow is your location: [[Coordinates: {CUR_LOCATION}].
The line behind you represent your historical movement path. 

Your navigation goal is to find the {TARGET}. You need to comprehensively consider the relevance of the scene image and the top-down semantic map, and choose the navigation frontier for the next timestep based on their relationship with your navigation goal. You need to select the Frontier corresponding to the black capital letter [A, B, C, D] and answer the reason for your choice.

Your output format is as follows: 

My choice: 
[You need to select the Frontier corresponding to the black capital letter [A, B, C, D]]

The reasons why I choose this Frontier : 
[(1) Consider the pixel counting area size. You need to consider the front pixel count in relation to the navigation target. Usually larger exploration objects such as beds and sofas have larger pixel counts. 
(2) Consider the proximity and accessibility of the frontier. You need to consider the proximity and accessibility of the front and your location. Frontiers that are closer and without barriers tend to have a higher exploration priority. 
(3) Consider the objects in the scene. Use your knowledge of the location of typical objects (e.g., the bed in your bedroom, the TV in your living room) to assess the likelihood of finding the target object.
(4) Consider possible potential rooms in the scene. Areas in the scene image may lead to different rooms, and navigation targets may be located in different rooms. For example, the appearance of a door in the image may indicate that the target object may be located in a potential room behind the door, which makes the corresponding frontier worth exploring.]
'''

Meta_Agent_Decision_Prompt = '''
You are a knowledgeable and skilled expert in indoor navigation planning. You are planning the navigation of multiple robots in an indoor environment, equipped with three specialized modules to help analyze and understand the connections between current semantic maps and navigation objectives:
(1) Scene image description module: This module records the current and historical navigation scene image description information, and can query any content related to the historical navigation image content. This module is especially useful when you don't know the scene information in the semantic map.
(2) Scene object detection module: This module identifies and locates the objects in the image, and provides the boundaries and confidence of these objects. This module is especially useful when you don't know the information about objects in a semantic map.
(3) Scene exploration and analysis module: This module records the exploration possibility of current and historical navigation scenes. This module is especially useful when you are not clear about the exploration possibilities of semantic graphs.
Your task is: based on the capabilities of each module, assign specific tasks as needed to gather the additional information needed to accurately answer the question.

Your output format is as follows:

Tasks of the module:
(1) Scene image description module: [If necessary, please return Yes. Otherwise, return No.]
(2) Scene object detection module: [If necessary, please return Yes. Otherwise, return No.]
(3) Scenario exploration analysis module: [If required, please raise any specific questions you have about the need for further exploration of the agent that require more in-depth visual analysis. Otherwise return to No.]

Make sure your answers fit into this format, using available modules or appropriate direct analysis to systematically address the question.

**Here are some examples:**
''' 
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

def form_prompt_for_PerceptionVLM(target, objs, agents_seg_list) -> str:

    object_detection = ''
    for item in objs:
        name, confidence, coords = item
        coord_pairs = coords[0].split(',')
        coord1 = coord_pairs[0], coord_pairs[1]
        coord2 = coord_pairs[2], coord_pairs[3]
        detection = f"{name}: {confidence}, <({int(float(coord1[0]))}, {int(float(coord1[1]))}), ({int(float(coord2[0]))}, {int(float(coord2[1]))})>\n"
        object_detection += detection

    semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in agents_seg_list.items()]) + "\n"

    Perception_Template = """
Target of navigation: {TARGET}

Scene object (object detection): 
{OBJECT_DETECTION}
Scene object (semantic segmentation):
{SEMANTIC_SEGMENTATION}
Are you confident that the robot is worth exploring in this scenario? You only need to answer "Yes" or "No".
"""
    User_Prompt = Perception_Template.format(
                        TARGET = target,
                        OBJECT_DETECTION = object_detection,
                        SEMANTIC_SEGMENTATION = semantic_segmentation
                    )
    User_Prompt = Perception_System_Prompt + User_Prompt



    return User_Prompt

def form_prompt_for_DecisionVLM_single(target, cur_location, Frontier_list) -> str:

    def convert_entry(entry):
        # 从字符串中提取坐标和像素计数
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}, Pixel Count: {number_str}]'

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
    
    User_Prompt = Single_Agent_Decision_Prompt.format(
        TARGET = target,
        CUR_LOCATION = cur_location,
        FRONTIERS_RESULTS = Frontiers_results
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

    task_prompt_analysis = '''
Please analyze the following elements based on the image: 1. the spatial layout and passable areas; 2. any potential obstacles; and 3. identify which areas may contain critical information or objects based on the current mission objectives {TARGET}. In conjunction with the above analysis, assess whether the areas in the image are worth exploring further and give your reasons for your recommendation.

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
(1) scene image description: [Input your scene image description]
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + end_prompt
    elif is_detection and not is_description and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image object detection: [Input your scene image object detection]
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection)
    elif is_analysis and not is_description and not is_detection:
        end_prompt = '''
Your output format is as follows:
(1) rationale for scene exploration: [Input your rationale for scene exploration]
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_analysis.format(TARGET=target)
    elif is_description and is_detection and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: [Input your scene image description]
(2) scene image object detection: [Input your scene image object detection]
'''     
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + end_prompt
    elif is_description and is_analysis and not is_detection:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: [Input your scene image description]
(2) rationale for scene exploration: [Input your rationale for scene exploration]
'''   
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    elif is_detection and not is_analysis and not is_description:    
        end_prompt = '''
Your output format is as follows:
(1) scene image object detection: [Input your scene image object detection]
(2) rationale for scene exploration: [Input your rationale for scene exploration]
'''   
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + \
            '(2) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    elif is_detection and is_analysis and is_description:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: [Input your scene image description]
(2) scene image object detection: [Input your scene image object detection]
(3) rationale for scene exploration: [Input your rationale for scene exploration]
'''     
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + \
                '(3) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    else:
        ### 跳过Module Prompt这一阶段
        Use_prompt = None
    
    return Use_prompt    




def form_prompt_for_DecisionVLM_Meta() -> str:
    return Meta_Agent_Decision_Prompt





def contains_yes_or_no(VLM_Pred: str) -> str:
    """
    检查模型B的输出字符串是否包含 'Yes' 或 'No'。

    参数:
    - model_b_output: 字符串 模型B的输出

    返回:
    - 如果包含 "Yes"，返回 "Yes"；如果包含 "No"，返回 "No"；如果都不包含，返回 "Neither"
    """
    if "Yes" in VLM_Pred:
        return "Yes"
    elif "No" in VLM_Pred:
        return "No"
    else:
        return "Neither"

def Perception_weight_decision(VLM_Rel: list, VLM_Pred: str) -> str:
    """
    根据模型A的输出权重和模型B的输出 进行加权决策 输出最终的决策结果。
    
    参数:
    - model_a_output: numpy 数组 模型A的输出 形如 [x, y]，其中 x 是 "Yes" 的权重 y 是 "No" 的权重。
    - model_b_output: 字符串 模型B的输出 只可能是 "Yes" 或 "No"。
    
    返回:
    - 最终加权后的决策结果，字符串，"Yes" 或 "No"。
    """
    b_decision = contains_yes_or_no(VLM_Pred)
    if b_decision == "Neither":
        return b_decision
    
    # 从模型A的输出中提取权重
    x, y = VLM_Rel
    
    if b_decision == "Yes":
        weighted_yes_prob = x
        weighted_no_prob = y * (1 - x)
    else:  # b_decision == "No"
        weighted_yes_prob = x * (1 - y)
        weighted_no_prob = y
    
    # 对权重进行归一化处理
    total_prob = weighted_yes_prob + weighted_no_prob
    if total_prob == 0:  # 避免划分零的问题
        return b_decision
    
    weighted_yes_prob /= total_prob
    weighted_no_prob /= total_prob
    
    # 最终决策
    return weighted_yes_prob, weighted_no_prob
